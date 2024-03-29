import random
from abc import ABC
from typing import Callable, cast

import torch

from torchemlp.ops import (
    I,
    LazyKron,
    LazyKronsum,
    LazyPerm,
    LazyShift,
    LinearOperator,
    MatrixLinearOperator,
    Rot90,
)
from torchemlp.utils import DEFAULT_DEVICE, rel_rms_diff

# Type aliases for group elements, which will always be dense matrices
GroupElem = torch.Tensor
GroupElems = torch.Tensor


class Group(ABC):
    """
    Abstract base class for groups
    """

    # Dimensionality of representation
    d: int = -1

    # Continuous generators
    lie_algebra: list[LinearOperator] = []

    # Discrete generators
    discrete_generators: list[LinearOperator] = []

    # Sampling scale noise
    z_scale: torch.Tensor | float = 1.0

    # Flags for simplifying computation
    is_orthogonal: bool = False
    is_permutation: bool = False

    def __init__(self, *args, **kwargs):
        """
        Fill in flags not specified in the specific implementation's __init__.

        Be sure to call super() after you specify self.lie_algebra,
        self.discrete_generators, or self.d, otherwise this will fail!
        """

        self.args = args
        self.kwargs = kwargs

        self.__name__ = repr(self)

        if "epsilon" in kwargs:
            epsilon = kwargs["epsilon"]
        else:
            epsilon = 1e-6

        if "device" in kwargs:
            self.device = kwargs["device"]
        else:
            self.device = DEFAULT_DEVICE

        # Fill in self.d
        if self.d == -1:
            if len(self.lie_algebra) > 0:
                self.d = self.lie_algebra[0].shape[-1]
            elif len(self.discrete_generators) > 0:
                self.d = self.discrete_generators[0].shape[-1]
            else:
                raise NotImplementedError("No generators found")

        # Check orthogonal flag
        if self.is_permutation:
            self.is_orthogonal = True
        if len(self.lie_algebra) > 0:
            A_dense = torch.stack([A.dense() for A in self.lie_algebra])
            if rel_rms_diff(-torch.transpose(A_dense, 2, 1), A_dense) < epsilon:
                self.is_orthogonal = True
        elif len(self.discrete_generators) > 0:
            h_dense = torch.stack([h.dense() for h in self.discrete_generators])
            if rel_rms_diff(-torch.transpose(-h_dense, 2, 1), h_dense) < epsilon:
                self.is_orthogonal = True

        # Check permutation flag
        if (
            self.is_orthogonal
            and len(self.lie_algebra) == 0
            and len(self.discrete_generators) > 0
        ):
            h_dense = torch.cat([h.dense() for h in self.discrete_generators], dim=0)
            if torch.all((h_dense == 1).sum(-1) == 1):
                self.is_permutation = True

    def dense_lie_algebra(self) -> torch.Tensor:
        return torch.stack([lg.dense() for lg in self.lie_algebra])

    def dense_discrete_generators(self) -> torch.Tensor:
        return torch.stack([dg.dense() for dg in self.discrete_generators])

    def sample(self) -> GroupElem:
        """
        Draw a single sample from the group (not necessarily Haar measure)
        """

        n_lie_bases = len(self.lie_algebra)
        n_discrete_gens = len(self.discrete_generators)

        # Basis of continuous generators
        if n_lie_bases > 0:
            A_dense = self.dense_lie_algebra()
        elif self.d != -1:
            A_dense = torch.zeros((0, self.d, self.d), device=self.device)
        else:
            raise NotImplementedError("No generators found")

        # Basis of discrete generators
        if n_discrete_gens > 0:
            h_dense = self.dense_discrete_generators()
        elif self.d != -1:
            h_dense = torch.zeros((0, self.d, self.d), device=self.device)
        else:
            raise NotImplementedError("No generators found")

        # Sampling noise
        z = self.z_scale * torch.randn(
            n_lie_bases, device=self.device
        )  # continous samples
        ks = torch.randint(
            -5, 5, size=(h_dense.shape[0], 3), device=self.device
        )  # discrete samples

        # Generate samples
        return self.__class__.noise2sample(z, ks, A_dense, h_dense)

    def samples(self, N) -> list[GroupElem]:
        return [self.sample() for _ in range(N)]  # TODO batch

    @staticmethod
    def noise2sample(
        z: torch.Tensor, ks: torch.Tensor, A_dense: torch.Tensor, h_dense: torch.Tensor
    ) -> GroupElem:
        """
        Method for drawing samples from the group by applying random discrete
        generators and exponentiating random Lie algebra basis vectors.

        Args:
            z: Direction in Lie algebra to travel in, i.e. exp(zA)
            ks: Number of times to apply each discrete generator
            A_dense: Lie algebra basis vectors as a dense tensor
            h_dense: Discrete generators as a dense tensor

        Returns:
            A sample from the group
        """

        # Output group element that will be mutated to its final state via action by the
        # continuous and discrete group generators
        g = torch.eye(A_dense.shape[-1], dtype=A_dense.dtype, device=A_dense.device)

        # If there are continuous generators, take exp(z * A)
        if A_dense.shape[0]:
            zA = torch.sum(z[:, None, None] * A_dense, dim=0)
            g = g @ torch.matrix_exp(zA)

        # If there are discrete generators, take g @ [h^k]
        # TODO see if we can vectorize this
        M, K = ks.shape
        if M is not None and M > 0:
            for k in range(K):
                for i in random.sample(range(M), M):
                    g = g @ torch.matrix_power(h_dense[i], int(ks[i, k]))
        return g

    def get_num_constraints(self) -> int:
        """
        Get the number of constraints of the group (i.e. the total number of basis elements)
        """
        return len(self.lie_algebra) + len(self.discrete_generators)

    def __str__(self) -> str:
        return repr(self)

    def __repr__(self) -> str:
        outstr = f"{self.__class__.__name__}"
        if self.args:
            outstr += "(" + "".join(map(repr, self.args)) + ")"
        return outstr

    def __eq__(self, G2: "Group") -> bool:
        """
        Check if two groups are equal
        """
        # TODO check using spans of generators
        return repr(self) == repr(G2)

    def __hash__(self) -> int:
        return hash(repr(self))

    def __lt__(self, G2: "Group") -> bool:
        return hash(self) < hash(G2)

    def __mul__(self, G2: "Group") -> "Group":
        return DirectProduct(self, G2)


class Trivial(Group):
    """
    The trivial group
    """

    def __init__(self, n: int):
        self.d = n
        super().__init__(n)


class SO(Group):
    """
    The special orthogonal group SO(n)
    """

    def __init__(self, n: int, device: torch.device = DEFAULT_DEVICE):
        n_gen = (n * (n - 1)) // 2
        A_dense = torch.zeros((n_gen, n, n), device=device)
        k = 0
        for i in range(n):
            for j in range(i):
                A_dense[k, i, j] = 1
                A_dense[k, j, i] = -1
                k += 1
        self.lie_algebra = [MatrixLinearOperator(AM) for AM in A_dense]

        super().__init__(n, device=device)


class O(SO):
    """
    The orthogonal group O(n)
    """

    def __init__(self, n: int, device: torch.device = DEFAULT_DEVICE):
        h_dense = torch.eye(n, device=device)[None]
        h_dense[0, 0, 0] = -1
        self.discrete_generators = [MatrixLinearOperator(hM) for hM in h_dense]

        super().__init__(n, device=device)


class C(Group):
    """
    The cyclic group C_k in 2D
    """

    def __init__(self, k: int, device: torch.device = DEFAULT_DEVICE):
        theta = torch.tensor(2 * torch.pi / k, device=device)
        h_dense = torch.zeros((1, 2, 2), device=device)
        h_dense[0, :, :] = torch.tensor(
            [
                [torch.cos(theta), torch.sin(theta)],
                [-torch.sin(theta), torch.cos(theta)],
            ],
            device=device,
        )
        self.discrete_generators = [MatrixLinearOperator(hM) for hM in h_dense]

        super().__init__(k, device=device)


class D(C):
    """
    The dihedral group D_k in 2D
    """

    def __init__(self, k: int, device: torch.device = DEFAULT_DEVICE):
        super().__init__(k, device=device)
        self.discrete_generators += [
            MatrixLinearOperator(torch.tensor([[-1, 0], [0, 1]], device=device))
        ]


class Scaling(Group):
    """
    The scaling group
    """

    def __init__(self, n: int, device: torch.device = DEFAULT_DEVICE):
        A_dense = torch.eye(n, device=device)[None]
        self.lie_algebra = [MatrixLinearOperator(AM) for AM in A_dense]
        super().__init__(n, device=device)


class Parity(Group):
    """
    The spacial parity group in (1+3)D
    """

    def __init__(self, device: torch.device = DEFAULT_DEVICE):
        h_dense = -torch.eye(4, device=device)[None]
        h_dense[0, 0, 0] = 1
        self.discrete_generators = [MatrixLinearOperator(hM) for hM in h_dense]

        super().__init__((1, 3), device=device)


class TimeReversal(Group):
    """
    The time reversal group in (1+3)D
    """

    def __init__(self, device: torch.device = DEFAULT_DEVICE):
        h_dense = -torch.eye(4, device=device)[None]
        h_dense[0, 0, 0] = -1
        self.discrete_generators = [MatrixLinearOperator(hM) for hM in h_dense]

        super().__init__((1, 3), device=device)


class SO13p(Group):
    """
    The component of the Lorentz group connected to the identity, SO+(1, 3)
    """

    def __init__(self, device: torch.device = DEFAULT_DEVICE):
        A_dense = torch.zeros((6, 4, 4), device=device)
        A_dense[3:, 1:, 1:] = SO(3, device=device).dense_lie_algebra()

        for i in range(3):
            A_dense[i, 1 + i, 0] = 1.0
            A_dense[i, 0, 1 + i] = 1.0

        self.lie_algebra = [MatrixLinearOperator(AM) for AM in A_dense]
        self.z_scale = torch.tensor([0.3, 0.3, 0.3, 1.0, 1.0, 1.0], device=device)

        super().__init__((1, 3), device=device)


class SO13(SO13p):
    """
    The special Lorentz group, SO(1, 3)
    """

    def __init__(self, device: torch.device = DEFAULT_DEVICE):
        h_dense = -torch.eye(4, device=device)[None]
        self.discrete_generators = [MatrixLinearOperator(hM) for hM in h_dense]

        super().__init__(device=device)


class O13(SO13):
    """
    The Lorentz group, O(1, 3)
    """

    def __init__(self, device: torch.device = DEFAULT_DEVICE):
        h_dense = torch.eye(4, device=device)[None] + torch.zeros(
            (2, 1, 1), device=device
        )
        h_dense[0] *= -1
        h_dense[1, 0, 0] = -1
        self.discrete_generators = [MatrixLinearOperator(hM) for hM in h_dense]

        super().__init__(device=device)


class Lorentz(O13):
    """
    Alias for the Lorentz group, O(1, 3)
    """

    pass


class SO11p(Group):
    """
    The component of O(1, 1) connected to the identity, SO+(1, 1)
    """

    def __init__(self, device: torch.device = DEFAULT_DEVICE):
        A_dense = torch.tensor([[0.0, 1.0], [1.0, 0.0]], device=device)[None]
        self.lie_algebra = [MatrixLinearOperator(AM) for AM in A_dense]

        super().__init__((1, 1), device=device)


class O11(SO11p):
    """
    The Lorentz group O(1, 1)
    """

    def __init__(self, device: torch.device = DEFAULT_DEVICE):
        h_dense = torch.eye(2, device=device)[None] + torch.zeros(
            (2, 1, 1), device=device
        )
        h_dense[0] *= -1
        h_dense[1, 0, 0] = -1
        self.discrete_generators = [MatrixLinearOperator(hM) for hM in h_dense]

        super().__init__(device=device)


class Sp(Group):
    """
    The symplectic group Sp(m) in 2m dimensions
    """

    def __init__(self, m: int, device: torch.device = DEFAULT_DEVICE):
        A_dense = torch.zeros((m * (2 * m + 1), 2 * m, 2 * m), device=device)
        self.m = m

        k = 0
        for i in range(m):
            for j in range(m):
                A_dense[k, i, j] = 1
                A_dense[k, m + j, m + i] = -1
                k += 1
        for i in range(m):
            for j in range(i + 1):
                A_dense[k, m + i, j] = 1
                A_dense[k, m + j, i] = 1
                k += 1
                A_dense[k, i, m + j] = 1
                A_dense[k, j, m + i] = 1
                k += 1

        self.lie_algebra = [MatrixLinearOperator(AM) for AM in A_dense]

        super().__init__(m, device=device)


class Z(Group):
    """
    The cyclic group Z_n (discrete granslation group)
    """

    def __init__(self, n: int):
        self.discrete_generators = [LazyShift(n)]
        super().__init__(n)


class S(Group):
    """
    The permutation group S_n
    """

    def __init__(self, n: int, device: torch.device = DEFAULT_DEVICE):
        perms = torch.arange(n, dtype=torch.int64, device=device)[None] + torch.zeros(
            (n - 1, 1), dtype=torch.int64, device=device
        )
        perms[:, 0] = torch.arange(1, n, dtype=torch.int64, device=device)
        perms[
            torch.arange(n - 1, dtype=torch.int64, device=device),
            torch.arange(1, n, dtype=torch.int64, device=device)[None],
        ] = 0
        self.discrete_generators = [LazyPerm(perm) for perm in perms]
        super().__init__(n, device=device)


class SL(Group):
    """
    The special linear group SL(n)
    """

    def __init__(self, n: int, device: torch.device = DEFAULT_DEVICE):
        A_dense = torch.zeros((n * n - 1, n, n), device=device)
        k = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    A_dense[k, i, j] = 1
                    k += 1
        for l in range(n - 1):
            A_dense[k, l, l] = 1
            A_dense[k, -1, -1] = -1
            k += 1

        self.lie_algebra = [MatrixLinearOperator(AM) for AM in A_dense]

        super().__init__(n, device=device)


class GL(Group):
    """
    The general linear group GL(n)
    """

    def __init__(self, n: int, device: torch.device = DEFAULT_DEVICE):
        A_dense = torch.zeros((n * n, n, n), device=device)
        k = 0
        for i in range(n):
            for j in range(n):
                A_dense[k, i, j] = 1
                k += 1

        self.lie_algebra = [MatrixLinearOperator(AM) for AM in A_dense]

        super().__init__(n, device=device)


class U(Group):
    """
    The unitary group U(N), complex
    """

    def __init__(self, n: int, device: torch.device = DEFAULT_DEVICE):
        lie_algebra_real = torch.zeros((n * n, n, n), device=device)
        lie_algebra_imag = torch.zeros((n * n, n, n), device=device)

        k = 0
        for i in range(n):
            for j in range(i):
                # Antisymmetric real generators
                lie_algebra_real[k, i, j] = 1
                lie_algebra_imag[k, j, i] = -1
                k += 1

                # Symmetric imaginary generators
                lie_algebra_imag[k, i, j] = 1
                lie_algebra_real[k, j, i] = -1
                k += 1

        # Diagonal imaginary generators
        for i in range(n):
            lie_algebra_imag[k, i, i] = 1
            k += 1

        A_dense = lie_algebra_real + 1.0j * lie_algebra_imag
        self.lie_algebra = [MatrixLinearOperator(AM) for AM in A_dense]

        super().__init__(n, device=device)


class SU(Group):
    """
    The special unitary group SU(n), complex
    """

    def __init__(self, n: int, device: torch.device = DEFAULT_DEVICE):
        if n == 1:
            return Trivial(1)

        lie_algebra_real = torch.zeros((n * n, n, n), device=device)
        lie_algebra_imag = torch.zeros((n * n, n, n), device=device)

        k = 0
        for i in range(n):
            for j in range(i):
                # Antisymmetric real generators
                lie_algebra_real[k, i, j] = 1
                lie_algebra_imag[k, j, i] = -1
                k += 1

                # Symmetric imaginary generators
                lie_algebra_imag[k, i, j] = 1
                lie_algebra_real[k, j, i] = -1
                k += 1

        # Diagonal traceless imaginary generators
        for i in range(n - 1):
            lie_algebra_imag[k, i, i] = 1
            for j in range(n):
                if i != j:
                    lie_algebra_imag[k, j, j] = -1.0 / (n - 1.0)
            k += 1

        A_dense = lie_algebra_real + 1.0j * lie_algebra_imag
        self.lie_algebra = [MatrixLinearOperator(AM) for AM in A_dense]

        super().__init__(n, device=device)


class Cube(Group):
    """
    Discrete version of SO(3) including all 90 degree rotations in 3d space,
    with 6-dim representation on the faces of a cube
    """

    def __init__(self):
        raise NotImplementedError


class RubiksCube(Group):
    """
    The Rubiks cube group G<S_48 consisting of all valid 3x3 Rubik's cube
    transformations generated by a quarter turn about each of the faces.
    """

    def __init__(self):
        raise NotImplementedError


class ZksZnxZn(Group):
    """
    The ℤₖ⋉(ℤₙ×ℤₙ) group for translation in x, y and rotation with the discrete
    90 degree rotations (k=4) or 180 degree rotations (k=2). One of the
    original GCNN groups.
    """

    def __init__(self, k: int, n: int):
        Zn = Z(n)
        Zk = Z(k)

        nshift = Zn.discrete_generators[0]
        kshift = Zk.discrete_generators[0]

        In = I(n)
        Ik = I(k)

        assert k in [2, 4]

        self.discrete_generators = [
            LazyKron([Ik, nshift, In]),
            LazyKron([Ik, In, nshift]),
            LazyKron([kshift, Rot90(n, 4 // k)]),
        ]

        super().__init__(k, n)


class Embed(Group):
    """
    A new group equivalent to an input group embedded in a larger vector space.
    """

    def __init__(
        self,
        G: Group,
        d: int,
        slice: slice | int = slice(None),
        device: torch.device = DEFAULT_DEVICE,
    ):
        A_dense = torch.zeros((len(G.lie_algebra), d, d), device=device)
        h_dense = torch.zeros((len(G.discrete_generators), d, d), device=device)
        h_dense += torch.eye(d, device=device)

        A_dense[:, slice, slice] = G.dense_lie_algebra()
        h_dense[:, slice, slice] = G.dense_discrete_generators()

        self.lie_algebra = [MatrixLinearOperator(AM) for AM in A_dense]
        self.discrete_generators = [MatrixLinearOperator(AM) for AM in h_dense]

        self.name = f"{G}_R{d}"
        super().__init__(d, device=device)

    def __repr__(self) -> str:
        return self.name


def SO2eR3(device: torch.device = DEFAULT_DEVICE):
    """
    SO(2) embedded in R^3 with rotations about the z axis
    """
    return Embed(SO(2), 3, slice(2), device=device)


def O2eR3(device: torch.device = DEFAULT_DEVICE):
    """
    O(2) embedded in R^3 with rotations about the z axis
    """
    return Embed(O(2), 3, slice(2), device=device)


def DkeR3(k, device: torch.device = DEFAULT_DEVICE):
    """
    Dihedral D(k) embedded in R^3 with rotations about the z axis
    """
    return Embed(D(k), 3, slice(2), device=device)


class DirectProduct(Group):
    """
    A new group equivalent to the direct product of two input groups
    """

    def __init__(self, G1: Group, G2: Group):
        # TODO resolve the actual type
        I1, I2 = I(G1.d), I(G2.d)

        G1k = [LazyKron([M1, I2]) for M1 in G1.discrete_generators]
        G2k = [LazyKron([I1, M2]) for M2 in G2.discrete_generators]

        G1ks = [LazyKronsum([A1, 0 * I2]) for A1 in G1.lie_algebra]
        G2ks = [LazyKronsum([0 * I1, A2]) for A2 in G2.lie_algebra]

        self.discrete_generators = cast(list[LinearOperator], G1k + G2k)
        self.lie_algebra = cast(list[LinearOperator], G1ks + G2ks)

        self.names = (repr(G1), repr(G2))
        super().__init__()

    def __repr__(self) -> str:
        return f"{self.names[0]}x{self.names[1]}"


class WreathProduct(Group):
    def __init__(self, G1: Group, G2: Group):
        raise NotImplementedError


class SemiDirectProduct(Group):
    def __init__(self, G1: Group, G2: Group, phi: Callable):
        raise NotImplementedError
