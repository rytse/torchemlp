from typing import Union, Optional, List
from abc import ABC
import random

import torch
import functorch

from torchemlp.ops import (
    LinearOperator,
    I,
    LazyKron,
    LazyKronsum,
    LazyShift,
    LazyPerm,
    Rot90,
)


def rel_err(A: LinearOperator, B: LinearOperator, epsilon: float = 1e-6):
    mad = torch.mean(torch.abs(A.dense - B.dense))  # mean abs diff
    ama = torch.mean(torch.abs(A.dense))  # a mean abs
    bma = torch.mean(torch.abs(B.dense))  # b mean abs
    return mad / (ama + bma + epsilon)


class Group(ABC):
    """
    Abstract base class for groups
    """

    # Dimensionality of representation
    d: Optional[int] = None

    # Continuous generators
    lie_algebra: List[LinearOperator] = []

    # Discrete generators
    discrete_generators: List[LinearOperator] = []

    # Sampling scale noise
    z_scale: float = 1

    # Flags for simplifying computation
    is_orthogonal: Optional[bool] = None
    is_permutation: Optional[bool] = None

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

        # Fill in self.d
        if self.d is None:
            if len(self.lie_algebra) > 0:
                self.d = self.lie_algebra[0].shape[-1]
            elif len(self.discrete_generators) > 0:
                self.d = self.discrete_generators[0].shape[-1]
            else:
                raise NotImplementedError("No generators found")

        """
        # Fill in missing generators
        if len(self.lie_algebra) == 0:
            self.lie_algebra = [MatrixLinearOperator(torch.zeros((0, self.d, self.d))]
        if len(self.discrete_generators) == 0:
            self.discrete_generators = [MatrixLinearOperator(torch.zeros((0, self.d, self.d)))]
        """

        # Check orthogonal flag
        if self.is_permutation:
            self.is_orthogonal = True
        if (
            len(self.lie_algebra) > 0
            and rel_err(self.lie_algebra.H, self.lie_algebra) < epsilon
            or len(self.discrete_generators) > 0
            and rel_err(self.discrete_generators.H, self.discrete_generators) < epsilon
        ):
            self.is_orthogonal = True

        # Check permutation flag
        if (
            self.is_orthogonal
            and len(self.lie_algebra) == 0
            and len(self.discrete_generators) > 0
        ):
            h_dense = torch.cat([h.dense for h in self.discrete_generators], dim=0)
            if torch.all((h_dense == 1).sum(-1) == 1):
                self.is_permutation = True

    def samples(self, N):
        """
        Draw N samples from the group (not necessarily Haar measure)
        """

        # Basis of continuous generators
        if self.lie_algebra is not NotImplemented and len(self.lie_algebra) > 0:
            A = self.lie_algebra
        else:
            A = torch.zeros((0, self.d, self.d))

        # Basis of discrete generators
        if (
            self.discrete_generators is not NotImplemented
            and len(self.discrete_generators) > 0
        ):
            h = self.discrete_generators
        else:
            h = torch.zeros((0, self.d, self.d))

        # Sampling noise
        z = self.z_scale * torch.randn(N, A.shape[0])  # continous samples
        ks = torch.randint(-5, 5, size=(N, h.shape[0], 3))  # discrete samples

        return self.__class__.noise2samples(z, ks, A, h)

    def sample(self):
        """
        Draw a single sample from the group (not necessarily Haar measure)
        """
        return self.samples(1)[0]

    @staticmethod
    def noise2sample(z, ks, A, h):
        """
        JITed method for drawing samples from continuous generators A and discrete generators h
        given pre-sampled noise
        """

        # Output group element that will be mutated to its final state via action by the
        # continuous and discrete group generators
        g = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)

        # If there are continuous generators, take exp(z * A)
        if A.shape[0]:
            zA = torch.sum(z[:, None, None] * A, dim=0)
            g = g @ torch.matrix_exp(zA)

        # If there are discrete generators, take g @ [h^k]
        # TODO see if we can vectorize this
        M, K = ks.shape
        if M != 0:
            for k in range(K):
                for i in random.shuffle(range(M)):
                    g = g @ torch.matrix_power(h[i], ks[i, k])

        return g

    @staticmethod
    def noise2samples(zs, ks, A, h):
        functorch.vmap(Group.noise2sample, (0, 0, None, None), 0)(zs, ks, A, h)

    def get_num_constraints(self):
        """
        Get the number of constraints of the group (i.e. the total number of basis elements)
        """
        return len(self.lie_algebra) + len(self.discrete_generators)

    def __str__(self):
        return repr(self)

    def __repr__(self):
        outstr = f"{self.__class__}"
        if self.args:
            outstr += "(" + "".join(map(repr, self.args)) + ")"
        return outstr

    def __eq__(self, G2):
        """
        Check if two groups are equal
        """
        # TODO check using spans of generators
        return repr(self) == repr(G2)

    def __hash__(self):
        return hash(repr(self))

    def __lt__(self, G2):
        return hash(self) < hash(G2)

    def __mul__(self, G2):
        return DirectProduct(self, G2)


class Trivial(Group):
    """
    The trivial group
    """

    def __init__(self, n):
        self.d = n
        super().__init__(n)


class SO(Group):
    """
    The special orthogonal group SO(n)
    """

    def __init__(self, n):
        n_gen = (n * (n - 1)) // 2
        self.lie_algebra = torch.zeros((n_gen, n, n))
        k = 0
        for i in range(n):
            for j in range(i):
                self.lie_algebra[k, i, j] = 1
                self.lie_algebra[k, j, i] = -1
                k += 1
        super().__init__(n)


class O(SO):
    """
    The orthogonal group O(n)
    """

    def __init__(self, n):
        self.discrete_generators = torch.eye(n)[None]
        self.discrete_generators[0, 0, 0] = -1
        super().__init__(n)


class C(Group):
    """
    The cyclic group C_k in 2D
    """

    def __init__(self, k):
        theta = 2 * torch.pi / k
        self.discrete_generators = torch.zeros((1, 2, 2))
        self.discrete_generators[0, :, :] = torch.tensor(
            [
                [torch.cos(theta), torch.sin(theta)],
                [-torch.sin(theta), torch.cos(theta)],
            ]
        )
        super().__init__(k)


class D(C):
    """
    The dihedral group D_k in 2D
    """

    def __init__(self, k):
        super().__init__(k)
        self.discrete_generators = torch.cat(
            self.discrete_generators, torch.tensor([[[-1, 0], [0, 1]]]), dim=0
        )


class Scaling(Group):
    """
    The scaling group
    """

    def __init__(self, n):
        self.lie_algebra = torch.eye(n)[None]
        super().__init__(n)


class Parity(Group):
    """
    The spacial parity group in (1+3)D
    """

    discrete_generators = -torch.eye(4)[None]
    discrete_generators[0, 0, 0] = 1


class TimeReversal(Group):
    """
    The time reversal group in (1+3)D
    """

    discrete_generators = -torch.eye(4)[None]
    discrete_generators[0, 0, 0] = -1


class SO13p(Group):
    """
    The component of the Lorentz group connected to the identity, SO+(1, 3)
    """

    lie_algebra = torch.zeros((6, 4, 4))
    lie_algebra[3:, 1:, 1:] = SO(3).lie_algebra

    for i in range(3):
        lie_algebra[i, 1 + i, 0] = 1.0
        lie_algebra[i, 0, 1 + i] = 1.0

    z_scale = torch.tensor([0.3, 0.3, 0.3, 1.0, 1.0, 1.0])


class SO13(SO13p):
    """
    The special Lorentz group, SO(1, 3)
    """

    discrete_generators = -torch.eye(4)[None]


class O13(SO13):
    """
    The Lorentz group, O(1, 3)
    """

    discrete_generators = torch.eye(4)[None] + torch.zeros((2, 1, 1))
    discrete_generators[0] *= -1
    discrete_generators[1, 0, 0] = -1


class Lorentz(O13):
    """
    Alias for the Lorentz group, O(1, 3)
    """

    pass


class SO11p(Group):
    """
    The component of O(1, 1) connected to the identity, SO+(1, 1)
    """

    lie_algebra = torch.tensor([[0.0, 1.0], [1.0, 0.0]])[None]


class O11(SO11p):
    """
    The Lorentz group O(1, 1)
    """

    discrete_generators = torch.eye(2)[None] + torch.zeros((2, 1, 1))
    discrete_generators[0] *= -1
    discrete_generators[1, 0, 0] = -1


class Sp(Group):
    """
    The symplectic group Sp(m) in 2m dimensions
    """

    def __init__(self, m):
        self.lie_algebra = torch.zeros((m * (2 * m + 1), 2 * m, 2 * m))
        self.m = m

        k = 0
        for i in range(m):
            for j in range(m):
                self.lie_algebra[k, i, j] = 1
                self.lie_algebra[k, m + j, m + i] = -1
                k += 1
        for i in range(m):
            for j in range(i + 1):
                self.lie_algebra[k, m + i, j] = 1
                self.lie_algebra[k, m + j, i] = 1
                k += 1
                self.lie_algebra[k, i, m + j] = 1
                self.lie_algebra[k, j, m + i] = 1
                k += 1

        super().__init__(m)


class Z(Group):
    """
    The cyclic group Z_n (discrete granslation group)
    """

    def __init__(self, n):
        self.discrete_generators = [LazyShift(n)]
        super().__init__(n)


class S(Group):
    """
    The permutation group S_n
    """

    def __init__(self, n):
        perms = torch.arange(n)[None] + torch.zeros((n - 1, 1)).astype(int)
        perms[:, 0] = torch.arange(1, n)
        perms[torch.arange(n - 1), -torch.arange(1, n)[None]] = 0
        self.discrete_generators = [LazyPerm(perm) for perm in perms]
        super().__init__(n)


class SL(Group):
    """
    The special linear group SL(n)
    """

    def __init__(self, n):
        self.lie_algebra = torch.zeros((n * n - 1, n, n))
        k = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    self.lie_algebra[k, i, j] = 1
                    k += 1
        for l in range(n - 1):
            self.lie_algebra[k, l, l] = 1
            self.lie_algebra[k, -1, -1] = -1
            k += 1
        super().__init__(n)


class GL(Group):
    """
    The general linear group GL(n)
    """

    def __init__(self, n):
        self.lie_algebra = torch.zeros((n * n, n, n))
        k = 0
        for i in range(n):
            for j in range(n):
                self.lie_algebra[k, i, j] = 1
                k += 1
        super().__init__(n)


class U(Group):
    """
    The unitary group U(N), complex
    """

    def __init__(self, n):
        lie_algebra_real = torch.zeros((n * n, n, n))
        lie_algebra_imag = torch.zeros((n * n, n, n))

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

        self.lie_algebra = lie_algebra_real + 1.0j * lie_algebra_imag
        super().__init__(n)


class SU(Group):
    """
    The special unitary group SU(n), complex
    """

    def __init__(self, n):
        if n == 1:
            return Trivial(1)

        lie_algebra_real = torch.zeros((n * n, n, n))
        lie_algebra_imag = torch.zeros((n * n, n, n))

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

        self.lie_algebra = lie_algebra_real + 1.0j * lie_algebra_imag
        super().__init__(n)


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

    def __init__(self, k, n):
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

    def __init__(self, G, d, slice):
        self.lie_algebra = torch.zeros((G.lie_algebra.shape[0], d, d))
        self.discrete_generators = torch.zeros((G.discrete_generators.shape[0], d, d))
        self.discrete_generators += torch.eye(d)

        self.lie_algebra[:, slice, slice] = G.lie_algebra
        self.discrete_generators[:, slice, slice] = G.discrete_generators

        self.name = f"{G}_R{d}"
        super().__init__()

    def __repr__(self):
        return self.name


def SO2eR3():
    """
    SO(2) embedded in R^3 with rotations about the z axis
    """
    return Embed(SO(2), 3, slice(2))


def O2eR3():
    """
    O(2) embedded in R^3 with rotations about the z axis
    """
    return Embed(O(2), 3, slice(2))


def DkeR3(k):
    """
    Dihedral D(k) embedded in R^3 with rotations about the z axis
    """
    return Embed(D(k), 3, slice(2))


class DirectProduct(Group):
    """
    A new group equivalent to the direct product of two input groups
    """

    def __init__(self, G1, G2):
        I1, I2 = I(G1.d), I(G2.d)

        G1k = [LazyKron([M1, I2]) for M1 in G1.discrete_generators]
        G2k = [LazyKron([I1, M2]) for M2 in G2.discrete_generators]

        G1ks = [LazyKronsum([A1, 0 * I2]) for A1 in G1.lie_algebra]
        G2ks = [LazyKronsum([0 * I1, A2]) for A2 in G2.lie_algebra]

        self.discrete_generators = torch.tensor(G1k + G2k)
        self.lie_algebra = torch.tensor(G1k + G2k)

        self.names = (repr(G1), repr(G2))
        super().__init__()

    def __repr__(self):
        return f"{self.names[0]}x{self.names[1]}"


class WreathProduct(Group):
    def __init__(self, G1, G2):
        raise NotImplementedError


class SemiDirectProduct(Group):
    def __init__(self, G1, G2, phi):
        raise NotImplementedError
