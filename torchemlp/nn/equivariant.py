from math import sqrt
from typing import Callable
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F

import functorch

from torchemlp.utils import lambertW, binom, rel_rms_diff
from torchemlp.groups import Group
from torchemlp.reps import Rep, T, SumRep, Scalar, ScalarRep, Zero, bilinear_weights


class EquivariantLinear(nn.Module):
    """
    Equivariant linear layer from repin to repout.
    """

    def __init__(self, repin: Rep, repout: Rep):
        super(EquivariantLinear, self).__init__()

        self.n_in, self.n_out = repin.size, repout.size
        self.rep_W = repin >> repout

        self.Pw = self.rep_W.equivariant_projector()
        self.Pb = repout.equivariant_projector()

        self.b = nn.Parameter(
            torch.rand(
                (self.n_out,),
            )
            / sqrt(self.n_out)
        )
        w_init = nn.init.orthogonal_(torch.rand((self.n_out, self.n_in)))
        self.w = nn.Parameter(w_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.Pw @ self.w
        b = self.Pb @ self.b
        return x @ W.T + b


class EquivariantBiLinear(nn.Module):
    """
    Equivariant bilinear layer from repin to repout.
    """

    def __init__(self, repin: SumRep, repout: Rep):
        super(EquivariantBiLinear, self).__init__()
        self.W_dim, self.W_proj = bilinear_weights(repin, repout)  # TODO jit
        self.W = nn.Parameter(
            torch.normal(
                torch.zeros(
                    (self.W_dim,),
                ),
                torch.ones(
                    (self.W_dim,),
                ),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W_proj = self.W_proj(self.W, x)
        return 0.1 * (W_proj @ x[..., None])[..., 0]


class GatedNonlinearity(nn.Module):
    """
    Gated nonlinearity for applying activation to each added scalar gate of a
    rep. Must call add_gate(ch_rep) on the input rep to this class.
    """

    def __init__(self, rep: Rep, act: Callable = F.sigmoid):
        super(GatedNonlinearity, self).__init__()
        self.rep = rep
        self.act = act

        self.gate_indices = GatedNonlinearity.get_gate_indices(self.rep)
        self.n_gates = self.rep.size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_scalars = x[..., self.gate_indices]
        return self.act(gate_scalars) * x[..., : self.n_gates]

    @staticmethod
    def gated(ch_rep: Rep) -> Rep:
        """
        Generate a new Rep that has an additional scalar 'gate' for each non-scalar
        and non-regular Rep that it is composed of. To be used as the output for
        linear and bilinear layers directly before a GatedNonlinearity to produce
        its scalar gates.
        """
        match ch_rep:
            case SumRep():
                gates = []
                for rep in ch_rep:
                    if not isinstance(rep, ScalarRep) and not rep.is_permutation:
                        gates.append(Scalar(rep.G))
                return ch_rep + sum(gates)
            case _:
                if ch_rep.is_permutation:
                    return ch_rep
                else:
                    return ch_rep + Scalar(ch_rep.G)

    @staticmethod
    @lru_cache(maxsize=None)  # TODO optimize torch -> CPU
    def get_gate_indices(ch_rep: Rep) -> torch.Tensor:
        """
        Determine which indices of a Rep's output, if any, are gates.
        """
        channels = ch_rep.size
        indices = torch.arange(channels)

        match ch_rep:
            case SumRep():
                perm = ch_rep.perm
                num_nonscalars = 0
                i = 0
                for rep in ch_rep:
                    if not isinstance(rep, ScalarRep) and not rep.is_permutation:
                        indices[perm[i : i + rep.size]] = channels + num_nonscalars
                        num_nonscalars += 1
                    i += rep.size
                return indices

            case _:
                if ch_rep.is_permutation:
                    return indices
                else:
                    return (
                        torch.ones(
                            ch_rep.size,
                        )
                        * ch_rep.size
                    )


class EMLPBlock(nn.Module):
    """
    Basic building block of an EMLP. Composition of a G-linear operation, a
    bilinear operation, and gated nonlinearity.
    """

    def __init__(self, repin: Rep, repout: Rep):
        super(EMLPBlock, self).__init__()

        gated_in = GatedNonlinearity.gated(repin)
        gated_out = GatedNonlinearity.gated(repout)

        if not (isinstance(gated_in, SumRep) and isinstance(gated_out, SumRep)):
            raise ValueError("gated(repin) and gated(repout) must be SumReps")

        self.linear = EquivariantLinear(repin, gated_out)
        self.bilinear = EquivariantBiLinear(gated_out, gated_out)
        self.nonlinear = GatedNonlinearity(repout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lin = self.linear(x)
        preact = self.bilinear(lin) + lin
        return self.nonlinear(preact)


class EMLP(nn.Module):
    """
    Equivariant Multi-Layer Perceptron.
    """

    def __init__(
        self,
        repin: Rep,
        repout: Rep,
        G: Group,
        ch: int | list[int | Rep] | Rep = 384,
        num_layers: int = 3,
    ):
        super(EMLP, self).__init__()

        self.G = G
        self.repin = repin(self.G)
        self.repout = repout(self.G)

        match ch:
            case int():
                middle_layers = num_layers * [EMLP.uniform_rep(ch, G)]
            case list():
                middle_layers = []
                for c in ch:
                    match c:
                        case int():
                            middle_layers.append(EMLP.uniform_rep(c, G))
                        case Rep():
                            middle_layers.append(c(G))
            case Rep():
                middle_layers = num_layers * [ch(G)]

        reps = [self.repin] + middle_layers

        layers = []
        for rin, rout in zip(reps, reps[1:]):
            layers.append(EMLPBlock(rin, rout))
        layers.append(EquivariantLinear(reps[-1], self.repout))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    @staticmethod
    def uniform_rep(ch: int, G: Group) -> Rep:
        """
        Heuristic method for allocating a given number of hidden layer channels
        into Rep tensor types. Attempts to distribute the channels evenly
        across the different tensor types.

        Args:
        ----------
            ch: total number of channels to allocate
            G: symmetry group

        Returns:
        ----------
            rep: direct sum representation of the Rep with dim(V) = ch
        """
        d = G.d
        Ns = torch.zeros((lambertW(ch, d) + 1,), dtype=torch.int64)
        while ch > 0:
            max_rank = lambertW(ch, d)
            Ns[: max_rank + 1] += torch.tensor(
                [d ** (max_rank - r) for r in range(max_rank + 1)], dtype=torch.int64
            )
            ch -= (max_rank + 1) * d**max_rank  # leftover

        rep = sum([EMLP.binomial_allocation(int(nr), r, G) for r, nr in enumerate(Ns)])
        match rep:
            case int():
                raise ValueError(
                    "Got a zero Rep: ch = {} and G = {} invalid".format(ch, G)
                )
            case Rep():
                can_rep, _ = rep.canonicalize()
                return can_rep

    @staticmethod
    def binomial_allocation(n: int, r: int, G: Group) -> Rep:
        """
        Allocate N tensors of total rank r=(p+q) into T(k, r-k) for k from 0
        to r to match the binomial distribution.
        """
        if n == 0:
            return Zero

        n_binoms: int = n // (2**r)
        n_leftover: int = n % (2**r)

        even_split = sum(
            [n_binoms * binom(r, k) * T(k, r - k, G) for k in range(r + 1)]
        )
        ps = torch.sum(torch.bernoulli(0.5 * torch.ones((r, n_leftover))), dim=0)
        ragged = sum([T(int(p), r - int(p), G) for p in ps])

        out = even_split + ragged
        match out:
            case int():
                raise ValueError(
                    "Got a zero Rep, n = {} and r = {} invalid".format(n, r)
                )
            case Rep():
                return out

    def equivariance_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Determine how far the EMLP is from equivariance for a given
        input-output pair (x, self(x)) by sampling from the symmetry group and
        checking different the group-acted outputs are from the original,
        untransformed output.
        """
        batch_size = x.shape[0]

        gs = self.G.samples(batch_size)

        act_in = torch.eye(x.shape[-1], device=x.device).repeat(batch_size, 1, 1)
        act_out = torch.eye(self(x).shape[-1], device=x.device).repeat(batch_size, 1, 1)

        # if len(self.G.discrete_generators) > 0:

        rho_gin = torch.stack([self.repin.rho_dense(g) for g in gs])
        rho_gout = torch.stack([self.repout.rho_dense(g) for g in gs])
        act_in = torch.bmm(rho_gin, act_in)
        act_out = torch.bmm(rho_gout, act_out)

        # if len(self.G.lie_algebra) > 0:
            # drho_gin = torch.stack([self.repin.drho_dense(g) for g in gs])
            # drho_gout = torch.stack([self.repout.drho_dense(g) for g in gs])
            # act_in = torch.bmm(drho_gin, act_in)
            # act_out = torch.bmm(drho_gout, act_out)

        if torch.count_nonzero(act_in) > 0:
            y1 = self((act_in @ x[..., None])[..., 0])
        else:
            y1 = self(x)

        if torch.count_nonzero(act_out) > 0:
            y2 = (act_out @ self(x)[..., None])[..., 0]
        else:
            y2 = self(x)

        return rel_rms_diff(y1, y2)
