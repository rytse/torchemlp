import torch

from tqdm import tqdm

from torchemlp.ops import LinearOperator


def orthogonal_complement(proj: torch.Tensor) -> torch.Tensor:
    """
    Compute the orthogonal complement of a projection
    """
    _, S, VH = torch.linalg.svd(proj, full_matrices=True)
    rank = torch.sum(S > 1e-5)
    return torch.conj(VH[rank:]).T


def krylov_constraint_solve(
    C: LinearOperator, device: torch.device, tol: float = 1e-5
) -> torch.Tensor:
    """
    Computes the solution basis Q for the linear constraint CQ = 0 and
    Q^T Q = I upt to the specified tolerance.
    """

    Q = torch.tensor([[0]], device=device)  # fallback value
    r = 5  # starting rank

    if C.shape[0] * r * 2 > 2e9:
        raise ValueError(
            f"Solutions for constraints {C.shape} too large to fit in memory"
        )

    found_rank = 5
    while found_rank == r:
        r *= 2
        if C.shape[0] * r > 2e9:
            print(
                f"Hit memory limits, switching to sample equivariant subspace of size {found_rank}"
            )
            break

        Q = krylov_constraint_solve_upto_r(C, device, r, tol)
        found_rank = Q.shape[-1]

    return Q


def krylov_constraint_solve_upto_r(
    C: LinearOperator,
    device: torch.device,
    r: int,
    tol: float = 1e-5,
    n_iters: int = 20_000,
    lr: float = 1e-2,
    momentum: float = 0.9,
) -> torch.Tensor:
    """
    Iterative routine to compute the solution basis tot he constraint CQ = 0
    and Q^T Q = I up to the rank r, with given tolerance. Uses gradient descent
    with momentum on the objective |CQ|^2, which provably converges at an
    exponential rate.
    """

    # Define optimization problem
    W = torch.randn(C.shape[-1], r, dtype=C.dtype, device=device) / torch.sqrt(
        torch.tensor(C.shape[-1], device=device)
    )
    W.requires_grad = True

    def loss_fn(W):
        return torch.sum(torch.abs(C @ W) ** 2) / 2.0

    optimizer = torch.optim.SGD([W], lr=lr, momentum=momentum)

    # Progress var
    pbar = tqdm(
        total=100,
        desc=f"Krylov Solving for Equivariant Subspace r<={r}",
        bar_format="{l_bar}{bar}| {n:.3g}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    )
    lstart = loss_fn(W)
    prog_val = 0

    # Perform optimization
    for i in range(n_iters):
        optimizer.zero_grad()
        loss = loss_fn(W)
        loss.backward()
        optimizer.step()

        # Update progress bar
        progress = float(
            100 * torch.log(loss / lstart) / torch.log(tol**2 / lstart) - prog_val
        )
        progress = max(progress, 0)
        progress = min(progress, 100 - prog_val)
        if progress > 0:
            prog_val += progress
            pbar.update(progress)

        # Check for convergence
        if torch.sqrt(loss) < tol:
            pbar.close()
            break
        if loss > 2e3 and i > 100:
            print(
                f"WARNING: Constraint solving diverged, t ry lowering learning rate {lr/3:0.2e}"
            )
            if lr < 1e-4:
                raise Exception(
                    "Constraint solving diverged even with small learning rate"
                )

            return krylov_constraint_solve_upto_r(C, device, r, tol, n_iters, lr / 3.0)
    else:
        raise Exception("Failed to converge.")

    # Orthogonalize the solution
    U, S, _ = torch.linalg.svd(W, full_matrices=False)
    rank = torch.sum(S > 10 * tol)
    Q = U[:, :rank]

    # Check if solution meets specified tolerance and SVD properties, printing
    # a warning if not.
    final_loss = loss_fn(Q)
    if final_loss > tol:
        print(f"WARNING: Failed to meet tolerance {tol}, final loss {final_loss}")
    scutoff = S[rank] if r > rank else 0
    if not (rank == 0 or scutoff < S[rank - 1] / 100):
        print("WARNING: Singular value gap too small")

    return Q


def sparsify_basis(
    Q: torch.Tensor, device: torch.device, lr: float = 1e-2
) -> torch.Tensor:
    """
    Attempt to sparsify a given basis by applying an orthogonal transformation

        W, Q' = QW

    where Q' is only 1s, 0s, and -1s.

    This method doesn't have the ssame convergence guarantees as the krylov
    solver, and will even fail silently. Use this for visualization, not for
    learning algorithms.
    """

    # Define optimization problem
    W = torch.rand(Q.shape[-1], Q.shape[-1], device=device)
    W, _ = torch.linalg.qr(W)
    W.requires_grad = True

    def loss_fn(W):
        return (
            torch.mean(torch.abs(Q @ W.T))
            + 0.1
            * torch.mean(torch.abs(W.T @ W - torch.eye(W.shape[0], device=device)))
            + 0.01 * torch.linalg.slogdet(W)[1] ** 2
        )

    optimizer = torch.optim.SGD([W], lr=lr)

    # Perform optimization
    for i in tqdm(range(3000), desc="Sparsifying basis"):
        optimizer.zero_grad()
        loss = loss_fn(W)
        loss.backward()
        optimizer.step()

        if loss > 1e2 and i > 100:
            print(
                f"Basis sparsification diverged, trying lower learning rate {lr/3:.2e}"
            )
            return sparsify_basis(Q, device, lr=lr / 3)

    # Decide what's a 0, +1, and -1
    Q = Q @ W.T
    Q[torch.abs(Q) < 1e-2] = 0
    Q[torch.abs(Q) > 1e-2] = torch.sgn(Q)[torch.abs(Q) > 1e-2]

    # Check convergence
    A = Q @ (1.0 + torch.arange(Q.shape[-1], device=device))
    n_pivots = len(torch.unique(torch.abs(A)))
    if n_pivots != Q.shape[-1] and n_pivots != Q.shape[-1] + 1:
        print(f"Basis elems did not seperate, foun only {n_pivots}/{Q.shape[-1] + 1}")

    return Q
