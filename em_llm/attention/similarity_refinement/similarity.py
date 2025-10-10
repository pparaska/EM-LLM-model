import torch
from typing import List, Sequence, Tuple, Union


def modularity(
    A: torch.Tensor,
    communities: Sequence[Sequence[int]],
) -> torch.Tensor:
    """
    Compute Newman–Girvan modularity for a given partition of nodes.

    The graph is represented by a (possibly batched) similarity/adjacency matrix A.
    For each batch item (e.g., layer/head) this returns a scalar modularity value.

    Parameters
    ----------
    A : torch.Tensor
        Similarity/adjacency matrix with shape (..., N, N). The leading dimensions
        (if any) are treated as batch (e.g., layer, head).
    communities : Sequence[Sequence[int]]
        A list of communities, where each community is a list of node indices.

    Returns
    -------
    torch.Tensor
        If A is batched (ndim > 2), returns a tensor of shape A.shape[:-2]
        containing one modularity value per batch item. Otherwise, returns a
        scalar tensor (0-dim).

    Notes
    -----
    - The implementation follows the standard modularity definition:
        Q = (1 / (4m)) * sum_c sum_{i,j in c} (A_ij - k_i k_j / (2m))
      where m = total edge weight / 2 (since A is symmetric).
    - This function moves intermediate sums to CPU for accumulation to mirror the
      original behavior; this can be adjusted if you want pure GPU execution.
    """
    ndims = len(A.shape)
    m = A.sum(dim=(-2, -1)) / 2  # total edge weight / 2

    # Expected edges under the configuration model: k_i k_j / (2m)
    k_i = A.sum(dim=-2)  # degree-like sum for each node i
    k_j = A.sum(dim=-1)  # degree-like sum for each node j
    expected_edges = torch.einsum('...i,...j->...ij', k_i, k_j)
    expected_edges /= (2 * m.unsqueeze(-1).unsqueeze(-1)) if ndims > 2 else (2 * m)

    Q = torch.zeros(A.shape[:-2]) if ndims > 2 else torch.tensor(0.0)
    for community in communities:
        sub_A = A[..., community, :][..., :, community]
        sub_expected_edges = expected_edges[..., community, :][..., :, community]
        # Accumulate on CPU to match original code path
        Q = Q + (sub_A - sub_expected_edges).sum(dim=(-2, -1)).cpu()

    return Q / (4 * m.cpu())


def conductance(
    A: torch.Tensor,
    communities: Sequence[Sequence[int]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute conductance statistics for a set of communities.

    For each community S, conductance(S) = cut(S, S̄) / min(vol(S), vol(S̄)),
    where vol(S) = sum of weights inside S, and cut(S, S̄) = sum of weights from
    S to its complement.

    Parameters
    ----------
    A : torch.Tensor
        Similarity/adjacency matrix with shape (..., N, N). Leading dims are batch.
    communities : Sequence[Sequence[int]]
        A list of communities, each a list of node indices.

    Returns
    -------
    min_conductance : torch.Tensor
        The minimum conductance across communities. If A is batched, this is the
        conductance tensor for the single community that minimizes the sum over
        batch dims (matching the original behavior). Otherwise, a scalar tensor.
    max_conductance : torch.Tensor
        The maximum conductance across communities (same batching convention).
    mean_conductance : torch.Tensor
        Mean conductance across communities (averaged along the community axis).
        Shape is A.shape[:-2] for batched, or scalar for unbatched.
    conductance : torch.Tensor
        Raw conductance values per community. If batched, shape is
        (num_communities, ...batch_dims...), else shape is (num_communities,).

    Notes
    -----
    - To avoid division by zero, a tiny epsilon (1e-15) is substituted when
      min(vol(S), vol(S̄)) == 0.
    - The selection of min/max for batched inputs follows the original code:
      it picks the community with the minimal/maximal sum over all batch dims.
    """
    conductance_vals: List[torch.Tensor] = []
    total_vol = torch.sum(A, dim=(-2, -1))

    for community in communities:
        community_bar = [i for i in range(A.shape[-1]) if i not in community]

        cut_edges = torch.sum(A[..., community, :][..., :, community_bar], dim=(-2, -1))
        vol_S = torch.sum(A[..., community, :][..., :, community], dim=(-2, -1))
        vol_S_bar = total_vol - vol_S

        # Avoid division by zero
        min_vol = torch.minimum(vol_S, vol_S_bar)
        if min_vol.ndim == 0:
            if min_vol == 0:
                min_vol = torch.tensor(1e-15, device=min_vol.device, dtype=min_vol.dtype)
        else:
            min_vol[min_vol == 0] = 1e-15

        conductance_vals.append(cut_edges / min_vol)
        del cut_edges, vol_S, vol_S_bar

    conductance = torch.stack(conductance_vals)

    if len(A.shape) > 2:
        # Choose community with min/max sum across all batch dims
        min_conductance = conductance[torch.argmin(conductance.sum(dim=tuple(range(1, conductance.ndim))))]
        max_conductance = conductance[torch.argmax(conductance.sum(dim=tuple(range(1, conductance.ndim))))]
        mean_conductance = conductance.mean(dim=0)
    else:
        min_conductance = conductance.min()
        max_conductance = conductance.max()
        mean_conductance = conductance.mean()

    return min_conductance, max_conductance, mean_conductance, conductance


def intra_inter_sim(
    A: torch.Tensor,
    communities: Sequence[Sequence[int]],
    return_mean: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute intra- vs inter-community similarity and their ratio.

    For each community S, we compute:
      intra(S) = mean(A[S, S])
      inter(S) = mean(A[S, S̄])
      ratio(S) = intra(S) / inter(S)

    Parameters
    ----------
    A : torch.Tensor
        Similarity/adjacency matrix with shape (..., N, N). Leading dims are batch.
    communities : Sequence[Sequence[int]]
        A list of communities, each a list of node indices.
    return_mean : bool, optional
        If True, return the mean over communities for each metric (ratio, intra, inter).
        If False, return per-community tensors stacked along a new first axis.

    Returns
    -------
    ratio : torch.Tensor
        If return_mean=True, shape is A.shape[:-2] (or scalar). Otherwise, shape is
        (num_communities, ...batch_dims...).
    intra : torch.Tensor
        Same shape conventions as ratio.
    inter : torch.Tensor
        Same shape conventions as ratio.

    Notes
    -----
    - When inter(S) is very small, ratio(S) can blow up. This behavior is kept
      identical to the original function; add clamping if needed.
    """
    inter_vals: List[torch.Tensor] = []
    intra_vals: List[torch.Tensor] = []

    for community in communities:
        intra_vals.append(torch.mean(A[..., community, :][..., :, community], dim=(-2, -1)))
        community_bar = [i for i in range(A.shape[-1]) if i not in community]
        inter_vals.append(torch.mean(A[..., community, :][..., :, community_bar], dim=(-2, -1)))

    ratio_vals = [i / j for i, j in zip(intra_vals, inter_vals)]

    if return_mean:
        intra = torch.mean(torch.stack(intra_vals), dim=0)
        inter = torch.mean(torch.stack(inter_vals), dim=0)
        ratio = torch.mean(torch.stack(ratio_vals), dim=0)
        return ratio, intra, inter
    else:
        intra = torch.stack(intra_vals)
        inter = torch.stack(inter_vals)
        ratio = torch.stack(ratio_vals)
        return ratio, intra, inter


def calc_adjacent_similarity_with_offset(
    A: torch.Tensor,
    first_indx: int,
    last_indx: int,
    sim_func=modularity,
) -> torch.Tensor:
    """
    Evaluate a similarity objective for all cut positions in a contiguous range.

    This helper scans candidate boundary indices t in [first_indx, last_indx),
    splits nodes into two communities [0..t-1] and [t..N-1], and applies the given
    similarity function to each split.

    Parameters
    ----------
    A : torch.Tensor
        Similarity/adjacency matrix of shape (N, N). (This helper assumes a
        single matrix, not batched; if you have batched A, slice it first.)
    first_indx : int
        First candidate index t to evaluate (inclusive).
    last_indx : int
        Last candidate index t to evaluate (exclusive).
    sim_func : Callable
        Similarity function with signature sim_func(A, communities) returning a
        scalar tensor (or broadcastable scalar) for this unbatched A. Typically
        one of: modularity, conductance (then you likely wrap to take [0]), or
        intra_inter_sim (wrap to take [0]).

    Returns
    -------
    torch.Tensor
        A 1-D tensor of length (last_indx - first_indx) with the similarity value
        for each candidate t.

    Raises
    ------
    Exception
        If indices are out of range or invalid (first > last, etc.).

    Notes
    -----
    - This function mirrors the original behavior and does not batch over A.
      If you need batched evaluation, call it per batch item in a loop.
    """
    if first_indx > last_indx or first_indx > A.shape[0] or last_indx > A.shape[0]:
        raise Exception(
            f'Problem with indices in similarity calculation: {first_indx}, {last_indx}, {A.shape}'
        )

    T = A.shape[0]
    result = torch.zeros(last_indx - first_indx)

    for t in range(first_indx, last_indx):
        communities = [list(range(0, t)), list(range(t, T))]
        # sim_func may return a tensor; rely on PyTorch itemwise assignment
        result[t - first_indx] = sim_func(A, communities)

    return result
