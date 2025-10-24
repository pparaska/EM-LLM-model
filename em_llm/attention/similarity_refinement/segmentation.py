"""
Event boundary refinement via graph-theoretic similarity.

This module refines a sequence of preliminary event boundaries (e.g., found via
surprise/novelty spikes) by searching locally around each boundary for a position
that optimizes a graph-based coherence criterion computed on a token–token
adjacency/similarity matrix A.

Supported similarity objectives:
- modularity: maximize partition modularity at the candidate cut
- conductance: minimize conductance at the candidate cut
- intra_inter_sim: maximize intra-cluster similarity minus inter-cluster similarity

Typical usage:
    refined = events_with_similarity_adjustment(
        events_base=initial_events,
        A=adj_matrix,                         # shape (T, T), torch.float
        similarity_metric="modularity",
        min_size=16,                          # minimum event length
        offset=local_window_start             # optional, if A is a sub-window of the full sequence
    )
"""

from typing import List, Sequence, Union
import torch
from .similarity import modularity, conductance, intra_inter_sim, calc_adjacent_similarity_with_offset


def events_with_similarity_adjustment(
    events_base: Sequence[int],
    A: torch.Tensor,
    similarity_metric: str = "modularity",
    min_size: int = 0,
    offset: Union[int, torch.Tensor] = 0,
) -> List[int]:
    """
    Refine event boundaries by locally optimizing a graph similarity objective.

    The function takes a list of initial event boundaries (indices into a token/step
    sequence) and an adjacency/similarity matrix A over those steps. For each event,
    it searches within a symmetric window around the original boundary—up to half of
    the preceding event's length on each side—and picks the position that maximizes
    modularity (or intra_inter_sim) or minimizes conductance.

    Parameters
    ----------
    events_base : Sequence[int]
        Monotonically increasing list of base (tentative) event boundary indices.
        These are absolute indices in the original sequence (before any offset).
    A : torch.Tensor
        Square adjacency/similarity matrix of shape (T, T), where T is the length
        of the (sub)sequence represented. Values are typically non-negative and
        larger means more similar/connected. If `offset` > 0, A usually corresponds
        to a sub-window [offset : offset+T) of the full sequence.
    similarity_metric : str, optional
        Which objective to optimize when repositioning the boundary:
          - "modularity": maximize modularity (community quality)
          - "conductance": minimize conductance (lower is better separation)
          - "intra_inter_sim": maximize intra - inter similarity
        Default is "modularity".
    min_size : int, optional
        Minimum allowed event length. Boundaries that would create an event shorter
        than this are not moved earlier than `events_temp[-1] + min_size`. Also used
        as a guard to skip evaluating the first `min_size` candidate cut positions
        inside the window.
    offset : int or torch.Tensor, optional
        Offset of A relative to the original sequence start. If A represents
        tokens [offset .. offset+T), then a base boundary at absolute index `b`
        corresponds to row/column `b - offset` inside A. Internally we shift base
        boundaries by `offset` to index into A correctly during refinement.

    Returns
    -------
    List[int]
        A list of refined event boundary indices in the original (absolute)
        coordinate system. The number of boundaries matches `len(events_base)`.

    Algorithm
    ---------
    1) Shift base boundaries by `offset` for A-index alignment and drop boundaries
       that are closer than `min_size` apart (to preserve minimum event length).
    2) Iterate boundaries in order. For each boundary:
       - If there is enough space (previous boundary at least `min_size` away),
         define a local search window centered on the original boundary with
         half-width equal to half the size of the preceding event (clamped to A).
       - Extract the submatrix TI_LES = A[start_from:end_to, start_from:end_to].
       - Use `calc_adjacent_similarity_with_offset` to evaluate the objective for
         all candidate cut positions from `first_indx_to_check` up to the original
         boundary position.
       - Choose argmax (modularity, intra_inter_sim) or argmin (conductance) over
         candidates starting at `min_size` to enforce the minimum length guard.
       - Append the chosen absolute position to the refined boundary list.
       - If there isn't enough room to search (e.g., near the very start), keep the
         original boundary.
    3) Shift refined boundaries back by `offset` to return absolute indices.
    4) Sanity-check that the count matches the input.

    Notes
    -----
    - The search window width is adaptive: it is twice the previous event size
      (i.e., ± half of that size around the original boundary), clamped to A's
      valid index range.
    - `min_size` acts both as a minimum event length and as a safety margin inside
      the search window to avoid producing too-small left segments.

    Raises
    ------
    NotImplementedError
        If an unknown `similarity_metric` is provided.
    Exception
        If an event would become shorter than `min_size` in a way not handled by
        the guards (should be rare; indicates inconsistent inputs).

    Examples
    --------
    >>> A = torch.rand(100, 100); A = (A + A.T) / 2  # symmetric similarity
    >>> base = [20, 45, 70, 90]
    >>> refined = events_with_similarity_adjustment(base, A, "modularity", min_size=8, offset=0)
    >>> len(refined) == len(base)
    True
    """
    events_temp = [0]

    # Align base boundaries to the coordinate system of A by adding offset.
    events_base = [i + offset for i in events_base]

    # Enforce minimum spacing between successive boundaries
    events_base_ = [events_base[0]] if events_base[0] >= min_size or offset == 0 else []
    events_base = events_base_ + [
        events_base[i]
        for i in range(1, len(events_base))
        if events_base[i] - events_base[i - 1] >= min_size
    ]

    # Choose similarity objective
    if similarity_metric == "modularity":
        sim_func = modularity
    elif similarity_metric == "conductance":
        sim_func = lambda a, c: conductance(a, c)[0]
    elif similarity_metric == "intra_inter_sim":
        sim_func = lambda a, c: intra_inter_sim(a, c)[0]
    else:
        raise NotImplementedError(f"Similarity metric {similarity_metric} not implemented")

    for event in events_base:
        # Only attempt a search if we can keep at least min_size on the left side
        if event - events_temp[-1] > min_size:
            if event - offset > min_size:
                # Adaptive window: up to half the previous event size on each side
                original_event_size = event - events_temp[-1]
                half_size = int(original_event_size / 2)

                # Window bounds in A's coordinates
                start_from = max(0, events_temp[-1] - half_size)
                end_to = min(A.shape[0], event + half_size)

                # Relative indices inside the window to start/stop checking cuts
                first_indx_to_check = max(
                    offset - start_from, events_temp[-1] - start_from
                )  # choose the later of (offset, left boundary)
                last_indx_to_check = event - start_from

                # Submatrix for the local window
                TI_LES = torch.clone(A[start_from:end_to, :][:, start_from:end_to])

                # Evaluate objective at all candidate cut positions within the window
                adj_mod = calc_adjacent_similarity_with_offset(
                    TI_LES,
                    first_indx_to_check,
                    last_indx_to_check,
                    sim_func=sim_func,
                )

                # Optimize: argmax for modularity/intra_inter_sim, argmin for conductance
                if similarity_metric == "conductance":
                    arg_mod = torch.argmin(adj_mod[min_size:])
                else:
                    arg_mod = torch.argmax(adj_mod[min_size:])

                # Convert back to absolute coordinate
                events_temp.append(start_from + first_indx_to_check + min_size + arg_mod)
            else:
                # Near the very start: keep the original boundary
                events_temp.append(event)

        # Allow equal-to-min_size if it's the very first event or offset==0
        elif event - events_temp[-1] == min_size or offset == 0:
            events_temp.append(event)
        else:
            # Inconsistent input or impossible minimum-size constraint
            print(len(events_base))
            print(len(events_temp))
            raise Exception(f"Problem with event size: {event}")

    # Convert back to absolute indices (remove the working 0 and subtract offset)
    events_temp = [i.item() - offset.item() for i in events_temp[1:]]  # type: ignore[attr-defined]

    assert len(events_temp) == len(
        events_base
    ), f"Problem with refinement does not have the same number of events: {len(events_temp)}, {len(events_base)}"
    return events_temp
