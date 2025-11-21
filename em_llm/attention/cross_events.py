from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEventReasoner(nn.Module):
    """
    Lightweight, *deterministic* cross-event reasoning module.

    Goal
    ----
    - Re-rank memory blocks ("events") using:
        1) Query ↔ event similarity
        2) Event ↔ event connectivity (how central an event is among others)
    - NO learnable parameters (so it works out-of-the-box without training).
    - Runs on CPU in float32 to avoid GPU OOM / dtype issues.

    Shapes
    ------
    - event_reps: [B, N, R, E] or [N, R, E] or [N, E]
        B: batch (we only use B=1 in EM-LLM)
        N: number of events / blocks
        R: tokens per event (we often pass R=1, a single block vector)
        E: embedding dimension (must match emb_dim)
    - query_rep: [B, E] or [E]
    """

    def __init__(
        self,
        emb_dim: int,
        num_heads: int = 4,              # kept for config compatibility (unused)
        summary_pool: str = "mean",      # how to pool tokens if R > 1
        dropout: float = 0.0,            # unused (no learnable layers)
        temperature: float = 1.0,
        enable_event_fusion: bool = True,   # controls use of cross-event centrality
        fusion_threshold: float = 0.5,      # unused (no hard fusion)
        similarity_metric: str = "cosine",  # default cosine for stability
        alpha_query: float = 0.7,           # weight for query similarity vs. cross-event centrality
    ):
        super().__init__()

        if summary_pool not in ("mean", "max"):
            raise ValueError("summary_pool must be 'mean' or 'max'")

        self.emb_dim = emb_dim
        self.summary_pool = summary_pool
        self.temperature = float(temperature) if temperature is not None else 1.0
        self.enable_event_fusion = enable_event_fusion
        self.similarity_metric = similarity_metric
        self.alpha_query = float(alpha_query)

        # No learnable parameters – this is intentionally a *functional* module.
        self.register_buffer("_dummy", torch.zeros(1), persistent=False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _normalize_events(self, reps: torch.Tensor) -> torch.Tensor:
        """
        reps: [B, N, E]
        Returns: L2-normalized reps with safe epsilon to avoid NaNs.
        """
        return F.normalize(reps, p=2, dim=-1, eps=1e-6)

    def _pool_events(self, reps: torch.Tensor) -> torch.Tensor:
        """
        reps: [B, N, R, E] -> [B, N, E]
        """
        if reps.dim() != 4:
            raise ValueError(f"_pool_events expects [B,N,R,E], got shape {reps.shape}")
        if self.summary_pool == "mean":
            return reps.mean(dim=-2)
        else:
            return reps.max(dim=-2).values

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        event_reps: torch.Tensor,              # [N,E] or [N,R,E] or [B,N,R,E]
        query_rep: torch.Tensor,               # [E] or [B,E]
        base_scores: Optional[torch.Tensor] = None,
        per_event_token_indices=None           # unused – kept for API compatibility
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns
        -------
        weights : [B, N]  (softmax over events)
        scores  : [B, N]  (pre-softmax)
        fused_tokens : None (we only operate at event level)
        """
        # ------------------------------------------------------------------
        # 1) Normalize shapes & dtypes
        # ------------------------------------------------------------------
        # Add batch / token dimensions as needed
        single_example = False

        if event_reps.dim() == 2:
            # [N,E] -> [1,N,1,E]
            event_reps = event_reps.unsqueeze(0).unsqueeze(2)
            single_example = True
        elif event_reps.dim() == 3:
            # [N,R,E] -> [1,N,R,E]
            event_reps = event_reps.unsqueeze(0)
            single_example = True
        elif event_reps.dim() != 4:
            raise ValueError(
                f"event_reps must have dim 2, 3 or 4; got shape {event_reps.shape}"
            )

        B, N, R, E = event_reps.shape
        if E != self.emb_dim:
            raise ValueError(
                f"CrossEventReasoner emb_dim={self.emb_dim}, but got event_reps with E={E}"
            )

        # query_rep: [E] or [B,E]
        if query_rep.dim() == 1:
            query_rep = query_rep.unsqueeze(0).expand(B, -1)
        elif query_rep.dim() == 2:
            if query_rep.size(0) != B:
                # broadcast single query across batch
                if query_rep.size(0) == 1:
                    query_rep = query_rep.expand(B, -1)
                else:
                    raise ValueError(
                        f"query_rep batch dim mismatch: B={B}, query_rep.shape={query_rep.shape}"
                    )
        else:
            raise ValueError(
                f"query_rep must have dim 1 or 2; got shape {query_rep.shape}"
            )

        # Move to float32 for stability (module itself should already live on CPU)
        event_reps = event_reps.to(dtype=torch.float32)
        query_rep = query_rep.to(dtype=torch.float32)

        # ------------------------------------------------------------------
        # 2) Pool tokens within each event: [B,N,R,E] -> [B,N,E]
        # ------------------------------------------------------------------
        event_vecs = self._pool_events(event_reps)  # [B,N,E]

        # ------------------------------------------------------------------
        # 3) Query ↔ event similarity  (cosine)
        # ------------------------------------------------------------------
        event_norm = self._normalize_events(event_vecs)             # [B,N,E]
        query_norm = F.normalize(query_rep, p=2, dim=-1, eps=1e-6)  # [B,E]

        # [B,N,E] · [B,E,1] -> [B,N,1] -> [B,N]
        q_scores = torch.bmm(event_norm, query_norm.unsqueeze(-1)).squeeze(-1)

        # ------------------------------------------------------------------
        # 4) Cross-event connectivity (centrality)
        # ------------------------------------------------------------------
        if self.enable_event_fusion and N > 1:
            # [B,N,E] @ [B,E,N] -> [B,N,N]
            cross_sim = torch.bmm(event_norm, event_norm.transpose(1, 2))
            # Zero out diagonal so an event doesn't "vote" for itself too strongly
            eye = torch.eye(N, device=cross_sim.device, dtype=cross_sim.dtype).unsqueeze(0)
            cross_sim = cross_sim * (1.0 - eye)

            # Simple centrality: mean similarity to all other events
            centrality = cross_sim.mean(dim=-1)  # [B,N]
        else:
            centrality = torch.zeros_like(q_scores)

        # ------------------------------------------------------------------
        # 5) Combine scores
        # ------------------------------------------------------------------
        # Final score = α * query_sim + (1-α) * centrality  (+ base_scores if provided)
        alpha = self.alpha_query
        scores = alpha * q_scores + (1.0 - alpha) * centrality

        if base_scores is not None:
            base_scores = base_scores.to(scores.dtype)
            if base_scores.dim() == 1:
                base_scores = base_scores.unsqueeze(0).expand(B, -1)
            elif base_scores.dim() == 2 and base_scores.size(0) == 1 and B > 1:
                base_scores = base_scores.expand(B, -1)
            scores = scores + base_scores

        # ------------------------------------------------------------------
        # 6) Softmax over events
        # ------------------------------------------------------------------
        logits = scores / max(self.temperature, 1e-6)
        weights = torch.softmax(logits, dim=-1)

        fused_tokens = None

        if single_example:
            # Collapse batch dim for compatibility with caller that expects [N]
            weights = weights.squeeze(0)   # [N]
            scores = scores.squeeze(0)     # [N]

        return weights, scores, fused_tokens
