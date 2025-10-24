from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEventReasoner(nn.Module):
    """
    Reweights retrieved events (and optionally fuses representative tokens)
    using attention. Works in two modes:
      - mode='query': query attends to events (R)
      - mode='self' : events attend to each other
    level='event' uses pooled reps per event; level='token' operates on token reps.
    """
    def __init__(
        self,
        emb_dim: int,
        num_heads: int = 4,
        mode: str = "query",            # "query" | "self"
        level: str = "event",           # "event" | "token"
        summary_pool: str = "mean",     # used when level='event'
        dropout: float = 0.0,
        temperature: float = 1.0,
        fuse_strategy: str = "none",    # "none" | "attn" (token-level fusion)
    ):
        super().__init__()
        assert mode in ("query", "self")
        assert level in ("event", "token")
        assert fuse_strategy in ("none", "attn")
        self.mode = mode
        self.level = level
        self.summary_pool = summary_pool
        self.temperature = temperature
        self.fuse_strategy = fuse_strategy

        self.q_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.k_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.v_proj = nn.Linear(emb_dim, emb_dim, bias=False)

        self.attn = nn.MultiheadAttention(
            embed_dim=emb_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.score_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.ReLU(),
            nn.Linear(emb_dim // 2, 1)
        )

    def _pool_events(self, reps: torch.Tensor) -> torch.Tensor:
        # reps: [N, R, E] -> [N, E]
        if self.summary_pool == "mean":
            return reps.mean(dim=1)
        elif self.summary_pool == "max":
            return reps.max(dim=1).values
        raise ValueError("Unknown summary_pool")

    @torch.no_grad()
    def forward(
        self,
        event_reps: torch.Tensor,              # [N,R,E] for level='event' or 'token'
        query_rep: Optional[torch.Tensor],     # [E] if mode='query'
        base_scores: Optional[torch.Tensor],   # [N] (logit-like), can be None
        per_event_token_indices=None           # optional, for returning fused tokens
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
          weights: [N] softmax over events
          scores:  [N] pre-softmax
          fused_tokens (optional): when level='token' and fuse_strategy='attn',
                                   shape [T*, E] fused token matrix to replace naive concat
        """
        N, R, E = event_reps.shape

        if self.level == "event":
            # pool per event to 1 vector
            pooled = self._pool_events(event_reps)     # [N, E]
            keys_vals = pooled.unsqueeze(0)            # [1, N, E]
        else:
            # treat each token rep as a node; keep an event index map
            # flatten tokens across events
            flat = event_reps.reshape(N * R, E)        # [N*R, E]
            keys_vals = flat.unsqueeze(0)              # [1, N*R, E]

        # Build Q, K, V
        if self.mode == "query":
            assert query_rep is not None
            Q = self.q_proj(query_rep).unsqueeze(0).unsqueeze(0)  # [1,1,E]
            K = self.k_proj(keys_vals)
            V = self.v_proj(keys_vals)
        else:
            Q = self.q_proj(keys_vals)
            K = self.k_proj(keys_vals)
            V = self.v_proj(keys_vals)

        attn_out, attn_w = self.attn(Q, K, V)  # attn_w: [1, Q_len, K_len]

        if self.level == "event":
            # contextual vector per event score head
            contextual = attn_out.squeeze(0).squeeze(0)        # [E]
            # Project the per-event keys (pooled) through V to score events
            per_event_vecs = V.squeeze(0)                      # [N, E]
            scores = self.score_head(per_event_vecs).squeeze(-1)  # [N]
        else:
            # token-level: reduce token outputs to per-event scores
            per_token_vecs = V.squeeze(0)                      # [N*R, E]
            token_scores = self.score_head(per_token_vecs).squeeze(-1)  # [N*R]
            scores = token_scores.view(N, R).mean(dim=1)       # [N]

        if base_scores is not None:
            scores = scores + base_scores

        weights = F.softmax(scores / self.temperature, dim=0)

        fused_tokens = None
        if self.level == "token" and self.fuse_strategy == "attn":
            # Produce a single fused token sequence via attention weights
            # Normalize attn_w across tokens and take weighted sum of tokens
            # attn_w: [1,1,N*R] or [1,N*R,N*R] (we're in query mode so it's [1,1,N*R])
            alpha = attn_w.squeeze(0).squeeze(0)    # [N*R]
            alpha = alpha / (alpha.sum() + 1e-6)
            fused = (alpha.unsqueeze(-1) * per_token_vecs).sum(dim=0)  # [E]
            fused_tokens = fused.unsqueeze(0)  # [1, E]
        return weights, scores, fused_tokens
