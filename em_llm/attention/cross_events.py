from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEventReasoner(nn.Module):
    """
    Cross-event reasoning module for episodic memory.
    Allows memory events to interact before final query matching.
    """
    def __init__(
        self,
        emb_dim: int,
        num_heads: int = 4,
        summary_pool: str = "mean",
        dropout: float = 0.0,
        temperature: float = 1.0,
        enable_event_fusion: bool = True,
        fusion_threshold: float = 0.5,
        similarity_metric: str = "dot_product"
    ):
        super().__init__()
        
        if similarity_metric not in ["dot_product", "cosine"]:
            raise ValueError("similarity_metric must be 'dot_product' or 'cosine'")
        
        self.mode = "query"
        self.level = "event"
        self.summary_pool = summary_pool
        self.similarity_metric = similarity_metric
        self.temperature = temperature
        self.enable_event_fusion = enable_event_fusion
        self.fusion_threshold = fusion_threshold
       
        # Event-to-event attention
        self.event_self_attn = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # MLP for fusing similar events
        self.fusion_layer = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )

        # Q, K, V projections
        self.q_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.k_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.v_proj = nn.Linear(emb_dim, emb_dim, bias=False)

        # Query to event attention
        self.attn = nn.MultiheadAttention(
            embed_dim=emb_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Score head for final event scoring
        self.score_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.ReLU(),
            nn.Linear(emb_dim // 2, 1)
        )

    def _pool_events(self, reps: torch.Tensor) -> torch.Tensor:
        # Combine tokens within each event to get event representation
        # reps shape: [B, N, R, E] where R is tokens per event
        if self.summary_pool == "mean":
            return reps.mean(dim=-2)
        elif self.summary_pool == "max":
            return reps.max(dim=-2).values
        raise ValueError("Unknown summary_pool")

    def _compute_cross_event_attention(self, pooled: torch.Tensor):
        # Let events attend to each other
        # This helps events share information before query matching
        
        if self.similarity_metric == "cosine":
            # Use cosine similarity
            pooled_norm = F.normalize(pooled, p=2, dim=-1)
            attention_scores = torch.bmm(pooled_norm, pooled_norm.transpose(-2, -1))
            cross_event_weights = F.softmax(attention_scores / self.temperature, dim=-1)
            cross_event_out = torch.bmm(cross_event_weights, pooled)
        else:
            # Use standard dot product attention
            cross_event_out, cross_event_weights = self.event_self_attn(
                pooled, pooled, pooled
            )
        
        return cross_event_out, cross_event_weights

    def _fuse_related_events(self, cross_event_out, cross_event_weights, attention_scores=None):
        # Merge events that are highly similar to reduce redundancy
        
        B, N, E = cross_event_out.shape
        fused_events = []
        used_events = set()
        
        for i in range(N):
            if i in used_events:
                continue
            
            # Find events similar to current event i
            if self.similarity_metric == "cosine" and attention_scores is not None:
                related_mask = attention_scores[0, i] > self.fusion_threshold
            else:
                related_mask = cross_event_weights[0, i] > self.fusion_threshold
            
            related_indices = related_mask.nonzero().squeeze(-1)
            
            if len(related_indices) > 1:
                # Fuse multiple related events
                related = cross_event_out[:, related_indices]
                current = cross_event_out[:, i:i+1].expand(-1, len(related_indices), -1)
                
                # Pass through fusion MLP
                fused = self.fusion_layer(torch.cat([current, related], dim=-1))
                fused_events.append(fused.mean(dim=1, keepdim=True))
                
                used_events.update(related_indices.tolist())
            else:
                # Keep event as is
                fused_events.append(cross_event_out[:, i:i+1])
                used_events.add(i)
        
        return torch.cat(fused_events, dim=1)

    def _compute_query_to_event_attention(self, query_rep, keys_vals, B):
        # Compute attention from query to events
        
        if query_rep.dim() == 1:
            query_rep = query_rep.unsqueeze(0).expand(B, -1)
        
        # Project to Q, K, V
        Q = self.q_proj(query_rep).unsqueeze(1)
        K = self.k_proj(keys_vals)
        V = self.v_proj(keys_vals)

        # Apply attention
        attn_out, attn_w = self.attn(Q, K, V)
        
        return attn_out, attn_w, V

    def _compute_event_scores(self, query_rep, per_event_vecs, base_scores, B):
        # Compute final score for each event
        
        if self.similarity_metric == "cosine":
            query_norm = F.normalize(query_rep, p=2, dim=-1)
            event_norm = F.normalize(per_event_vecs, p=2, dim=-1)
            scores = torch.bmm(query_norm, event_norm.transpose(-2, -1)).squeeze(1)
        else:
            scores = self.score_head(per_event_vecs).squeeze(-1)
        
        # Add base scores if provided
        if base_scores is not None:
            if base_scores.dim() == 1:
                base_scores = base_scores.unsqueeze(0).expand(B, -1)
            scores = scores + base_scores

        # Apply softmax
        weights = F.softmax(scores / self.temperature, dim=1)
        
        return weights, scores

    def forward(self, event_reps, query_rep, base_scores=None, per_event_token_indices=None):
        """
        Main forward pass for cross-event reasoning.
        
        Args:
            event_reps: Event representations [N,R,E] or [B,N,R,E]
            query_rep: Query vector [E] or [B,E]
            base_scores: Optional initial scores
            
        Returns:
            weights: Normalized event weights
            scores: Raw event scores
            fused_tokens: None (not used)
        """
        # Convert to float32 for compatibility
        event_reps = event_reps.float()
        query_rep = query_rep.float()
        
        # Add batch dimension if needed
        single_example = False
        if event_reps.dim() == 3:
            event_reps = event_reps.unsqueeze(0)
            single_example = True

        B, N, R, E = event_reps.shape

        # Step 1: Pool tokens to get event vectors
        pooled = self._pool_events(event_reps)
        
        # Step 2: Let events attend to each other
        cross_event_out, cross_event_weights = self._compute_cross_event_attention(pooled)
        
        # Compute scores for fusion (only for cosine metric)
        attention_scores = None
        if self.similarity_metric == "cosine":
            pooled_norm = F.normalize(pooled, p=2, dim=-1)
            attention_scores = torch.bmm(pooled_norm, pooled_norm.transpose(-2, -1))
        
        # Step 3: Fuse similar events if enabled
        if self.enable_event_fusion:
            keys_vals = self._fuse_related_events(
                cross_event_out, 
                cross_event_weights,
                attention_scores
            )
        else:
            keys_vals = cross_event_out

        # Step 4: Query attends to events
        attn_out, attn_w, per_event_vecs = self._compute_query_to_event_attention(
            query_rep, keys_vals, B
        )

        # Step 5: Get final scores
        weights, scores = self._compute_event_scores(
            attn_out, per_event_vecs, base_scores, B
        )

        fused_tokens = None

        # Remove batch dimension for single examples
        if single_example:
            weights = weights.squeeze(0)
            scores = scores.squeeze(0)
        
        return weights, scores, fused_tokens
