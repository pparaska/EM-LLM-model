from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEventReasoner(nn.Module):
    """
    Enhanced event reasoning module that:
    1. Computes cross-event relationships through event-to-event attention
    2. Identifies and strengthens multi-hop connections between events
    3. Reweights events based on both query relevance and cross-event importance
    4. Optionally fuses strongly related events
    emb_dim: Embedding dimension size - the vector dimension for representing each event
    num_heads: Number of parallel attention heads for different relationship patterns
    summary_pool: Method for pooling token representations ("mean" or "max")
    temperature: Controls softmax distribution sharpness
    fusion_threshold: Minimum similarity score (0.5) for event fusion
    """
    def __init__(
    self,
    emb_dim: int,        # Embedding dimension - size of the vector that represents each event
    num_heads: int = 4,  # Number of attention heads
    summary_pool: str = "mean",  # How to combine token vectors
    dropout: float = 0.0,        # Regularization rate
        #Dropout is a regularization technique
        #During training, randomly "turns off" some neurons
        #Helps prevent overfitting
        #Example: dropout=0.1 means 10% of neurons are randomly deactivated
        # 0.0 means no dropout (all neurons active)
    temperature: float = 1.0,    # Controls softmax sharpness
    enable_event_fusion: bool = True,  # Whether to combine related events
    fusion_threshold: float = 0.5,     # Minimum similarity for fusion
    similarity_metric: str = "dot_product"  # Similarity metric: "dot_product" or "cosine"
    ):
        super().__init__()
        # Validate similarity metric
        if similarity_metric not in ["dot_product", "cosine"]:
            raise ValueError("similarity_metric must be either 'dot_product' or 'cosine'")
            
        # module behaviour
        self.mode = "query"      # Operating mode - processes queries against events
        self.level = "event"     # Processing level - works at event level (not token level)
        self.summary_pool = summary_pool  # How to combine tokens ("mean" or "max")
        self.similarity_metric = similarity_metric  # Type of similarity to use
        self.temperature = temperature  # Controls how "sharp" the attention focus is
        # Lower temperature (e.g., 0.5) = More focused on highest scores
        # Higher temperature (e.g., 2.0) = More distributed attention

        self.enable_event_fusion = enable_event_fusion  # Whether to combine related events
        self.fusion_threshold = fusion_threshold  # Minimum similarity score (0.5 = 50% similar)
       
        # Cross-event attention for event-to-event reasoning
        self.event_self_attn = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Event fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )

        # Projection layers for Query, Key, Value
        self.q_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.k_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.v_proj = nn.Linear(emb_dim, emb_dim, bias=False)

        # Attention layer
        self.attn = nn.MultiheadAttention(
            embed_dim=emb_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Scoring head to compute per-event scores
        self.score_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2), # Reduces dimension by half
            nn.ReLU(),
            nn.Linear(emb_dim // 2, 1)
        )

        # B: Batch size (number of sequences being processed)
        # N: Number of events
        # R: Number of tokens per event
        # E: Embedding dimension (size of each token's vector)

    def _pool_events(self, reps: torch.Tensor) -> torch.Tensor:
            # reps: Input tensor with shape [*, N, R, E]
            # *: Optional batch dimension
             # Returns: [*, N, E] - combines token dimension R
        if self.summary_pool == "mean":
            return reps.mean(dim=-2)
        elif self.summary_pool == "max":
            return reps.max(dim=-2).values
        raise ValueError("Unknown summary_pool")

    def forward(
        self,
        event_reps: torch.Tensor,              # [N,R,E] or [B,N,R,E] Event Representations
        query_rep: torch.Tensor,               # [E] or [B,E] Query Representation
        base_scores: Optional[torch.Tensor] = None,   # optional, currently supported but can be commented
        per_event_token_indices=None           # reserved / unused
        ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward supports both single-example and batched inputs.

        Inputs:
          event_reps: [N,R,E] (single) or [B,N,R,E] (batched)
          query_rep: [E] or [B,E] when mode='query'
          base_scores: [N] or [B,N]

        Returns:
          weights: [B,N] softmax over events (batch first)
          scores:  [B,N] pre-softmax
          fused_tokens (optional): when level='token' and fuse_strategy='attn',
                                   shape [B,E] (one fused token per batch)
        """
        # Normalize incoming shapes to have a batch dimension.
        single_example = False
        if event_reps.dim() == 3:
            # [N,R,E] -> [1,N,R,E]
            event_reps = event_reps.unsqueeze(0)
            single_example = True

        # Now event_reps: [B, N, R, E]
        B, N, R, E = event_reps.shape

        # Build keys/values sequence per batch: pool tokens per event -> [B, N, E]
        pooled = self._pool_events(event_reps)     # [B, N, E]
        
        # 1. Cross-Event Reasoning: Let events interact with each other
        if self.similarity_metric == "cosine":
            # Normalize for cosine similarity
            pooled_norm = F.normalize(pooled, p=2, dim=-1)
            # Custom attention with cosine similarity
            attention_scores = torch.bmm(pooled_norm, pooled_norm.transpose(-2, -1))
            cross_event_weights = F.softmax(attention_scores / self.temperature, dim=-1)
            cross_event_out = torch.bmm(cross_event_weights, pooled)
        else:  # dot product (default)
            cross_event_out, cross_event_weights = self.event_self_attn(
                pooled, pooled, pooled
            )
        
        # 2. Optional Event Fusion: Combine strongly related events
        if self.enable_event_fusion:
            fused_events = []
            used_events = set()
            
            for i in range(N):
                if i in used_events:
                    continue
                
                # Find strongly related events using the chosen similarity metric
                if self.similarity_metric == "cosine":
                    # Cosine similarity is already normalized in attention_scores
                    related_mask = attention_scores[0, i] > self.fusion_threshold
                else:
                    # For dot product, use attention weights directly
                    related_mask = cross_event_weights[0, i] > self.fusion_threshold
                related_indices = related_mask.nonzero().squeeze(-1)
                
                if len(related_indices) > 1:  # If there are multiple events to fuse
                    # Get all related events
                    related = cross_event_out[:, related_indices]
                    
                    # Concatenate with current event and fuse
                    current = cross_event_out[:, i:i+1].expand(-1, len(related_indices), -1)
                    fused = self.fusion_layer(
                        torch.cat([current, related], dim=-1)
                    )
                    
                    # Add fused representation
                    fused_events.append(fused.mean(dim=1, keepdim=True))
                    used_events.update(related_indices.tolist())
                else:
                    # Keep original event if no strong relationships
                    fused_events.append(cross_event_out[:, i:i+1])
                    used_events.add(i)
            
            # Replace pooled events with fused ones
            keys_vals = torch.cat(fused_events, dim=1)
        else:
            # Use cross-event attention output directly
            keys_vals = cross_event_out

        # 3. Prepare Q, K, V for query -> event attention.
        # query_rep must be provided (either [E] or [B,E]).
        if query_rep.dim() == 1:
            query_rep = query_rep.unsqueeze(0).expand(B, -1)
        Q = self.q_proj(query_rep).unsqueeze(1)   # [B,1,E]
        K = self.k_proj(keys_vals)                # [B,N,E]
        V = self.v_proj(keys_vals)                # [B,N,E]

        # MultiheadAttention with batch_first=True expects shapes [B, seq, E]
        attn_out, attn_w = self.attn(Q, K, V)
        # attn_w: [B, 1, N]

        # Compute per-event scores from V (pooled per-event vectors)
        per_event_vecs = V  # [B, N, E]
        
        if self.similarity_metric == "cosine":
            # Normalize vectors for cosine similarity
            query_norm = F.normalize(Q, p=2, dim=-1)  # [B,1,E]
            event_norm = F.normalize(per_event_vecs, p=2, dim=-1)  # [B,N,E]
            # Compute cosine similarity
            scores = torch.bmm(query_norm, event_norm.transpose(-2, -1)).squeeze(1)  # [B,N]
        else:  # dot product # Query how to calculate similarity?
            scores = self.score_head(per_event_vecs).squeeze(-1)  # [B, N]
            
        ### Optional
        # Add optional base scores (support [N] or [B,N])
        if base_scores is not None:
            if base_scores.dim() == 1:
                base_scores = base_scores.unsqueeze(0).expand(B, -1)
            scores = scores + base_scores

        # Softmax across events dimension (dim=1)
        weights = F.softmax(scores / self.temperature, dim=1)

        # token fusion / token-level behaviour intentionally disabled; return None
        fused_tokens = None

        # If caller passed single-example shapes, squeeze batch dim for compatibility
        if single_example:
            weights = weights.squeeze(0)   # [N]
            scores = scores.squeeze(0)     # [N]
        return weights, scores, fused_tokens
