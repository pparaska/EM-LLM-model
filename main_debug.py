import torch
from em_llm.attention.cross_events import CrossEventReasoner

# Sample configuration
emb_dim = 8  # Small for debug
num_heads = 2
num_events = 3
num_tokens = 4
batch_size = 1

# Instantiate CrossEventReasoner
reasoner = CrossEventReasoner(
    emb_dim=emb_dim,
    num_heads=num_heads,
    summary_pool="mean",
    dropout=0.0,
    temperature=1.0,
    enable_event_fusion=True,
    fusion_threshold=0.5,
    similarity_metric="dot_product"  # Change to "cosine" to test
)

# Create sample event representations: [B, N, R, E]
event_reps = torch.randn(batch_size, num_events, num_tokens, emb_dim)
# Create sample query representation: [B, E]
query_rep = torch.randn(batch_size, emb_dim)

# Optional: base scores (can be None)
base_scores = torch.randn(batch_size, num_events)

# Call the reasoner (this will invoke forward)
weights, scores, fused_tokens = reasoner(event_reps, query_rep, base_scores)

print("Event representations shape:", event_reps.shape)
print("Query representation shape:", query_rep.shape)
print("Weights:", weights)
print("Scores:", scores)
print("Fused tokens:", fused_tokens)
