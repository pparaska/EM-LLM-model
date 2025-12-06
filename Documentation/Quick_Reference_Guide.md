# EM-LLM Quick Reference Guide

## Visual Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER STARTS DEBUGGING                        │
│                    run_pipeline_debug.py                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MAIN PROCESSING ENGINE                        │
│                    benchmark/pred.py                            │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Load Config  │→ │  Load Model  │→ │ Load Dataset │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL MODIFICATION                            │
│                    em_llm/utils/patch_hf.py                     │
│                                                                  │
│  Standard Transformer  →  EM-LLM Enhanced Transformer           │
│                                                                  │
│  ┌────────────────┐      ┌────────────────────────────┐       │
│  │ Normal         │  →   │ + Memory Management        │       │
│  │ Attention      │      │ + Block Retrieval          │       │
│  │                │      │ + Cross-Event Reasoning    │       │
│  └────────────────┘      └────────────────────────────┘       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   MEMORY MANAGEMENT                              │
│         em_llm/attention/context_manager.py                     │
│                                                                  │
│  ┌──────────────────────────────────────────────────────┐     │
│  │  Input: Long Document (e.g., 100,000 tokens)         │     │
│  └────────────────┬─────────────────────────────────────┘     │
│                   │                                             │
│  ┌────────────────▼─────────────────────────────────────┐     │
│  │  Step 1: Segment into Blocks                         │     │
│  │  [Block1: 0-32] [Block2: 33-64] ... [BlockN]        │     │
│  └────────────────┬─────────────────────────────────────┘     │
│                   │                                             │
│  ┌────────────────▼─────────────────────────────────────┐     │
│  │  Step 2: Store in Hierarchy                          │     │
│  │  GPU: 128 most important blocks                      │     │
│  │  CPU: Recently used blocks                           │     │
│  │  Disk: All blocks                                    │     │
│  └────────────────┬─────────────────────────────────────┘     │
│                   │                                             │
│  ┌────────────────▼─────────────────────────────────────┐     │
│  │  Step 3: Retrieval                                   │     │
│  │  Query: "Who won the game?"                          │     │
│  │  → Rank all blocks by relevance                      │     │
│  │  → Load top-K blocks to GPU                          │     │
│  └────────────────┬─────────────────────────────────────┘     │
│                   │                                             │
│  ┌────────────────▼─────────────────────────────────────┐     │
│  │  Step 4: Cross-Event Reasoning                       │     │
│  │  Find connections between blocks                     │     │
│  │  Reweight based on relationships                     │     │
│  └────────────────┬─────────────────────────────────────┘     │
│                   │                                             │
│  ┌────────────────▼─────────────────────────────────────┐     │
│  │  Step 5: Multi-Stage Attention                       │     │
│  │  Local: Recent 1024 tokens                           │     │
│  │  Global: Retrieved blocks                            │     │
│  └────────────────┬─────────────────────────────────────┘     │
│                   │                                             │
│                   ▼                                             │
│              [Output]                                           │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TEXT GENERATION                               │
│              em_llm/utils/greedy_search.py                      │
│                                                                  │
│  ┌──────────────────────────────────────────────────────┐     │
│  │  For each new token:                                 │     │
│  │  1. Use memory to understand context                 │     │
│  │  2. Predict next word (greedy = pick most likely)    │     │
│  │  3. Add to sequence                                  │     │
│  │  4. Repeat until done                                │     │
│  └──────────────────────────────────────────────────────┘     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EVALUATION                                    │
│                    benchmark/eval.py                            │
│                                                                  │
│  Compare predictions vs. ground truth                           │
│  Calculate metrics (F1, ROUGE, etc.)                            │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow Example

### Example: Answering "Who won the championship?"

```
Document: 
"The Lakers played against the Celtics in the finals. 
 It was a close game. The score was tied at halftime.
 [... 95,000 more tokens ...]
 In the final seconds, LeBron scored the winning basket.
 The Lakers won 98-97!"

┌────────────────────────────────────────────────────────────────┐
│ STEP 1: Segmentation (update_memory)                           │
└────────────────────────────────────────────────────────────────┘

Blocks created:
Block 1: "The Lakers played against the Celtics..."
Block 2: "It was a close game. The score was..."
...
Block 2950: "... [middle content] ..."
...
Block 3125: "In the final seconds, LeBron scored... Lakers won 98-97!"

┌────────────────────────────────────────────────────────────────┐
│ STEP 2: Storage (ContextManager.__init__)                      │
└────────────────────────────────────────────────────────────────┘

GPU Cache (128 blocks):
  - Block 1 (initial context)
  - Block 3123, 3124, 3125 (recent blocks)
  - Block 47, 89, 234, ... (recently accessed)

CPU Cache (500 blocks):
  - Block 2-46, 90-233, ... (warm cache)

Disk (all 3125 blocks):
  - Complete storage
  
┌────────────────────────────────────────────────────────────────┐
│ STEP 3: Query Processing (append)                              │
└────────────────────────────────────────────────────────────────┘

Query: "Who won the championship?"
Tokenized: [8241, 1747, 272, 22264, 28804]

Split into Q, K, V:
Q: [0.23, -0.45, 0.67, ...] (query representation)
K: [0.12, 0.89, -0.34, ...] (key representation)  
V: [0.56, -0.23, 0.91, ...] (value representation)

┌────────────────────────────────────────────────────────────────┐
│ STEP 4: Retrieval (_calc_topk_blocks)                          │
└────────────────────────────────────────────────────────────────┘

Compute similarity:
Block 1: Query similarity = 0.3 (mentions Lakers, Celtics - relevant!)
Block 2: Query similarity = 0.1 (just game details)
...
Block 3125: Query similarity = 0.95 (contains "won" - very relevant!)

Top-K selection (K=10):
Retrieved blocks: [3125, 3124, 3123, 1, 47, 89, 234, 567, 892, 1023]
                   ↑                    ↑
              Most relevant        Initial context

┌────────────────────────────────────────────────────────────────┐
│ STEP 5: Cross-Event Reasoning (CrossEventReasoner)             │
└────────────────────────────────────────────────────────────────┘

Cross-block attention:
Block 1 ↔ Block 3125: High similarity (both mention Lakers)
Block 3124 ↔ Block 3125: High similarity (sequential narrative)

Reweighting:
Block 3125: weight increased from 0.95 → 0.98 (connected to other relevant blocks)
Block 1: weight increased from 0.3 → 0.45 (connected to final answer)

┌────────────────────────────────────────────────────────────────┐
│ STEP 6: Multi-Stage Attention (_retrieve_and_attend)           │
└────────────────────────────────────────────────────────────────┘

Stage 1 - Local Attention:
Context: "Who won the championship?"
Attention over: ["Who", "won", "the", "championship", "?"]
Self-attention weights: championship ← won (strong connection)

Stage 2 - Global Attention:
Retrieved blocks loaded to GPU:
"The Lakers played..." [Block 1]
"In the final seconds... Lakers won 98-97!" [Block 3125]
...

Attention over combined context:
"Who won the championship?" + [Block 1] + [Block 3125] + ...

Attention focuses on:
- "Lakers won 98-97!" (highest weight)
- "The Lakers played" (context)
- "championship" (query word)

┌────────────────────────────────────────────────────────────────┐
│ STEP 7: Generation (GreedySearch._decode)                      │
└────────────────────────────────────────────────────────────────┘

Token by token generation:

Step 1:
Input: "Who won the championship? [context...]"
Logits: [0.01, 0.02, ..., 0.87 (The), ..., 0.03]
Generated: "The" 

Step 2:
Input: "... championship? [context...] The"
Logits: [0.02, ..., 0.91 (Lakers), ..., 0.04]
Generated: "Lakers"

Step 3:
Input: "... The Lakers"
Logits: [0.03, ..., 0.88 (won), ..., 0.05]
Generated: "won"

Final output: "The Lakers won the championship 98-97."

┌────────────────────────────────────────────────────────────────┐
│ STEP 8: Evaluation (scorer)                                    │
└────────────────────────────────────────────────────────────────┘

Prediction: "The Lakers won the championship 98-97."
Ground truth: ["The Lakers", "Lakers", "LA Lakers"]

F1 Score calculation:
Predicted tokens: ["The", "Lakers", "won", "the", "championship", "98-97"]
Ground truth tokens: ["The", "Lakers"]

Precision: 2/6 = 0.33 (2 correct out of 6 predicted)
Recall: 2/2 = 1.0 (found all 2 ground truth tokens)
F1: 2 * (0.33 * 1.0) / (0.33 + 1.0) = 0.50

Result: 50% F1 score (partial credit for correct answer with extra details)
```

## Configuration File Breakdown

### mistral.yaml - Key Settings

```yaml
# ═══════════════════════════════════════════════════════════════
# MAIN SETTINGS
# ═══════════════════════════════════════════════════════════════

max_len: 8192           # Maximum context length model can handle
                        # → If document is 10,000 tokens, only use last 8,192
                        
chunk_size: 128         # Process this many tokens at a time
                        # → Smaller = less memory, but slower
                        # → Larger = more memory, but faster

conv_type: mistral-inst # Chat template to use
                        # → Formats conversation properly for Mistral

# ═══════════════════════════════════════════════════════════════
# MEMORY SETTINGS (model.*)
# ═══════════════════════════════════════════════════════════════

model:
  type: em-llm          # Use EM-LLM enhancements
  path: mistralai/Mistral-7B-Instruct-v0.2  # Model location
  
  # BLOCK CONFIGURATION
  min_block_size: 8     # Minimum tokens per block
                        # → Don't create tiny blocks (inefficient)
                        
  max_block_size: 32    # Maximum tokens per block
                        # → Don't create huge blocks (won't fit many in memory)
  
  # MEMORY ALLOCATION
  n_init: 64            # Always keep first 64 tokens
                        # → Usually contains the question/prompt
                        
  n_local: 1024         # Keep recent 1,024 tokens in fast memory
                        # → Recent context is usually most relevant
                        
  n_mem: 512            # Retrieve up to 512 tokens from old blocks
                        # → Balance between context and speed
  
  # CACHE SETTINGS
  max_cached_block: 128 # Keep up to 128 blocks in GPU memory
                        # → More blocks = more context, but needs more memory
                        
  exc_block_size: 128   # Process 128 tokens at a time
                        # → Chunk size for incremental processing
  
  # RETRIEVAL SETTINGS
  repr_topk: 4          # Use top 4 tokens per head for block representation
                        # → More = better representation, but slower
  
  surprisal_threshold_gamma: 1.0  # Threshold for block boundaries
                                  # → Higher = fewer blocks (only very surprising tokens)
                                  # → Lower = more blocks (more granular)
  
  # OFFLOADING (memory management)
  disk_offload_threshold: 50000    # If input > 50k tokens, use disk offload
  vector_offload_threshold: 5000   # If input > 5k tokens, offload vectors
  min_free_cpu_memory: 50         # Keep at least 50GB CPU RAM free

# ═══════════════════════════════════════════════════════════════
# ADVANCED SETTINGS
# ═══════════════════════════════════════════════════════════════

  similarity_refinement_kwargs:
    similarity_refinement: true      # Use graph-based refinement
    refine_with_buffer: true        # Use contiguity buffer
    refine_from_layer: 20           # Start refinement from layer 20
    similarity_metric: modularity    # Metric for similarity

  contiguity_buffer_kwargs:
    use_contiguity_buffer: false    # Use neighbor blocks
    contiguity_buffer_size: 0.3     # 30% of budget for neighbors

  uniform_blocks: false              # Use surprisal-based segmentation
  random_topk_blocks: false          # Don't randomly select blocks
```

## Common Debugging Scenarios

### Scenario 1: Out of Memory Error

```
Error: RuntimeError: CUDA out of memory
```

**What's happening**: GPU memory is full

**Solutions**:

1. **Reduce cached blocks**:
```yaml
max_cached_block: 64  # Instead of 128
```

2. **Reduce local window**:
```yaml
n_local: 512  # Instead of 1024
```

3. **Enable more aggressive offloading**:
```yaml
disk_offload_threshold: 10000  # Instead of 50000
vector_offload_threshold: 2000  # Instead of 5000
```

4. **Reduce block size**:
```yaml
max_block_size: 16  # Instead of 32
```

### Scenario 2: Very Slow Processing

```
Processing takes hours for a single question
```

**What's happening**: Too much computation or disk I/O

**Solutions**:

1. **Increase chunk size**:
```yaml
chunk_size: 256  # Instead of 128
```

2. **Reduce retrieved memory**:
```yaml
n_mem: 256  # Instead of 512
```

3. **Use fewer blocks**:
```yaml
surprisal_threshold_gamma: 2.0  # Instead of 1.0 (creates fewer blocks)
```

4. **Keep more in GPU**:
```yaml
max_cached_block: 256  # If you have enough memory
```

### Scenario 3: Poor Answer Quality

```
Model gives wrong or incomplete answers
```

**What's happening**: Not retrieving relevant context

**Solutions**:

1. **Retrieve more context**:
```yaml
n_mem: 1024  # Instead of 512
```

2. **Use more blocks**:
```yaml
surprisal_threshold_gamma: 0.5  # Instead of 1.0 (creates more blocks)
```

3. **Enable cross-event reasoning**:
```python
enable_cross_event_reasoning: true
```

4. **Increase repr_topk**:
```yaml
repr_topk: 8  # Instead of 4 (better block representations)
```

### Scenario 4: Debugging Specific Components

**To debug block creation**:
```python
# In context_manager.py, in update_memory()
print(f"Creating block: start={remainder_st}, end={remainder_ed}")
print(f"Block size: {remainder_ed - remainder_st}")
print(f"Total blocks: {len(self.global_blocks[0])}")
```

**To debug retrieval**:
```python
# In context_manager.py, in _calc_topk_blocks()
for u in range(self.batch_size):
    scores = []
    for idx in sorted_block_idx:
        score = self.block_repr_k[u].get_similarity(global_q[u])
        scores.append(score.item())
    print(f"Top 10 block scores: {scores[:10]}")
```

**To debug generation**:
```python
# In greedy_search.py, in _decode()
if i > 0:  # Skip first iteration (prompt)
    print(f"Step {i}: Generated '{tokenizer.decode(word)}' "
          f"(token_id: {word.item()})")
    print(f"Top 5 alternatives: {logits.topk(5).indices.tolist()}")
```

## Quick Checklist for Running

- [ ] Python environment activated
- [ ] CUDA available (`torch.cuda.is_available() == True`)
- [ ] Model downloaded to cache (`~/.cache/huggingface/`)
- [ ] Dataset downloaded (`benchmark/data/`)
- [ ] Config file exists (`config/mistral.yaml`)
- [ ] Enough disk space (50GB+ recommended)
- [ ] Enough GPU memory (12GB+ recommended)

## Command Reference

```bash
# Run full pipeline
python run_pipeline_debug.py

# Run with different dataset
# Edit run_pipeline_debug.py: DATASET = "hotpotqa"

# Run in debug mode (VS Code)
# Press F5 or use "Debug EM-LLM Pipeline" configuration

# Check GPU usage
nvidia-smi

# Monitor GPU usage continuously
watch -n 1 nvidia-smi

# Check Python packages
pip list | grep torch
pip list | grep transformers

# Clear cache
rm -rf benchmark/results/mistral/long-bench/offload_data/
```

---

**This quick reference covers:**
✓ Visual architecture
✓ Complete data flow example
✓ Configuration breakdown
✓ Common debugging scenarios
✓ Component-specific debugging
✓ Running checklist
✓ Useful commands
