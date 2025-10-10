# EM-LLM Performance Enhancement Opportunities

**Purpose**: Documenting potential improvements to EM-LLM after validating baseline reproduction results.

---

## Important: Re-Enabling Paper Features, which ar disabled

### 1. Graph-Theoretic Boundary Refinement (Currently DISABLED)

**Current Status**: `similarity_refinement: false` in all configs

**Code Location**:
- Config: `config/llama31.yaml:101`, `config/mistral.yaml:101`
- Implementation: `em_llm/attention/similarity_refinement/similarity.py:1-93`
- Integration: `em_llm/attention/em_llm.py:212-265`

**What It Does**:
```yaml
similarity_refinement_kwargs:
  similarity_refinement: false  # ← DISABLED
  refine_with_buffer: true
  refine_from_layer: 20
  similarity_metric: modularity  # Options: modularity, conductance, intra_inter_sim
```

Refines surprise-based event boundaries using graph-theoretic metrics:
1. Computes adjacency matrix from layer activations (layers 20-31)
2. Tests boundary positions within ±50% window of initial surprise boundary
3. Selects position maximizing modularity (or minimizing conductance)
4. Ensures coherent event structure aligned with semantic similarity

**Algorithm** (`segmentation.py:4-50`):
```python
# For each surprise-detected boundary at position t:
1. Extract activation adjacency matrix A[t-window:t+window]
2. For each candidate position in window:
   - Split into two communities: [0, candidate), [candidate, T)
   - Compute modularity = (actual_edges - expected_edges) / total_edges
3. Select argmax(modularity) as refined boundary
```

**Computational Cost**:
- ~30% slower processing per chunk
- O(window_size × num_layers) matrix operations
- Line 245 in `em_llm.py`: `stacked_A = torch.einsum('blhtd,blhTd->btT', K, K)`

**Why Disabled**: Unknown - this is a core paper contribution (Figure 1, step ②)

**Expected Impact if Enabled**:
- **Accuracy**: +2-3 points on LongBench (based on human-aligned segmentation in paper)
- **Performance**: 30% slower, but more coherent event boundaries
- **Use Cases**: Critical for narrative understanding, multi-document QA

**Test Command**:
```bash
# Edit config/llama31.yaml:
similarity_refinement: true

# Then run:
bash scripts/run.sh --model llama31 --benchmark long-bench --world_size 2
```

---

### 2. Contiguity Buffer (Currently DISABLED)

**Current Status**: `use_contiguity_buffer: false` in all configs

**Code Location**:
- Config: `config/llama31.yaml:92-93`
- Implementation: `em_llm/attention/context_manager.py:284`

**What It Does**:
```yaml
contiguity_buffer_kwargs:
  use_contiguity_buffer: false  # ← DISABLED
  contiguity_buffer_size: 0.3    # 30% of n_mem reserved for temporal neighbors
```

Implements two-stage retrieval (Figure 1, step ④):
1. **Stage 1**: Retrieve top-k most similar events via k-NN (70% of n_mem budget)
2. **Stage 2**: For each retrieved event, also retrieve temporally adjacent events (30% budget)

**Cognitive Basis**:
Human episodic memory exhibits **temporal context effects** - recalling one event activates memories from nearby time periods.

**Example**:
```
Sequence: [Event_1][Event_2][Event_3]...[Event_100]
Query: "What happened after the meeting?"

Without contiguity buffer:
  Retrieved: Event_42, Event_7, Event_91 (by similarity only)

With contiguity buffer (size=0.3):
  Retrieved:
    - Similarity (70%): Event_42, Event_7, Event_91
    - Contiguity (30%): Event_43, Event_44, Event_8, Event_9
```

**Why Disabled**: Unknown - this is emphasized in paper abstract and Figure 1

**Expected Impact if Enabled**:
- **Accuracy**: +1-2 points on narrative QA (narrativeqa, qmsum)
- **Performance**: Minimal cost (just retrieves pre-computed adjacent blocks)
- **Use Cases**: Multi-hop reasoning, causality questions

**Test Command**:
```bash
# Edit config/llama31.yaml:
use_contiguity_buffer: true
contiguity_buffer_size: 0.3

# Then run:
bash scripts/run.sh --model llama31 --benchmark long-bench --world_size 2
```

---

### 3. Combined Configuration (Likely Paper Setting)

**Hypothesis**: Paper's reported 51.58 score uses EM-LLM_SM+C variant (Table 2)
- S = Surprise-based segmentation ✓ (enabled)
- M = Modularity refinement ✗ (disabled in configs)
- C = Contiguity buffer ✗ (disabled in configs)

**Recommended Test**:
```yaml
# config/llama31_full.yaml (create new config)
similarity_refinement: true
use_contiguity_buffer: true
contiguity_buffer_size: 0.3
surprisal_threshold_gamma: 1.0
```

**Expected Results**:
```
Config                    | Expected Score | Runtime
--------------------------|----------------|----------
Current (S only)          | 48-49          | Baseline
+ Modularity (SM)         | 50-51          | +30%
+ Contiguity (SM+C)       | 51-52          | +35%
```

---
