# EM-LLM Development Progress and Enhancements

## Latest Enhancement: Cross-Event Reasoning Integration

### Theoretical Background and Concepts

1. **Cross-Event Reasoning (CXR) Fundamentals**
   
   a) **Core Concept**
   - CXR is an unsupervised learning approach for discovering relationships between events
   - Events are represented as high-dimensional vectors in a shared embedding space
   - Relationships are learned through attention-based similarity patterns
   
   b) **Mathematical Foundation**
   ```python
   # Event Representation
   E = {e1, e2, ..., en} where ei ∈ ℝᵈ  # d-dimensional event vectors
   
   # Attention Score Computation
   score(qi, kj) = (qi · kj) / √d      # Scaled dot-product attention
   attention(Q, K, V) = softmax(QK^T/τ)V  # τ is temperature parameter
   ```

2. **Unsupervised Learning in CXR**
   
   a) **Self-Attention Mechanism**
   - Events learn from each other without explicit labels
   - Attention weights represent learned relationship strengths
   - Pattern discovery through multi-head attention:
     ```
     Each head learns different relationship types:
     - Temporal dependencies
     - Causal relationships
     - Contextual similarities
     ```

   b) **Probabilistic Framework**
   - Softmax converts attention scores to probability distributions:
     ```python
     P(relationship) = softmax(scores/temperature)
     = exp(score/T) / Σ(exp(scores/T))
     ```
   - Temperature (T) controls distribution sharpness:
     - Low T (< 1.0): More confident, focused relationships
     - High T (> 1.0): More exploratory, diverse relationships

3. **Weighted Probability in Event Selection**
   
   a) **Scoring Mechanism**
   ```python
   # Event importance is weighted sum of:
   score = α * similarity_score +    # Direct relevance
           β * relationship_score +   # Cross-event importance
           γ * position_score        # Temporal relevance
   
   where α + β + γ = 1  # Normalized weights
   ```

   b) **Dynamic Weight Adjustment**
   - Weights evolve based on:
     - Query type patterns
     - Historical success rates
     - Memory utilization metrics

4. **Advanced Concepts in Implementation**

   a) **Temperature-Based Control**
   ```python
   class CrossEventReasoner:
       def __init__(self, temperature=1.0):
           self.temperature = temperature
   
       def compute_attention(self, scores):
           # Controlled probability distribution
           return torch.softmax(scores/self.temperature, dim=-1)
   ```

   b) **Multi-Head Attention Theory**
   - Each head (h) learns specialized patterns:
   ```
   head_h = attention(QWq_h, KWk_h, VWv_h)
   where Wq_h, Wk_h, Wv_h are learned projections
   ```
   - Heads combine for comprehensive relationship modeling:
   ```python
   output = concat(head_1, ..., head_h)Wo
   ```

### What We Have Done

We have successfully integrated a cross-event reasoning mechanism into the EM-LLM model's context management system. The key enhancements include:

1. **Cross-Event Reasoner (CXR) Technical Implementation**

   a) **Core Architecture**
   ```python
   class CrossEventReasoner(nn.Module):
       def __init__(self, emb_dim, num_heads=4, summary_pool="mean"):
           # Multi-head attention for event-to-event reasoning
           self.event_self_attn = nn.MultiheadAttention(
               embed_dim=emb_dim,
               num_heads=num_heads,
               batch_first=True
           )
           # Event fusion layer for combining related events
           self.fusion_layer = nn.Sequential(
               nn.Linear(emb_dim * 2, emb_dim),
               nn.ReLU(),
               nn.Linear(emb_dim, emb_dim)
           )
   ```

   b) **Key Components**
   - Multi-head attention mechanism for event relationship processing
   - Event fusion module for combining strongly related events
   - Configurable pooling strategies for event representation
   - Score normalization and relationship strength calculation

2. **Context Manager Integration Details**

   a) **Initialization in ContextManager**
   ```python
   # In ContextManager.__init__
   self.enable_cross_event_reasoning = kwargs.get("enable_cross_event_reasoning", True)
   self.cross_event_heads = kwargs.get("cross_event_heads", 4)
   
   # Initialized after dimensions are known in _init
   self.cross_event_reasoner = CrossEventReasoner(
       emb_dim=dim_head * num_heads_kv,
       num_heads=self.cross_event_heads,
       summary_pool="mean"
   )
   ```

   b) **Integration Points**
   - Block retrieval process in `_retrieve_and_attend`
   - Event relationship processing during block selection
   - Memory management with relationship awareness
   - Cross-event attention computation in parallel CUDA stream

3. **Technical Implementation Details**

   a) **Event Processing Pipeline**
   ```python
   # In _retrieve_and_attend
   if hasattr(self, 'cross_event_reasoner'):
       # 1. Pool event representations
       pooled_events = [block.get()[0].mean(dim=1) for block in blocks]
       event_reps = torch.stack(pooled_events)
       
       # 2. Apply cross-event reasoning
       weights, scores, _ = self.cross_event_reasoner(
           event_reps,
           global_q.mean(dim=2)  # Current context
       )
       
       # 3. Reorder blocks based on relationship strength
       _, indices = torch.sort(weights, descending=True)
   ```

   b) **Memory Management Enhancement**
   - Relationship-aware block eviction strategy
   - Event fusion for memory efficiency
   - Cached relationship information for frequent patterns

### Cross-Event Reasoning: Technical Necessity and Implementation Benefits

1. **Problem Being Solved**
   ```
   Original Block Selection:
   Query -> [Block1][Block2][Block3] -> Individual Similarity Scores
   Limited: Misses inter-block relationships and patterns
   
   With CXR:
   Query -> [Block1 <-> Block2 <-> Block3] -> Relationship-Aware Scores
   Enhanced: Considers block relationships and context patterns
   ```

2. **Technical Benefits in Context Management**

   a) **Smarter Block Selection**
   ```python
   # Before: Simple similarity-based selection
   scores = compute_similarity(query, blocks)
   selected = top_k(scores)
   
   # After: Relationship-aware selection
   event_reps = pool_events(blocks)
   weights, scores = cross_event_reasoner(event_reps, query)
   selected = reorder_by_relationships(weights, scores)
   ```

   b) **Memory Optimization**
   - Fusion of related events reduces memory usage
   - Example: Two strongly related blocks (A, B)
   ```python
   # Without fusion: Store both blocks
   memory_used = size(A) + size(B)
   
   # With fusion: Store combined representation
   if relationship_strength(A, B) > fusion_threshold:
       memory_used = size(fuse(A, B))  # Typically < size(A) + size(B)
   ```

3. **Integration with Existing Architecture**

   a) **Context Manager Enhancement**
   ```python
   # Integration in block retrieval pipeline
   def _retrieve_and_attend(self, local_q, local_k, local_v, global_q):
       # 1. Regular similarity computation
       base_scores = compute_similarity(global_q, blocks)
       
       # 2. Cross-event relationship processing
       if self.enable_cross_event_reasoning:
           relationship_scores = self.process_relationships(blocks)
           final_scores = combine_scores(base_scores, relationship_scores)
       
       # 3. Block selection with relationship awareness
       selected_blocks = select_blocks(final_scores)
   ```

   b) **Memory Management Benefits**
   - Improved cache utilization through relationship-aware eviction
   - Better block reuse through pattern recognition
   - Reduced redundancy through smart fusion

4. **Performance Impact**
   ```
   Memory Efficiency:
   - Without CXR: O(n) blocks stored separately
   - With CXR: O(n-f) blocks where f = fused_blocks
   
   Processing Time:
   - Additional cost: O(h×n²) for n blocks, h heads
   - Offset by: Reduced memory operations, better cache hits
   ```

### Technical Benefits

1. **Memory Efficiency**
   - Event fusion reduces redundancy by combining strongly related events
   - Smarter block selection leads to better utilization of context window
   - Reduced memory footprint through intelligent event representation

2. **Computational Improvements**
   - Parallel processing of cross-event attention in dedicated CUDA stream
   - Efficient block reordering based on relationship strengths
   - Optimized memory management through enhanced LRU cache

### Focused 2-Week Implementation Plan

#### Week 1: Core Improvements and Evaluation

1. **Basic Evaluation Setup**
   - Implement simple F1 score calculation for event relationships
   - Set up basic evaluation pipeline focusing on:
     - Direct event relationship accuracy
     - Basic relationship type identification (temporal/causal)
   - Create baseline measurements with current implementation

2. **Essential CXR Enhancements** (3 days)
   - Add static thresholding for event fusion (simpler than adaptive)
   - Implement basic confidence scoring for relationships
   - Add simple positional bias in cross-event attention

3. **Basic Testing** (2 days)
   - Create core unit tests for new functionality
   - Test on a small subset of the dataset
   - Document initial results

#### Week 2: Optimization and Refinement

1. **Performance Optimization and Parameter Tuning**
   
   a) **Cross-Event Attention Parameters**
   - Fine-tune attention head count (current default: 4)
   ```python
   # Configurable parameters in CrossEventReasoner
   num_heads: Number of attention heads (default=4)
   - Lower -> Faster, more focused relationships
   - Higher -> Better relationship detection, more compute
   ```
   
   b) **Event Fusion Thresholds**
   ```python
   fusion_threshold: float = 0.5  # Current default
   # Determines when events should be combined
   # - Lower -> More aggressive fusion, better memory
   # - Higher -> More selective fusion, better precision
   ```
   
   c) **Memory and Performance Settings**
   ```python
   summary_pool: str = "mean"  # Options: mean, max
   enable_event_fusion: bool = True
   temperature: float = 1.0  # For attention softmax
   ```

2. **Focused Improvements**
   - Implement simple relationship verification
   - Add basic false positive filtering
   - Fine-tune threshold values based on results

3. **Documentation and Final Testing**
   - Complete essential documentation
   - Final performance testing
   - Prepare results summary

### Key Focus Areas for Maximum Impact

1. **Priority Features**
   - Basic event relationship scoring
   - Simple threshold-based fusion
   - Position-aware attention
   - Basic verification checks

2. **Implementation Strategy**
   - Keep implementations simple and modular
   - Focus on core functionality first
   - Use existing infrastructure where possible
   - Minimize complex dependencies

3. **Testing Approach**
   - Quick iteration cycles
   - Focus on key metrics only
   - Use small test sets initially
   - Gradual scaling of test coverage

---

## Previously Planned Improvements

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

---
