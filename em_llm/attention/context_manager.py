"""
Context retrieval and memory management utilities for long-context attention.

This module implements a block-based memory system for transformer attention layers.
It maintains Key/Value (KV) tensors for past tokens in a hierarchy spanning GPU
(quick access), CPU (warm cache), and disk (cold storage). The high-level flow is:

1) During each forward pass, the most recent tokens are kept in a local sliding window
   (n_local). In parallel, older tokens are grouped into MemoryBlocks (global blocks)
   built from the running global remainder stream and optionally segmented by surprisal.

2) The ContextManager ranks previously stored blocks using representation similarity
   (repr_score, derived from attention) and optionally Q–μK relevance (a head-aware
   cosine similarity between current mean Q heads and each block’s mean K heads).
   It then loads the top blocks back to GPU and forms a contiguous global buffer for
   multi-stage attention (local + recalled global, with complement sliding window).

3) The system manages placement across GPU, CPU, and disk, enforcing token budgets,
   LRU eviction, and optional CPU/disk offloading under constrained memory.

Main classes:
- CudaCache:          A fixed-size allocator over a pre-allocated torch.Tensor buffer
                      (device can be CUDA or CPU). Used for pooling and reusing
                      contiguous memory for MemoryBlock data.
- MemoryBlock:        Encapsulates a block of KV tensors. It can live in GPU cache,
                      CPU cache, or be offloaded to disk as .pt shards. Provides
                      load()/get()/offload() to move data as needed.
- VectorTensor:       A dynamically growing 2D tensor array with similarity utilities,
                      used to store per-block representations for ranking.
- ContextManager:     Layer-level orchestrator. Builds blocks, tracks caches,
                      computes rankings, retrieves blocks, and runs attention.

Notes:
- Q–μK scoring: we compute a per-block mean K (μK) over its span and compare to the
  live mean Q heads of the current forward pass for head-aware relevance.
- Surprisal blocking: optional segmentation logic that divides the global stream
  into blocks based on surprisal spikes (or by uniform chunking if enabled).
- Offloading: CPU and disk offload are optional; file structure is sharded for scale.

This module is framework-agnostic except for PyTorch, and is designed to be called by
patched transformer attention code (e.g., via TorchMultiStageDotProductAttention).
"""

import os
import random
import functools
import psutil
import torch
from typing import Optional, Tuple
from .dot_product_attention import TorchMultiStageDotProductAttention
from .cross_events import CrossEventReasoner



class CudaCache:
    """A simple fixed-size memory pool backed by a single pre-allocated tensor.

    The cache exposes `alloc()` / `delete()` to carve out and return contiguous
    slices ("units") of the backing storage. It does not implement compaction—
    indices are reused via an idle set.

    Parameters
    ----------
    num_units : int
        Number of allocatable units.
    unit_size : int
        Number of elements in each unit (flattened). The backing tensor shape is
        (num_units, unit_size).
    max_block_size : int
        Maximum token length for any MemoryBlock. Used to reshape views.
    dtype : torch.dtype
        Data type for the backing tensor.
    device : str or torch.device, default 'cuda'
        Device where the backing tensor is allocated. Use 'cpu' for CPU caches.

    Attributes
    ----------
    data : torch.Tensor
        The backing tensor of shape (num_units, unit_size).
    idle_set : set[int]
        Indices of free units available for allocation.
    """

    def __init__(self, num_units, unit_size, max_block_size, dtype, device='cuda',
                 qk_retrieval: bool = True, qk_weight: float = 1.0, qk_top_h: int = 0):
        # The qk_* arguments are accepted for constructor symmetry but unused here.
        self.qk_retrieval = qk_retrieval
        self.qk_weight = float(qk_weight)
        self.qk_top_h = int(qk_top_h)
        self._live_q_heads = None  # (H, Dh) — unused in CudaCache

        self.num_units = num_units
        self.unit_size = unit_size
        self.dtype = dtype
        self.data = torch.empty((num_units, unit_size), device=device, dtype=dtype)
        self.idle_set = set(list(range(num_units)))
        self.max_block_size = max_block_size

    def alloc(self):
        """Allocate a free unit from the pool.

        Returns
        -------
        (view, idx) : Tuple[torch.Tensor, int]
            A 1D view tensor of length `unit_size` and the index of the unit.

        Raises
        ------
        AssertionError
            If there are no free units left.
        """
        assert len(self.idle_set) > 0, "No more idle units in cache."
        idx = self.idle_set.pop()
        return self.data[idx], idx

    def delete(self, idx):
        """Return a previously allocated unit back to the idle set.

        Parameters
        ----------
        idx : int
            Unit index returned by `alloc()`.
        """
        assert idx not in self.idle_set
        self.idle_set.add(idx)


class MemoryBlock:
    """Container for a single block of K/V tensors with hierarchical residency.

    Each block may reside on:
    - GPU (through a `CudaCache` on a CUDA device) for fast attention reuse,
    - CPU (through a `CudaCache` on CPU) as a warm cache,
    - Disk (two .pt files for K and V) as cold storage.

    It supports moving between tiers via `load()`, `get()`, `offload()`, and
    `offload_to_disk()`. Disk files are cleaned up on deletion if enabled.

    Parameters
    ----------
    kv : Tuple[torch.Tensor, torch.Tensor]
        Tensors shaped (H_kv, span, Dh) for K and V.
    cache : CudaCache
        GPU cache pool used when loading block data onto the device.
    load_to_cache : bool, default False
        If True, immediately allocate a GPU unit and copy `kv` onto it.
    pin_memory : bool, default False
        If True, pin the CPU copies to speed up H2D transfers.
    allow_disk_offload : bool, default False
        If True, enable disk offloading and file cleanup.
    offload_dir : str, default './offload_data'
        Base directory for on-disk shards.
    load_to_disk : bool, default False
        If True, store the initial CPU representation directly to disk and
        free CPU memory.
    cpu_cache : Optional[CudaCache], default None
        CPU cache pool. If provided, CPU copies are stored in this pool; else
        they live as standalone CPU tensors.

    Attributes
    ----------
    cpu_data : Optional[Tuple[torch.Tensor, torch.Tensor]]
        CPU copies if resident on CPU cache or standalone tensors.
    gpu_data : Optional[torch.Tensor]
        View over a GPU cache unit shaped as (2, H_kv, max_block_size, Dh),
        containing K at index 0 and V at index 1. Only the first `size`
        positions are valid for this block.
    size : int
        Actual token span of this block.
    num_heads_kv : int
        Number of KV heads.
    dim_head : int
        Per-head dimension.
    on_disk : bool
        Whether the block has been offloaded to disk.
    """

    # individual event's KV store
    _instance_counter = 0

    def __init__(
        self,
        kv: Tuple[torch.Tensor, torch.Tensor],
        cache: CudaCache,
        load_to_cache: bool = False,
        pin_memory: bool = False,
        allow_disk_offload: bool = False,
        offload_dir: str = "./offload_data",
        load_to_disk: bool = False,
        cpu_cache: Optional[CudaCache] = None,
    ):
        # Disk offload setup (first-phase directory sharding)
        if allow_disk_offload is not False:
            self.allow_disk_offload = True
            self.id = MemoryBlock._instance_counter
            MemoryBlock._instance_counter += 1
            self.offload_dir = offload_dir + f"/{self.id // 10000}"
            os.makedirs(self.offload_dir + "/0", exist_ok=True)
            os.makedirs(self.offload_dir + "/1", exist_ok=True)
        else:
            self.allow_disk_offload = False

        num_heads_kv, size, dim_head = kv[0].shape
        self.cache = cache
        self.cpu_cache = cpu_cache
        self.on_disk = False
        assert size <= self.cache.max_block_size

        # host (CPU) copies
        if load_to_disk:
            torch.save(
                kv[0].contiguous(),
                os.path.join(self.offload_dir, f"0/{self.id}.pt"),
                pickle_protocol=4,
            )
            torch.save(
                kv[1].contiguous(),
                os.path.join(self.offload_dir, f"1/{self.id}.pt"),
                pickle_protocol=4,
            )
            self.on_disk = True
            cpu_data = None
            self.cpu_data_id = None
        elif cpu_cache is not None:
            cpu_data, cpu_data_id = cpu_cache.alloc()
            cpu_data = cpu_data.view((2, num_heads_kv, self.cache.max_block_size, dim_head))
            cpu_data[0][:, :size, :].copy_(kv[0].contiguous(), non_blocking=True)
            cpu_data[1][:, :size, :].copy_(kv[1].contiguous(), non_blocking=True)
            self.cpu_data_id = cpu_data_id
        else:
            if kv[0].is_cuda:
                cpu_data = tuple(_t.contiguous().to("cpu", non_blocking=True) for _t in kv)
            else:
                cpu_data = tuple(_t.contiguous() for _t in kv)
            if pin_memory:
                cpu_data = tuple(_t.pin_memory() for _t in cpu_data)

        # device (GPU) copies
        if load_to_cache:
            gpu_data, gpu_data_id = cache.alloc()
            gpu_data = gpu_data.view((2, num_heads_kv, self.cache.max_block_size, dim_head))
            gpu_data[0][:, :size, :].copy_(kv[0], non_blocking=True)
            gpu_data[1][:, :size, :].copy_(kv[1], non_blocking=True)
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream())
        else:
            gpu_data, gpu_data_id, event = None, None, None

        self.cpu_data = cpu_data
        self.gpu_data = gpu_data
        self.gpu_data_id = gpu_data_id
        self.event = event
        self.size = size
        self.num_heads_kv = num_heads_kv
        self.dim_head = dim_head
        self.pin_memory = pin_memory

        # Disk offload setup (second-phase base dir)
        if allow_disk_offload is not False:
            self.allow_disk_offload = True
            self.offload_dir = offload_dir
            os.makedirs(self.offload_dir, exist_ok=True)
            if load_to_disk:
                self.offload_to_disk()
        else:
            self.allow_disk_offload = False

    def __del__(self):
        """Clean up disk shards if disk offload is enabled."""
        if hasattr(self, "allow_disk_offload") and self.allow_disk_offload:
            self._delete_from_disk()

    def load(self, target: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, load_cache: bool = True):
        """Ensure the block is resident on GPU, copying from CPU/disk if needed.

        Parameters
        ----------
        target : Optional[Tuple[torch.Tensor, torch.Tensor]]
            Optional pre-allocated (K, V) slices on GPU to copy into. Shapes must
            be (H_kv, size, Dh). If provided, we copy into both `target` and the
            GPU cache unit to avoid redundant reads.
        load_cache : bool
            Unused flag kept for interface compatibility.

        Returns
        -------
        (loaded, target_event) : Tuple[bool, Optional[torch.cuda.Event]]
            loaded = True if we allocated a new GPU cache unit this call.
            target_event is a CUDA event recorded after copies into `target`.

        Raises
        ------
        AssertionError
            If CPU data is unexpectedly missing and not on disk.
        """
        target_event = None
        if self.cpu_data is None:
            assert self.on_disk, "CPU data is None but on_disk is also set to False"
            self._load_from_disk()
        num_heads_kv, _, dim_head = self.cpu_data[0].shape
        if target is not None:
            assert target[0].shape == (num_heads_kv, self.size, dim_head)

        if self.gpu_data is not None:
            if target is not None:
                target[0].copy_(self.gpu_data[0][:, :self.size, :], non_blocking=True)
                target[1].copy_(self.gpu_data[1][:, :self.size, :], non_blocking=True)
                target_event = torch.cuda.Event()
                target_event.record(torch.cuda.current_stream())
            return False, target_event

        gpu_data, gpu_data_id = self.cache.alloc()
        gpu_data = gpu_data.view((2, num_heads_kv, self.cache.max_block_size, dim_head))

        if target is not None:
            target[0].copy_(self.cpu_data[0][:, :self.size, :], non_blocking=True)
            target[1].copy_(self.cpu_data[1][:, :self.size, :], non_blocking=True)
            target_event = torch.cuda.Event()
            target_event.record(torch.cuda.current_stream())
            gpu_data[0][:, :self.size, :].copy_(target[0], non_blocking=True)
            gpu_data[1][:, :self.size, :].copy_(target[1], non_blocking=True)
        else:
            gpu_data[0][:, :self.size, :].copy_(self.cpu_data[0][:, :self.size, :], non_blocking=True)
            gpu_data[1][:, :self.size, :].copy_(self.cpu_data[1][:, :self.size, :], non_blocking=True)

        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        self.event = event
        self.gpu_data = gpu_data
        self.gpu_data_id = gpu_data_id

        return True, target_event

    def get(self):
        """Return the resident GPU data (K/V) for this block.

        Waits for the copy `event` before returning to ensure data readiness.

        Returns
        -------
        torch.Tensor
            A view shaped as (2, H_kv, size, Dh) containing K then V.
        """
        assert self.gpu_data is not None
        self.event.wait()
        return self.gpu_data[:, :, :self.size, :]

    def offload(self):
        """Drop the GPU residency of this block and free the cache unit."""
        assert self.gpu_data is not None
        self.event.wait()
        self.gpu_data = None
        self.cache.delete(self.gpu_data_id)
        self.gpu_data_id = None

    def offload_to_disk(self):
        """Persist CPU data to disk as two .pt files and free CPU cache memory."""
        if not self.on_disk:
            torch.save(
                self.cpu_data[0][:, :self.size, :].clone(),
                os.path.join(self.offload_dir, f"0/{self.id}.pt"),
                pickle_protocol=4,
            )
            torch.save(
                self.cpu_data[1][:, :self.size, :].clone(),
                os.path.join(self.offload_dir, f"1/{self.id}.pt"),
                pickle_protocol=4,
            )
            self.on_disk = True
        self.cpu_data = None
        self.cpu_cache.delete(self.cpu_data_id)
        self.cpu_data_id = None

    def _load_from_disk(self):
        """Load CPU tensors from disk into the CPU cache."""
        self.cpu_data, self.cpu_data_id = self.cpu_cache.alloc()
        self.cpu_data = self.cpu_data.view((2, self.num_heads_kv, self.cache.max_block_size, self.dim_head))
        self.cpu_data[0, :, :self.size, :].copy_(
            torch.load(os.path.join(self.offload_dir, f"0/{self.id}.pt"), map_location="cpu"), non_blocking=True
        )
        self.cpu_data[1, :, :self.size, :].copy_(
            torch.load(os.path.join(self.offload_dir, f"1/{self.id}.pt"), map_location="cpu"), non_blocking=True
        )

    def _delete_from_disk(self):
        """Delete on-disk shards if present."""
        if self.on_disk:
            os.remove(os.path.join(self.offload_dir, f"0/{self.id}.pt"))
            os.remove(os.path.join(self.offload_dir, f"1/{self.id}.pt"))


class VectorTensor:
    """A growable 2D tensor with similarity utilities for ranking blocks.

    Used to store per-block representation vectors (e.g., averaged repr K per block)
    and compute similarity against a query (e.g., the current step’s mean Q).

    Parameters
    ----------
    hidden_size : int
        Vector dimensionality per row.
    element_dtype : torch.dtype
        Dtype for the tensor storage.
    layer_idx : int
        Layer identifier (for logging/debugging).
    device : str or torch.device, default 'cuda'
        Device to store the data.
    """

    def __init__(self, hidden_size, element_dtype, layer_idx, device="cuda"):
        init_cached_size = 16
        self.data = torch.empty((init_cached_size, hidden_size), dtype=element_dtype, device=device)
        self.length = 0
        self.cache_size = init_cached_size
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx

    def append_cache(self):
        """Increase underlying capacity by a fixed step to amortize reallocations."""
        new_cache_size = self.cache_size + 128
        data_shape = self.data.shape
        new_data = torch.empty((new_cache_size,) + data_shape[1:], device=self.data.device, dtype=self.data.dtype)
        new_data[: self.cache_size, ...].copy_(self.data)
        self.data = new_data
        self.cache_size = new_cache_size

    def append(self, tensor: torch.Tensor):
        """Append one or more vectors to the end of the array.

        Parameters
        ----------
        tensor : torch.Tensor
            2D tensor of shape (N, hidden_size), contiguous and dtype-matching.
        """
        assert tensor.dtype == self.data.dtype
        assert tensor.size(1) == self.hidden_size
        assert tensor.is_contiguous()
        append_l = tensor.size(0)
        while self.length + append_l > self.cache_size:
            self.append_cache()
        self.data[self.length : self.length + append_l, ...].copy_(tensor)
        self.length += append_l

    def get_data(self):
        """Return a view of valid rows currently stored."""
        return self.data[: self.length, ...]

    def get_similarity(self, tensor: torch.Tensor):
        """Compute dot-product similarity between all rows and a 1D query vector.

        Parameters
        ----------
        tensor : torch.Tensor
            1D vector of shape (hidden_size,).

        Returns
        -------
        torch.Tensor
            1D tensor of length `len(self)` containing similarities.
        """
        assert tensor.dim() == 1 and tensor.size(0) == self.hidden_size
        logits = torch.matmul(self.data[: self.length], tensor[:, None].to(self.data.device)).squeeze(dim=-1)
        assert logits.dim() == 1 and logits.size(0) == self.length
        return logits

    def get_topk(self, tensor: torch.Tensor, topk):
        """Return indices of the top-k most similar rows to the query vector."""
        logits = self.get_similarity(tensor)
        return logits.topk(topk, dim=0).indices

    def sort_by_similarity(self, tensor: torch.Tensor):
        """Return row indices sorted by similarity to the query vector (desc)."""
        logits = self.get_similarity(tensor)
        return torch.sort(logits, descending=True).indices.cpu().tolist()

    def __len__(self):
        """Number of valid rows currently stored."""
        return self.length


GLOBAL_STREAM = None

class ContextManager:
    """Layer-level orchestrator for block creation, ranking, retrieval, and attention.

    Typical cycle per forward pass:
    1) `_init()` once to set shapes, buffers, caches.
    2) `append()` with the most recent local/global QKV slices to extend the
       running global remainder and perform retrieval+attention.
    3) `update_memory()` to segment the global remainder into blocks (via surprisal
       or uniform) and finalize residency/offloading decisions.

    Parameters
    ----------
    layer_idx : int
        Layer identifier for logging/debugging.
    position_embedding : object
        Provides rotary position methods: `apply_rotary_pos_emb_one_angle` and
        `_update_cos_sin_tables_len`.
    n_init : int
        Initial global tokens to always include (prefix) when length > n_local.
    n_local : int
        Sliding window length used as the local context.
    max_block_size : int
        Upper bound on block span (tokens) used during segmentation and storage.
    max_cached_block : int
        Maximum number of blocks that can be resident in the GPU cache per batch.
    exc_block_size : int
        Max number of new tokens appended per `append()` call (sanity check).
    min_block_size : int, default 1
        Minimum allowable block span when segmenting.
    async_global_stream : bool, default True
        Use a dedicated CUDA stream for global copies and block management.
    pin_memory : bool, default False
        Pin host memory buffers for faster H2D copies.
    perhead : bool, default False
        If True, expands QKV to per-head batch for attention (debug/experiments).
    repr_topk : int, default 1
        Number of top tokens per head used to form each block’s representation K.
    surprisal_threshold_gamma : float, default 1.1
        Multiplier on std-dev above mean to mark segmentation boundaries.
    n_mem : int, default 2048
        Budget for recalled global tokens (beyond n_init and exc_block_size).
    uniform_blocks : bool, default False
        If True, segment by fixed-size chunks rather than surprisal.
    random_topk_blocks : bool, default False
        If True, pick blocks randomly instead of similarity ranking (ablations).
    similarity_refinement : bool, default False
        Placeholder for graph-theoretic refinements (not active in this code).
    refine_with_buffer : bool, default False
        Placeholder toggle for refinement with a contiguity buffer.
    refine_from_layer : int, default 0
        Placeholder layer index from which refinements apply.
    similarity_metric : str, default 'modularity'
        Placeholder string name for refinement metric.
    use_contiguity_buffer : bool, default False
        If True, allocate part of budget to neighbors of selected blocks.
    contiguity_buffer_size : float, default 0.3
        Buffer size in tokens (int) or fraction (0..1) of remaining budget.
    use_hf_acc : bool, default False
        If True, move certain tensors to the right device when using HF accelerators.
    disk_offload_dir : str, default './offload_data'
        Directory for block shards if disk offload is enabled.
    allow_disk_offload : bool, default False
        Toggle for CPU/disk hierarchical offload.
    vector_offload : bool, default False
        If True and single-GPU, offload VectorTensor to CPU to reduce VRAM.

    Additional kwargs
    ------------------
    min_free_cpu_memory : float (GB), default 100
        Guardrail for CPU memory when allocating CPU cache.
    world_size : int, default depends on GPUs
        Controls CPU cache sizing per process.
    qk_retrieval : bool, default True
        Enable head-aware Q–μK relevance scoring.
    qk_weight : float, default 1.0
        Weighting for combining Q–μK with representation similarity (hook).
    qk_top_h : int, default 0
        If >0, use top-H head similarities per block; else use max over heads.
    """

    # """
    # Orchestrates everything for one layer: builds blocks from the stream, maintains caches and offloading,
    # computes similarity ranking and picks which blocks to recall, loads recalled KV into a contiguous global
    # buffer used for attention.
    # """

    def __init__(
        self,
        layer_idx,
        position_embedding,
        n_init,
        n_local,
        max_block_size,
        max_cached_block,
        exc_block_size,
        min_block_size: int = 1,
        async_global_stream: bool = True,
        pin_memory: bool = False,
        perhead: bool = False,
        repr_topk: int = 1,
        surprisal_threshold_gamma: float = 1.1,
        n_mem=2048,
        uniform_blocks: bool = False,
        random_topk_blocks: bool = False,
        similarity_refinement: bool = False,
        refine_with_buffer: bool = False,
        refine_from_layer: int = 0,
        similarity_metric: str = "modularity",
        use_contiguity_buffer: bool = False,
        contiguity_buffer_size: float = 0.3,
        use_hf_acc: bool = False,
        disk_offload_dir: str = "./offload_data",
        allow_disk_offload: bool = False,
        vector_offload: bool = False,
        **kwargs,
    ):
        self.length = 0
        self.position_embedding = position_embedding
        self.n_init = n_init
        self.n_local = n_local
        self.max_block_size = max_block_size
        self.min_block_size = min_block_size
        self.repr_topk = repr_topk
        self.max_cached_block = max_cached_block
        self.exc_block_size = exc_block_size
        assert exc_block_size <= n_local
        self.Attn = TorchMultiStageDotProductAttention
        self.initialized = False
        self.load_count = 0
        self.async_global_stream = async_global_stream
        self.pin_memory = pin_memory
        self.perhead = perhead
        self.global_context_cap = n_init + exc_block_size + n_mem
        self.surprisal_threshold_gamma = surprisal_threshold_gamma
        self.layer_idx = layer_idx
        self.random_topk_blocks = random_topk_blocks
        self.uniform_blocks = uniform_blocks
        self.similarity_refinement = similarity_refinement
        self.refine_with_buffer = refine_with_buffer
        self.refine_from_layer = refine_from_layer
        self.similarity_metric = similarity_metric
        self.use_contiguity_buffer = use_contiguity_buffer
        self.contiguity_buffer_size = contiguity_buffer_size

        self.use_hf_acc = use_hf_acc
        self.disk_offload_dir = disk_offload_dir
        self.allow_disk_offload = False if allow_disk_offload is False else None
        self.vector_offload = vector_offload
        if self.allow_disk_offload is None:
            self.min_free_cpu_memory = kwargs.get("min_free_cpu_memory", 100)  # GB
            world_size = 4 if torch.cuda.device_count() == 1 else 2
            self.world_size = kwargs.get("world_size", world_size)

        global GLOBAL_STREAM
        if self.async_global_stream and GLOBAL_STREAM is None:
            GLOBAL_STREAM = torch.cuda.Stream()

        self.max_total_retrieved_tokens = self.global_context_cap - exc_block_size - n_init
        assert (
            self.max_total_retrieved_tokens <= max_cached_block * min_block_size
        ), f"Not enough cached blocks to fit {self.max_total_retrieved_tokens} tokens."

        # live q-heads (set per forward by the caller)
        self._live_q_heads = None
        # retrieval weighting (optional)
        self.qk_retrieval = kwargs.get("qk_retrieval", True)
        self.qk_weight = float(kwargs.get("qk_weight", 1.0))
        self.qk_top_h = int(kwargs.get("qk_top_h", 0))
        
        # Cross-event reasoning configuration (will initialize reasoner in _init when dims are known)
        self.enable_cross_event_reasoning = kwargs.get("enable_cross_event_reasoning", True)
        self.cross_event_heads = kwargs.get("cross_event_heads", 4)
        self.cross_event_dropout = kwargs.get("cross_event_dropout", 0.0)
        self.cross_event_temperature = kwargs.get("cross_event_temperature", 1.0)
        self.cross_event_enable_fusion = kwargs.get("cross_event_enable_fusion", True)
        self.cross_event_fusion_threshold = kwargs.get("cross_event_fusion_threshold", 0.5)
        self.cross_event_similarity_metric = kwargs.get("cross_event_similarity_metric", "dot_product")
        self.cross_event_reasoner = None  # Will be initialized in _init()

    def set_live_q_heads(self, q_heads):
        """Set per-layer mean Q heads (H, Dh) for current forward."""
        if q_heads is not None:
            self._live_q_heads = torch.nn.functional.normalize(q_heads.detach(), dim=-1)
        else:
            self._live_q_heads = None

    def _init(self, local_q, local_k, local_v, global_q, global_k, global_v):
        # NOTE: batch_size must be read before using it anywhere else
        assert local_q.dim() == 4
        batch_size, num_heads, len_q, dim_head = local_q.shape
        num_heads_kv = local_k.size(1)

        for _t in [local_q, local_k, local_v, global_q, global_k, global_v]:
            assert _t.size(0) == batch_size
            assert (_t.size(1) == num_heads or _t.size(1) == num_heads_kv)
            assert _t.size(2) == len_q
            assert _t.size(3) == dim_head
            assert _t.is_cuda

        self.batch_size = batch_size
        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv
        self.unit_size_kv = num_heads_kv
        self.dim_head = dim_head

        self.global_blocks = [[] for _ in range(self.batch_size)]
        self.cached_blocks = [{} for _ in range(self.batch_size)]
        if self.allow_disk_offload is not False:
            self.block_usage = [{} for _ in range(self.batch_size)]
        self.num_global_block = 0

        if self.use_contiguity_buffer:
            self.contiguity_buffer = [[] for _ in range(self.batch_size)]

        # Representative K (mean of top-k tokens per block) used for similarity retrieval
        self.block_repr_k = [
            VectorTensor(dim_head * self.num_heads, global_k.dtype, self.layer_idx, device=local_k.device)
            for _ in range(self.batch_size)
        ]

        # Per-block mean-K signatures for optional Q–μK retrieval
        self.block_muK = [[] for _ in range(self.batch_size)]

        # Local KV cache
        self.local_k = torch.empty((self.batch_size, self.num_heads_kv, 0, dim_head), dtype=local_k.dtype, device=local_k.device)
        self.local_v = torch.empty((self.batch_size, self.num_heads_kv, 0, dim_head), dtype=local_v.dtype, device=local_v.device)

        # Global remainder buffers
        self.global_remainder = (
            torch.empty((self.batch_size, self.num_heads_kv, 0, dim_head), dtype=global_k.dtype, device=global_k.device),
            torch.empty((self.batch_size, self.num_heads_kv, 0, dim_head), dtype=global_k.dtype, device=global_k.device),
            torch.empty((self.batch_size, self.num_heads, 0, dim_head), dtype=global_q.dtype, device=global_q.device),
        )

        self.global_remainder_surprisal = torch.empty((self.batch_size, 0), dtype=global_k.dtype, device=global_k.device)
        self.global_remainder_repr_score = torch.empty(
            (self.batch_size, self.num_heads, 0), dtype=global_k.dtype, device=global_k.device
        )
        self.global_block_divide = torch.empty((self.batch_size, 0), dtype=torch.bool, device=global_k.device)
        self.global_remainder_repr_score_buffer = torch.empty(
            (self.batch_size, self.num_heads, self.max_block_size), dtype=global_k.dtype, device=global_k.device
        )
        self.global_remainder_k_buffer = torch.empty(
            (self.batch_size, self.num_heads_kv, self.max_block_size, self.dim_head), dtype=global_k.dtype, device=global_k.device
        )

        # Initial tokens (attention sink)
        self.init_k = torch.empty((self.batch_size, self.num_heads_kv, 0, dim_head), dtype=global_k.dtype, device=global_k.device)
        self.init_v = torch.empty((self.batch_size, self.num_heads_kv, 0, dim_head), dtype=global_k.dtype, device=global_k.device)
        self.init_exc = False

        self.dtype = local_q.dtype
        self.position_embedding._update_cos_sin_tables_len(
            self.n_local + self.exc_block_size + 1, local_k.device, local_k.dim()
        )
        
        # Initialize cross-event reasoner now that we have dimensions
        if self.enable_cross_event_reasoning:
            from .cross_events import CrossEventReasoner
            cross_event_emb_dim = dim_head * self.num_heads

            self.cross_event_reasoner = CrossEventReasoner(
                emb_dim=cross_event_emb_dim,
                num_heads=self.cross_event_heads,
                summary_pool="mean",
                dropout=self.cross_event_dropout,
                temperature=self.cross_event_temperature,
                enable_event_fusion=self.cross_event_enable_fusion,
                fusion_threshold=self.cross_event_fusion_threshold,
                similarity_metric=self.cross_event_similarity_metric,
            )

            # Run CrossEventReasoner on CPU to avoid GPU OOM
            self.cross_event_reasoner.to("cpu")
            self.cross_event_reasoner_device = torch.device("cpu")

            if self.layer_idx == 0:
                print(
                    f"Initialized CrossEventReasoner with emb_dim={cross_event_emb_dim}, "
                    f"num_heads={self.cross_event_heads}, device={local_k.device}"
                )

        # Retrieved KV memory buffer
        buffer_len = self.global_context_cap + 2 * self.max_block_size
        self.global_buffer = torch.empty(
            (2, self.batch_size, self.num_heads_kv, buffer_len, dim_head),
            dtype=global_k.dtype,
            device=global_k.device,
        )
        self.global_buffer_init_st = 0
        self.global_buffer_init_ed = 0

        # Cached memory blocks
        cuda_cache_device = local_k.device if self.use_hf_acc else "cuda"
        self.cuda_cache = CudaCache(
            self.max_cached_block * self.batch_size,
            self.unit_size_kv * self.max_block_size * dim_head * 2,
            self.max_block_size,
            local_k.dtype,
            device=cuda_cache_device,
        )
        self.create_memory_block = functools.partial(
            MemoryBlock, pin_memory=self.pin_memory, offload_dir=self.disk_offload_dir
        )

        self.cpu_cache = None
        self.global_length = [0 for _ in range(self.batch_size)]
        self.initialized = True

    def _init_cpu_cache(self, local_q, local_k):
        """
        Initialize a CPU-resident cache for KV blocks when disk offload is enabled.

        What it does
        ------------
        - Computes the maximum CPU memory we can use (respecting a minimum free-memory guard).
        - Estimates the per-token KV memory footprint and derives how many blocks can fit on CPU.
        - Allocates a CudaCache-like structure on CPU to hold offloaded KV blocks.

        Parameters
        ----------
        local_q : torch.Tensor
            Local query tensor with shape (B, H, T_local, Dh). Used to read head dim (Dh).
        local_k : torch.Tensor
            Local key tensor with shape (B, H_kv, T_local, Dh). Used to estimate token size.

        Side Effects
        ------------
        - Sets self.max_cpu_cached_blocks.
        - Creates and assigns self.cpu_cache (on CPU).
        - Logs one-time info when layer_idx == 0.
        """
        _, _, _, dim_head = local_q.shape

    # Guardrail: keep at least min_free_cpu_memory GiB free across all ranks.
        max_cpu_cache_memory = int(
            max(
                self.min_free_cpu_memory * (1024**3),
                (psutil.virtual_memory().available - self.min_free_cpu_memory * (1024**3)) / self.world_size,
            )
        )

        # Approx bytes per token for one head-group slice (same dtype as K)
        token_size = local_k[:, 0, :].element_size() * local_k[:, 0, :].numel()

        # How many blocks of size `max_block_size` (K and V => ×2) fit into the budget
        self.max_cpu_cached_blocks = max_cpu_cache_memory // (token_size * 2 * self.max_block_size)

        if self.layer_idx == 0:
            print(f"Initialising CPU cache. Number of blocks allocated: {self.max_cpu_cached_blocks}")

        # Each cache "unit" is sized for one block worth of K and V across all heads
        self.cpu_cache = CudaCache(
            self.max_cpu_cached_blocks * self.batch_size,
            self.unit_size_kv * self.max_block_size * dim_head * 2,
            self.max_block_size,
            local_k.dtype,
            device=torch.device("cpu"),
        )


    def _offload_vector(self):
        """
        Offload block-representation vectors (used for retrieval ranking) to CPU.

        Notes
        -----
        - This reduces GPU memory pressure for very long contexts on single-GPU runs.
        - Only touches the small per-block vectors, not full KV blocks.
        """
        if self.layer_idx == 0:
            print("Offloading VectorTensor to CPU to run single-GPUs on long-contexts.")
        for u in range(self.batch_size):
            # Move similarity vectors used by the retriever onto CPU
            self.block_repr_k[u].data = self.block_repr_k[u].data.to(torch.device("cpu"))


    def _num_memory_blocks(self):
        """
        Return the number of global memory blocks currently tracked for batch index 0.

        Returns
        -------
        int
            Count of memory blocks stored for u=0 (all batches should be aligned).
        """
        return len(self.global_blocks[0])


    def _remove_lru_blocks(self, u, num_remove: Optional[int] = None, ignore_blocks=None):
        """
        Evict least-recently-used (LRU) blocks from GPU cache to satisfy a token budget.

        Parameters
        ----------
        u : int
            Batch index.
        num_remove : int, optional
            Number of tokens to remove from cache. If None, computed from overflow versus
            self.max_total_retrieved_tokens.
        ignore_blocks : Iterable[int] or None
            Block indices that must not be evicted (e.g., currently selected top-k).

        Side Effects
        ------------
        - May offload evicted blocks (GPU → CPU/disk) via block.offload().
        - Updates self.cached_blocks[u] (LRU timestamps map).
        """
        if num_remove is None:
            tokens_in_cache = sum([self.global_blocks[u][bidx].size for bidx in self.cached_blocks[u].keys()])
            num_remove = tokens_in_cache - self.max_total_retrieved_tokens
        if num_remove <= 0:
            return

        # Sort blocks by last-access timestamp (ascending => LRU first)
        lst = list(self.cached_blocks[u].items())
        lst.sort(key=lambda x: x[1])

        removed = 0
        for idx, _ts in lst:
            if ignore_blocks is None or (idx not in ignore_blocks):
                # Offload the block’s GPU content; metadata stays
                self.global_blocks[u][idx].offload()
                self.cached_blocks[u].pop(idx)
                removed += self.global_blocks[u][idx].size
            if removed >= num_remove:
                return


    def _remove_cpu_lru_blocks(self, ignore_blocks=None):
        """
        Ensure there are enough free CPU cache units by offloading least-used blocks to disk.

        Parameters
        ----------
        ignore_blocks : Iterable[int] or None
            Block indices that must not be offloaded to disk.

        Notes
        -----
        - Triggers only if the CPU cache is nearly full based on exc + retrieved budget.
        - Uses self.block_usage[u] timestamps to find cold blocks whose cpu_data exists.
        """
        num_remove = ((self.exc_block_size + self.max_total_retrieved_tokens) // self.min_block_size) + 1 - len(
            self.cpu_cache.idle_set
        )
        if num_remove > 0:
            for u in range(self.batch_size):
                lst = list(self.block_usage[u].items())
                lst.sort(key=lambda x: x[1])  # LRU on CPU
                removed = 0
                for idx, _ts in lst:
                    if (ignore_blocks is None or idx not in ignore_blocks) and self.global_blocks[u][idx].cpu_data is not None:
                        self.global_blocks[u][idx].offload_to_disk()  # CPU → disk
                        removed += 1
                        if removed >= num_remove:
                            break


    def _calc_topk_blocks(self, len_q, global_q):
        """
        Select the next set of global memory blocks to retrieve given the current query.

        Strategy
        --------
        - Compute effective global context capacity (reserves room for remainder/init and optional contiguity buffer).
        - If everything fits, return all blocks.
        - Otherwise, rank blocks by similarity between per-block representations and the flattened global query.
        Optionally compute head-aware Q–μK scores (hook provided) for hybrid ranking.
        - Accumulate blocks until capacity is filled (greedy, respects block sizes).

        Parameters
        ----------
        len_q : int
            Current local query length.
        global_q : torch.Tensor
            Global query (B, H, T_local, Dh) before flattening/mean.

        Returns
        -------
        List[List[int]]
            For each batch u, an ordered list of block indices to retrieve.
        """
        global_remainder_len = max(self._global_remainder_ed - self._global_remainder_st + len_q - self.n_local, 0)
        global_context_cap = self.global_context_cap - global_remainder_len - (self.length > self.n_local) * self.init_k.size(-2)

        # Reserve capacity for optional contiguity buffer
        if self.use_contiguity_buffer:
            if self.contiguity_buffer_size < 1:
                global_context_cap -= int(self.contiguity_buffer_size * global_context_cap + 1)
            else:
                global_context_cap -= self.contiguity_buffer_size

        # If all blocks fit, just take them all
        if self.global_length[0] <= global_context_cap:
            return [list(range(len(self.global_blocks[0]))) for _ in range(self.batch_size)]

        # Pool heads → (B, H*Dh) for retrieval scoring
        global_q = global_q.mean(dim=2, keepdim=False)
        assert global_q.shape == (self.batch_size, self.num_heads, self.dim_head)
        global_q = global_q.reshape(self.batch_size, self.dim_head * self.num_heads)

        retrieved_blocks = []
        for u in range(self.batch_size):
            # Optional Q–μK scoring hook (currently not fused into final score)
            use_qk = self.qk_retrieval and (self._live_q_heads is not None) and (len(self.block_muK[u]) > 0)
            if use_qk:
                qn = self._live_q_heads  # (H, Dh)
                mu_list = [m for m in self.block_muK[u] if m is not None]
                if len(mu_list) == len(self.block_muK[u]) and len(mu_list) > 0:
                    muK = torch.stack(mu_list, dim=0)  # (B, H_kv, Dh)
                    H_kv = muK.size(1)
                    # Align μK heads to Q heads
                    if self.num_heads % H_kv == 0:
                        repeat = self.num_heads // H_kv
                        muK = muK.unsqueeze(2).expand(-1, H_kv, repeat, -1).reshape(muK.size(0), self.num_heads, self.dim_head)
                    elif H_kv != self.num_heads:
                        muK = torch.nn.functional.pad(muK, (0, 0, 0, max(0, self.num_heads - H_kv)))[:, : self.num_heads, :]
                    muK = torch.nn.functional.normalize(muK, dim=-1)
                    sims = torch.einsum("bhd,hd->bh", muK, qn)  # (B, H)
                    if 0 < self.qk_top_h < sims.size(1):
                        block_scores_qk = torch.topk(sims, k=self.qk_top_h, dim=1).values.mean(dim=1)  # (B,)
                    else:
                        block_scores_qk = sims.max(dim=1).values  # (B,)
                    # Combine here with repr similarity if desired using self.qk_weight.

            # Sort by learned representation similarity or random (for debugging)
            if self.random_topk_blocks:
                sorted_block_idx = list(range(self.num_global_block))
                random.shuffle(sorted_block_idx)
            else:
                sorted_block_idx = self.block_repr_k[u].sort_by_similarity(global_q[u])

            sorted_block_idx = iter(sorted_block_idx)

            # Greedily pick blocks until capacity reached, honoring block sizes
            context_len = 0
            filled_global_context = False
            batch_retrieved_blocks = []

            while not filled_global_context:
                b_idx = next(sorted_block_idx, None)
                cur_blocks = [b_idx]
                if b_idx is None:
                    filled_global_context = True
                    break
                for cur in cur_blocks:
                    if cur in batch_retrieved_blocks or cur < 0 or cur > self.num_global_block - 1:
                        continue
                    batch_retrieved_blocks.append(cur)
                    prev_context_len = context_len
                    context_len += self.global_blocks[u][cur].size
                    if context_len >= global_context_cap:
                        # If we overshoot, keep the closer fit
                        if abs(global_context_cap - prev_context_len) <= abs(context_len - global_context_cap):
                            batch_retrieved_blocks.pop()
                            context_len -= self.global_blocks[u][cur].size
                        filled_global_context = True
                        break

            retrieved_blocks.append(batch_retrieved_blocks)

        return retrieved_blocks


    def _update_contiguity_buffer(self, len_q, topk_blocks):
        """
        Build/refresh a contiguity buffer with neighbors of the selected top-k blocks.

        Purpose
        -------
        - Encourages temporal/local continuity by prefetching adjacent blocks (+1, -1).
        - Maintains a separate capacity budget for this buffer to avoid crowding out top-k.

        Parameters
        ----------
        len_q : int
            Current local query length.
        topk_blocks : List[List[int]]
            Top-k blocks per batch from _calc_topk_blocks.

        Side Effects
        ------------
        - Updates self.contiguity_buffer[u] in place with a list of adjacent block indices.
        """
        global_remainder_len = max(self._global_remainder_ed - self._global_remainder_st + len_q - self.n_local, 0)
        topk_global_context_cap = self.global_context_cap - global_remainder_len - (self.length > self.n_local) * self.init_k.size(-2)

        # Compute contiguity capacity
        if self.contiguity_buffer_size < 1:
            ctg_global_context_cap = int(self.contiguity_buffer_size * topk_global_context_cap + 1)
        else:
            ctg_global_context_cap = self.contiguity_buffer_size

        for u in range(self.batch_size):
            if len(topk_blocks[u]) == 0:
                continue
            batch_ctg_blocks = []
            context_len = 0
            filled_global_context = False
            batch_topk_blocks = iter(topk_blocks[u])

            # Consider immediate neighbors around each chosen block
            while not filled_global_context:
                bidx = next(batch_topk_blocks, None)
                if bidx is None:
                    break
                cur_blocks = [bidx + 1, bidx - 1]
                for cur in cur_blocks:
                    if cur in batch_ctg_blocks or cur in topk_blocks[u] or cur < 0 or cur > self.num_global_block - 1:
                        continue
                    batch_ctg_blocks.append(cur)
                    prev_context_len = context_len
                    context_len += self.global_blocks[u][cur].size
                    if context_len >= ctg_global_context_cap:
                        if abs(ctg_global_context_cap - prev_context_len) <= abs(context_len - ctg_global_context_cap):
                            batch_ctg_blocks.pop()
                            context_len -= self.global_blocks[u][cur].size
                        filled_global_context = True
                        break

            # Integrate with past buffer content (bounded by capacity)
            batch_ctg_blocks.reverse()
            if len(batch_ctg_blocks) > 0:
                if len(self.contiguity_buffer[u]) == 0 or filled_global_context:
                    self.contiguity_buffer[u] = batch_ctg_blocks
                else:
                    total_buffer_len = sum([self.global_blocks[u][b].size for b in self.contiguity_buffer[u]])
                    if total_buffer_len + context_len < ctg_global_context_cap:
                        self.contiguity_buffer[u] += batch_ctg_blocks
                    else:
                        past_blocks = self.contiguity_buffer[u]
                        past_blocks.reverse()
                        self.contiguity_buffer[u] = batch_ctg_blocks
                        for cur in past_blocks:
                            if cur in batch_ctg_blocks or cur in topk_blocks[u] or cur < 0 or cur > self.num_global_block - 1:
                                continue
                            self.contiguity_buffer[u].insert(0, cur)
                            prev_context_len = context_len
                            context_len += self.global_blocks[u][cur].size
                            if context_len >= ctg_global_context_cap:
                                if abs(ctg_global_context_cap - prev_context_len) <= abs(context_len - ctg_global_context_cap):
                                    self.contiguity_buffer[u].pop(0)
                                    context_len -= self.global_blocks[u][cur].size
                                break


    def _get_init_and_remainder_context(self, init_st, global_h_k, global_h_v, global_remainder_len):
        """
        Populate the global K/V buffers with:
        - Initial context (init_k/init_v) if we've passed n_local,
        - The current global remainder window,
        and return a trimmed view plus the sliding window definition.

        Parameters
        ----------
        init_st : int
            Start index where init K/V should be written in the global buffer.
        global_h_k, global_h_v : torch.Tensor
            Preallocated global K/V buffers (B, H_kv, T_max, Dh).
        global_remainder_len : int
            Number of remainder tokens to append after init.

        Returns
        -------
        global_h_k, global_h_v : torch.Tensor
            Sliced views that include init + remainder segments only.
        sliding_window : Tuple[int, int]
            (absolute_end_of_remainder, n_local) used by attention to mask local vs global.
        """
        init_len = self.init_k.size(-2)
        init_ed = init_st + init_len

        # Write init context if we’ve progressed beyond local-only phase
        if self.length > self.n_local:
            global_h_k[:, :, init_st:init_ed, :].copy_(self.init_k, non_blocking=True)
            global_h_v[:, :, init_st:init_ed, :].copy_(self.init_v, non_blocking=True)

        ed = init_ed
        rmd_st = init_ed
        rmd_ed = rmd_st + global_remainder_len
        ed = rmd_ed

        # Append the global remainder segment
        global_h_k[:, :, rmd_st:rmd_ed, :].copy_(
            self.global_remainder[0][:, :, self._global_remainder_st : self._global_remainder_st + global_remainder_len, :],
            non_blocking=True,
        )
        global_h_v[:, :, rmd_st:rmd_ed, :].copy_(
            self.global_remainder[1][:, :, self._global_remainder_st : self._global_remainder_st + global_remainder_len, :],
            non_blocking=True,
        )

        sliding_window = (self.global_remainder[0].size(-2) + rmd_st, self.n_local)

        # Trim to the filled region
        global_h_k = global_h_k[:, :, :ed, :]
        global_h_v = global_h_v[:, :, :ed, :]
        return global_h_k, global_h_v, sliding_window


    def _get_global_hidden_and_mask(self, len_q, topk_blocks):
        """
        Load selected global blocks (by index) into the global K/V buffers and
        append init + remainder. Returns the buffers, sliding window, and init start.

        Parameters
        ----------
        len_q : int
            Current local query length.
        topk_blocks : List[List[int]]
            For each batch, the ordered block indices to load.

        Returns
        -------
        global_h_k, global_h_v : torch.Tensor
            Populated global K/V tensors of shape (B, H_kv, T_global_loaded, Dh).
        sliding_window : Tuple[int, int]
            (absolute_end_of_remainder, n_local) for masking.
        init_st : int
            The index where init context started (used by caller for layout bookkeeping).
        """
        assert len(topk_blocks) == self.batch_size
        global_remainder_len = max(self._global_remainder_ed - self._global_remainder_st + len_q - self.n_local, 0)

        global_h_k = self.global_buffer[0]
        global_h_v = self.global_buffer[1]
        num_retrieved_blocks = len(topk_blocks[0])

        size = 0
        for u in range(self.batch_size):
            assert len(topk_blocks[u]) == num_retrieved_blocks
            # Do not sort topk_blocks - preserve order of relevance 
            st = 0
            ed = 0
            for b_idx in topk_blocks[u]:
                assert b_idx in self.cached_blocks[u]
                ed = st + self.global_blocks[u][b_idx].size
                # Load from cache (GPU or CPU/disk with staging) into global_h_*
                self.global_blocks[u][b_idx].load((global_h_k[u, :, st:ed, :], global_h_v[u, :, st:ed, :]))
                size += self.global_blocks[u][b_idx].size
                st = ed

        init_st = st
        global_h_k, global_h_v, sliding_window = self._get_init_and_remainder_context(
            init_st, global_h_k, global_h_v, global_remainder_len
        )
        return global_h_k, global_h_v, sliding_window, init_st


    def _retrieve_and_attend(self, local_q, local_k, local_v, global_q):
        """
        Core attention pass that:
        1) Pos-embeds local Q/K and appends them to the local attention buffer,
        2) Computes which global blocks to fetch,
        3) Loads those blocks (plus init + remainder) into global K/V buffers,
        4) Runs attention over [local | global] with complement sliding window masking.

        Parameters
        ----------
        local_q, local_k, local_v : torch.Tensor
            Local Q/K/V of shape (B, H or H_kv, T_local, Dh).
        global_q : torch.Tensor
            Global Q for retrieval scoring (same shape as local_q).

        Returns
        -------
        torch.Tensor
            Attention output of shape (B, H, T_local, Dh).
        """
        # Rotary/position embeddings for local segment
        local_h_q, local_h_k = self.position_embedding(local_q, local_k)
        local_h_v = local_v
        if self.use_hf_acc:
            local_h_q = local_h_q.to(local_q.device)
            local_h_k = local_h_k.to(local_k.device)

        # Start attention accumulator with local window
        attn = self.Attn(local_h_q.shape, local_h_q.dtype, local_h_q.device)
        attn.append(local_h_q, local_h_k, local_h_v, get_score=True, sliding_window=self.n_local)

        # In parallel stream: determine and stage global context
        with torch.cuda.stream(GLOBAL_STREAM):
            # 1. Get top-k relevant blocks
            topk_blocks = self._calc_topk_blocks(local_h_q.size(-2), global_q)

            # 2. Optionally prepend contiguity neighbors
            if self.use_contiguity_buffer:
                self._update_contiguity_buffer(local_h_q.size(-2), topk_blocks)
                for u in range(self.batch_size):
                    if len(self.contiguity_buffer[u]) > 0:
                        topk_blocks[u] = self.contiguity_buffer[u] + topk_blocks[u]


            # --- FORCE all block representation vectors to GPU for cross-event reasoning ---
            # if self.cross_event_reasoner is not None:
            #     device = global_q.device
            #     for u in range(self.batch_size):
            #         # Move every block repr to GPU if not already
            #         for b_idx in topk_blocks[u]:
            #             repr_vec = self.block_repr_k[u].data[b_idx]
            #             if repr_vec.device != device:
            #                 self.block_repr_k[u].data[b_idx] = repr_vec.to(device, non_blocking=True)
    

            # 3. Apply cross-event reasoning to reweight and reorder blocks
            # The forward() method is called IMPLICITLY via PyTorch's __call__ mechanism
            # When you do: self.cross_event_reasoner(inputs), PyTorch automatically calls forward(inputs)
            if self.cross_event_reasoner is not None and len(topk_blocks[0]) > 0:
                for u in range(self.batch_size):
                    if len(topk_blocks[u]) == 0:
                        continue

                    block_indices = topk_blocks[u]

                    # Use stored representation vectors: each block is one "event"
                    block_reps = []
                    for b_idx in block_indices:
                        repr_vec = self.block_repr_k[u].data[b_idx]   # [E] per block, E = num_heads * dim_head
                        block_reps.append(repr_vec)

                    if len(block_reps) > 0:
                        # [N, 1, E] where N = num_blocks, R = 1 token per event
                        event_reps = torch.stack(block_reps, dim=0).unsqueeze(1)

                        # ---- FIX: build query_rep with the SAME E = H * Dh ----
                        # global_q[u]: [H, T, Dh] -> mean over time (T) -> [H, Dh] -> flatten -> [H*Dh]
                        q_heads = global_q[u].mean(dim=1)   # [H, Dh]
                        query_rep = q_heads.reshape(-1)     # [H*Dh] = 4096, matches emb_dim

                        # --- Move tiny tensors to CPU for CrossEventReasoner ---
                        event_reps_cpu = event_reps.to(self.cross_event_reasoner_device, non_blocking=False)
                        query_rep_cpu = query_rep.to(self.cross_event_reasoner_device, non_blocking=False)

                        # CPU forward pass
                        weights_cpu, scores_cpu, _ = self.cross_event_reasoner(
                            event_reps_cpu,
                            query_rep_cpu
                        )

                        # Move results back to GPU
                        weights = weights_cpu.to(global_q.device)
                        scores = scores_cpu.to(global_q.device)

                        # Debug: Log cross-event reasoning (only first few times)
                        if self.layer_idx == 0 and self.load_count < 3:
                            print(f"[Cross-Event] Layer {self.layer_idx}, Load {self.load_count}:")
                            print(f"  Blocks before: {block_indices[:5]}")
                            print(f"  Weights: {weights.cpu().tolist()[:5]}")

                        # Reorder blocks by cross-event importance
                        _, indices = torch.sort(weights, descending=True)
                        topk_blocks[u] = [block_indices[i] for i in indices.cpu().tolist()]

                        if self.layer_idx == 0 and self.load_count < 3:
                            print(f"  Blocks after: {topk_blocks[u][:5]}")
                            print("  Reordered successfully!")

            
            # LRU maintenance: mark access, evict to meet token budget, then optionally nudge CPU→disk
            self.load_count += 1
            for u in range(self.batch_size):
                tokens_in_cache = sum([self.global_blocks[u][bidx].size for bidx in self.cached_blocks[u].keys()])
                num_remove = tokens_in_cache - self.max_total_retrieved_tokens
                for b_idx in topk_blocks[u]:
                    if b_idx not in self.cached_blocks[u]:
                        num_remove += self.global_blocks[u][b_idx].size

                self._remove_lru_blocks(u, num_remove, topk_blocks[u])
                if self.allow_disk_offload is True:
                    self._remove_cpu_lru_blocks(set(list(self.cached_blocks[u].keys())))

                # Touch timestamps for blocks we’re going to use
                for bidx in topk_blocks[u]:
                    self.cached_blocks[u][bidx] = self.load_count
                    if self.allow_disk_offload is not False:
                        self.block_usage[u][bidx] = self.load_count

            global_h_q = global_q
            global_h_k, global_h_v, global_sliding_window, init_st = self._get_global_hidden_and_mask(
                local_h_q.size(-2), topk_blocks
            )

        # Ensure the current stream waits for global staging
        if self.async_global_stream:
            torch.cuda.current_stream().wait_stream(GLOBAL_STREAM)

        # Append global segment with complement sliding window (prevents double-count overlap)
        attn.append(
            global_h_q,
            global_h_k,
            global_h_v,
            end=True,
            get_score=False,
            sliding_window=global_sliding_window,
            complement_sliding_window=True,
        )

        # Finalize attention and capture representation scores for memory update
        attn_output, repr_score = attn.get_result()
        self.exc_repr_score = repr_score[0]

        if self.async_global_stream:
            GLOBAL_STREAM.wait_stream(torch.cuda.current_stream())

        self.attn = None
        return attn_output.view((self.batch_size, self.num_heads, -1, self.dim_head))


    def _from_group_kv(self, tensor):
        """
        Expand grouped KV heads to match the number of Q heads.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor shaped (H_kv, T, Dh) or (H, T, Dh). If already H==num_heads, returns as is.

        Returns
        -------
        torch.Tensor
            Tensor with shape (H, T, Dh) where H == self.num_heads.
        """
        assert tensor.dim() == 3
        if tensor.size(0) == self.num_heads:
            return tensor
        _, length, dim_head = tensor.shape
        num_group = self.num_heads // self.num_heads_kv
        tensor = tensor.view((self.num_heads_kv, 1, length, dim_head))
        tensor = tensor.expand((self.num_heads_kv, num_group, length, dim_head)).reshape((self.num_heads, length, dim_head))
        return tensor


    def get_block_k(self, k, repr_score):
        """
        Select the top-K key positions per head based on representation scores.

        Parameters
        ----------
        k : torch.Tensor
            Keys with shape (..., T, Dh); at least 2D with time on the penultimate axis.
        repr_score : torch.Tensor
            Representation scores with shape (..., T) matching k without Dh.

        Returns
        -------
        (torch.Tensor, int)
            - Gathered K entries of shape (H, repr_topk, Dh) after head expansion.
            - The integer repr_topk actually used (min(self.repr_topk, T)).
        """
        assert isinstance(repr_score, torch.Tensor)
        assert k.dim() >= 2
        k = self._from_group_kv(k)
        assert k.shape[:-1] == repr_score.shape

        repr_topk = min(self.repr_topk, repr_score.shape[-1])
        score_topk = repr_score.topk(repr_topk, dim=-1).indices
        assert score_topk.shape == (self.num_heads, repr_topk)

        gathered = torch.gather(k, -2, score_topk[:, :, None].expand(self.num_heads, repr_topk, self.dim_head))
        return gathered, repr_topk


    def _add_block(self, u, remainder_st, remainder_ed, load_to_disk=False):
        """
        Create a new memory block from the global remainder window and register its metadata.

        Parameters
        ----------
        u : int
            Batch index.
        remainder_st, remainder_ed : int
            Start/end indices into the global remainder to form this block.
        load_to_disk : bool
            If True, directly offload block payload to disk instead of CPU cache.

        Side Effects
        ------------
        - Appends a new MemoryBlock to self.global_blocks[u].
        - Updates μK signatures (block_muK) and block representation vectors (block_repr_k).
        - Increments self.num_global_block and self.global_length[u].
        """
        kv = (
            self.global_remainder[0][u, :, remainder_st:remainder_ed, :],
            self.global_remainder[1][u, :, remainder_st:remainder_ed, :],
        )

        # Per-block head-mean key signature (normalized), used for optional Q–μK retrieval
        try:
            k_slice = self.global_remainder[0][u, :, remainder_st:remainder_ed, :]  # (H_kv, span, Dh)
            mu_k = k_slice.mean(dim=1)  # (H_kv, Dh)
            mu_k = torch.nn.functional.normalize(mu_k, dim=-1)
            self.block_muK[u].append(mu_k)
        except Exception:
            self.block_muK[u].append(None)

        block = self.create_memory_block(
            kv=kv,
            cache=self.cuda_cache,
            load_to_cache=False,
            allow_disk_offload=self.allow_disk_offload,
            load_to_disk=load_to_disk,
            cpu_cache=self.cpu_cache,
        )
        self.global_blocks[u].append(block)

        if self.allow_disk_offload is not False:
            bidx = len(self.global_blocks[u]) - 1
            self.block_usage[u][bidx] = 0  # initialize “last used” timestamp

        # Compute block-level retrieval vector from top-k per-head positions
        global_block_repr_k, repr_topk = self.get_block_k(
            self.global_remainder[0][u, :, remainder_st:remainder_ed, :],
            self.global_remainder_repr_score[u, :, remainder_st:remainder_ed],
        )
        assert global_block_repr_k.shape == (self.num_heads, repr_topk, self.dim_head)
        global_block_repr_k = global_block_repr_k.mean(dim=-2, keepdim=False)
        # Always keep block representation vectors on GPU for cross-event reasoning
        global_block_repr_k = global_block_repr_k.reshape(self.num_heads * self.dim_head)[None, :]

        # Force GPU residency (avoid CPU→GPU cost during retrieval)
        if self.cross_event_reasoner is not None:
            global_block_repr_k = global_block_repr_k.to(self.global_remainder[0].device, non_blocking=True)

        self.block_repr_k[u].append(global_block_repr_k)


        self.num_global_block += 1
        self.global_length[u] += remainder_ed - remainder_st


    def append(self, local_q, local_k, local_v, global_q, global_k, global_v):
        """
        Append a new local segment and corresponding global remainder, then run retrieval+attention.

        Steps
        -----
        1) Initialize state on first call.
        2) Ensure CPU cache/vector offload if enabled.
        3) Extend the rolling local KV caches with the new local segment.
        4) Extend the global remainder (K, V, Q, and score buffers).
        5) Call _retrieve_and_attend to run the attention pass and return output.

        Parameters
        ----------
        local_q, local_k, local_v : torch.Tensor
            New local segment Q/K/V with shape (B, H or H_kv, T_local, Dh).
        global_q, global_k, global_v : torch.Tensor
            Global segment Q/K/V aligned to the same T_local (after rotary shift inside).

        Returns
        -------
        torch.Tensor
            Attention output of shape (B, H, T_local, Dh).
        """
        batch_size = local_q.size(0)
        input_length = local_q.size(-2)
        assert input_length <= self.exc_block_size
        assert batch_size == 1

        # If per-head mode, expand KV heads to match Q heads
        if self.perhead:
            num_heads = local_q.size(1)
            num_heads_kv = local_v.size(1)

            def repeat_kv(t):
                t = t.view(batch_size, num_heads_kv, 1, input_length, -1)
                t = t.expand(batch_size, num_heads_kv, num_heads // num_heads_kv, input_length, -1)
                t = t.reshape(batch_size * num_heads, 1, input_length, -1)
                return t

            local_q = local_q.view(batch_size * num_heads, 1, input_length, -1)
            local_k = repeat_kv(local_k)
            local_v = repeat_kv(local_v)
            global_q = global_q.view(batch_size * num_heads, 1, input_length, -1)
            global_k = repeat_kv(global_k)
            global_v = repeat_kv(global_v)

        # Lazy init of buffers and bookkeeping
        if not self.initialized:
            self._init(local_q, local_k, local_v, global_q, global_k, global_v)

        # Prepare CPU caches / vector offload when enabled
        if self.allow_disk_offload is True:
            if self.cpu_cache is None:
                self._init_cpu_cache(local_q, local_k)
            if self.vector_offload and self.block_repr_k[0].data.device != torch.device("cpu"):
                self._offload_vector()

        # Keep global stream in sync if async staging is used
        if self.async_global_stream:
            GLOBAL_STREAM.wait_stream(torch.cuda.current_stream())

        # 1) Extend rolling local KV caches
        self.local_k = torch.cat((self.local_k, local_k), dim=-2)
        self.local_v = torch.cat((self.local_v, local_v), dim=-2)
        self.kv_length = self.local_k.size(-2)

        # 2) Extend global remainder and associated score buffers (done on the GLOBAL stream)
        with torch.cuda.stream(GLOBAL_STREAM):
            # Apply rotary to the new global segment (offset by current n_local)
            global_q = self.position_embedding.apply_rotary_pos_emb_one_angle(global_q, self.n_local)

            self._global_remainder_st = 0
            self._global_remainder_ed = self.global_remainder[0].size(-2)

            self.global_remainder = (
                torch.cat((self.global_remainder[0], global_k), dim=-2),
                torch.cat((self.global_remainder[1], global_v), dim=-2),
                torch.cat((self.global_remainder[2], global_q), dim=-2),
            )

            # Extend placeholder tensors for representation scores and surprisal/divide flags
            self.global_remainder_repr_score = torch.cat(
                (
                    self.global_remainder_repr_score,
                    torch.zeros(
                        (self.batch_size, self.num_heads, global_k.size(-2)),
                        dtype=global_k.dtype,
                        device=global_k.device,
                    ),
                ),
                dim=-1,
            )

            self.global_remainder_surprisal = torch.cat(
                (
                    self.global_remainder_surprisal,
                    torch.zeros((self.batch_size, global_k.size(-2)), dtype=global_k.dtype, device=global_k.device),
                ),
                dim=-1,
            )

            self.global_block_divide = torch.cat(
                (
                    self.global_block_divide,
                    torch.zeros((self.batch_size, global_k.size(-2)), dtype=torch.bool, device=global_k.device),
                ),
                dim=-1,
            )

        # 3) Run retrieval + attention
        attn_output = self._retrieve_and_attend(local_q, self.local_k, self.local_v, global_q)

        if self.perhead:
            attn_output = attn_output.view(batch_size, self.num_heads, input_length, -1)

        return attn_output


    def update_memory(self, exc_length, exc_surprisal, surprisal_values=None):
        """
        After producing attention for a new local chunk, update the long-term memory.

        What happens
        ------------
        - Accumulate representation scores for the just-processed positions.
        - Update surprisal (or divide) indicators to decide where to cut blocks.
        - Grow the 'init' context up to n_init tokens.
        - Partition the global remainder into blocks by divide markers, add full or partial blocks.
        - Advance the global remainder window and prune local KV to n_local.

        Parameters
        ----------
        exc_length : int
            Number of new tokens just processed.
        exc_surprisal : torch.Tensor or torch.BoolTensor or None
            Either per-token surprisal values (B, T_exc) or boolean divide flags.
            If None, only representation scores are updated.
        surprisal_values : torch.Tensor or None
            Optional override for surprisal values (same shape as exc_surprisal float).

        Side Effects
        ------------
        - Updates global_remainder buffers, block boundaries (global_block_divide),
        block metadata (global_blocks, block_muK, block_repr_k), and local KV caches.
        """
        with torch.cuda.stream(GLOBAL_STREAM):
            global_remainder_ed = self._global_remainder_ed + exc_length
            global_remainder_st = self._global_remainder_st
            global_remainder_len = global_remainder_ed - global_remainder_st

            # 1) Accumulate representation scores over the processed region
            assert self.exc_repr_score.shape[:3] == (self.batch_size, self.num_heads, self.kv_length)
            self.exc_repr_score = self.exc_repr_score[:, :, -exc_length - self.n_local :]
            self.global_remainder_repr_score[:, :, global_remainder_ed - self.exc_repr_score.size(-1) : global_remainder_ed].add_(
                self.exc_repr_score
            )

            # 2) Update divide markers from surprisal or uniform sizing
            if exc_surprisal is not None:
                if self.use_hf_acc:
                    exc_surprisal = exc_surprisal.to(self.global_remainder_surprisal.device)

                if not self.uniform_blocks:
                    assert exc_surprisal.shape == (self.batch_size, exc_length)
                    if surprisal_values is None:
                        self.global_remainder_surprisal[:, global_remainder_ed - exc_length : global_remainder_ed].copy_(
                            exc_surprisal
                        )
                    else:
                        if self.use_hf_acc:
                            surprisal_values = surprisal_values.to(self.global_remainder_surprisal.device)
                        self.global_remainder_surprisal[:, global_remainder_ed - exc_length : global_remainder_ed].copy_(
                            surprisal_values
                        )

                    # Convert float surprisal → boolean divide threshold; if already bool, use directly
                    if exc_surprisal.dtype == torch.bool:
                        divide = exc_surprisal
                    else:
                        avg_st = max(global_remainder_ed - self.n_local, 0)
                        avg_ed = max(global_remainder_ed - exc_length, self.n_init)
                        divide = exc_surprisal > (
                            self.surprisal_threshold_gamma * torch.std(self.global_remainder_surprisal[:, avg_st:avg_ed], dim=-1)
                            + torch.mean(self.global_remainder_surprisal[:, avg_st:avg_ed], dim=-1)
                        )
                else:
                    # Uniform block partitioning mode (debug/ablation)
                    divide = torch.zeros(exc_surprisal.shape, dtype=torch.bool)
                    divide[:, :: self.max_block_size] = True

                self.global_block_divide[:, global_remainder_ed - exc_length : global_remainder_ed].copy_(divide)

            # 3) Grow the init context up to n_init using the earliest part of the remainder
            if not self.init_exc and global_remainder_len > self.n_local:
                global_k = self.global_remainder[0]
                global_v = self.global_remainder[1]
                append_init_len = min(self.n_init - self.init_k.size(-2), global_remainder_len - self.n_local)
                self.init_k = torch.cat(
                    (self.init_k, global_k[:, :, global_remainder_st : global_remainder_st + append_init_len, :]),
                    dim=-2,
                )
                self.init_v = torch.cat(
                    (self.init_v, global_v[:, :, global_remainder_st : global_remainder_st + append_init_len, :]),
                    dim=-2,
                )
                global_remainder_st += append_init_len
                global_remainder_len -= append_init_len
                if self.init_k.size(-2) == self.n_init:
                    self.init_exc = True

            # 4) Partition the remainder into blocks and register them
            if global_remainder_len >= self.n_local + self.min_block_size:
                ed = global_remainder_len - self.n_local
                for u in range(self.batch_size):
                    divide = self.global_block_divide[u, global_remainder_st : global_remainder_st + ed]
                    surprising_token_idx = torch.where(divide > 0)[0]

                    # Fallback: if no divides, enforce a stride at max_block_size (or a final tail)
                    if surprising_token_idx.shape[-1] == 0:
                        if divide.shape[-1] > self.max_block_size:
                            surprising_token_idx = torch.tensor(
                                range(0, divide.shape[-1], self.max_block_size), dtype=torch.int16, device=divide.device
                            )
                        else:
                            surprising_token_idx = torch.tensor([divide.shape[-1]], dtype=torch.int16, device=divide.device)

                    # Make inclusive block edges [0, ..., len]
                    surprising_token_idx = torch.cat(
                        (
                            torch.tensor([0], device=surprising_token_idx.device),
                            surprising_token_idx,
                            torch.tensor([len(divide)], device=surprising_token_idx.device),
                        )
                    )

                    block_sizes = surprising_token_idx[1:] - surprising_token_idx[:-1]
                    mask = torch.where(block_sizes != 0)[0]
                    block_sizes = block_sizes[mask]

                    # If CPU cache nearly full, directly create blocks as disk-resident
                    load_to_disk = False
                    if self.allow_disk_offload is True and len(self.cpu_cache.idle_set) == 1:
                        print("=== WARNING! ===> ONLY ONE CPU CACHE UNIT LEFT! OFFLOADING DIRECTLY TO DISK")
                        load_to_disk = True

                    # Accumulate sub-blocks until reaching max_block_size; flush and continue
                    acc_b = 0
                    for i, b in enumerate(block_sizes):
                        acc_b += int(b.item())
                        while acc_b >= self.max_block_size:
                            self._add_block(u, global_remainder_st, global_remainder_st + self.max_block_size, load_to_disk=load_to_disk)
                            global_remainder_st += self.max_block_size
                            acc_b -= self.max_block_size

                        # Avoid creating tiny sub-blocks below min_block_size (except at the very end)
                        if acc_b < min(self.min_block_size, divide.shape[-1]):
                            continue

                        # Flush mid-run partial block if not the last one
                        if acc_b > 0 and i != len(block_sizes) - 1:
                            self._add_block(u, global_remainder_st, global_remainder_st + acc_b, load_to_disk=load_to_disk)
                            global_remainder_st += acc_b
                        acc_b = 0

            # 5) Commit new remainder bounds
            self._global_remainder_ed = global_remainder_ed
            self._global_remainder_st = global_remainder_st

        # Synchronize: ensure memory updates finish before next compute on current stream
        if self.async_global_stream:
            torch.cuda.current_stream().wait_stream(GLOBAL_STREAM)

        # Advance the global length counter
        self.length += exc_length

        # 6) Keep only the last n_local tokens in local KV (rolling window)
        if self.local_k.size(-2) >= self.n_local:
            self.local_k = self.local_k[:, :, -self.n_local :, :]
            self.local_v = self.local_v[:, :, -self.n_local :, :]

        # Sanity: the edited index should match the new remainder tail
        assert self._global_remainder_ed == self.global_remainder[0].size(-2)

        # Trim remainder buffers to drop what we’ve turned into blocks
        with torch.cuda.stream(GLOBAL_STREAM):
            self.global_remainder = (
                self.global_remainder[0][:, :, self._global_remainder_st :, :],
                self.global_remainder[1][:, :, self._global_remainder_st :, :],
                self.global_remainder[2][:, :, self._global_remainder_st :, :],
            )
            self.global_remainder_surprisal = self.global_remainder_surprisal[:, self._global_remainder_st :]
            self.global_block_divide = self.global_block_divide[:, self._global_remainder_st :]
            self.global_remainder_repr_score = self.global_remainder_repr_score[:, :, self._global_remainder_st :]
