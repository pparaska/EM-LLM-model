"""
EM-LLM forward patches for attention and causal language modeling.

This module provides:
- em_llm_attn_forward: a factory that returns a patched attention forward
  function compatible with HF-style attention blocks. It wires a ContextManager
  that segments the stream into local+global memory blocks, performs optional
  surprisal-based boundary detection, and can refine boundaries using a
  graph-theoretic similarity objective.
- em_llm_causal_lm_forward: a replacement for a model's generate-time / train-time
  forward that computes logits and (optionally) a cross-entropy loss, derives
  surprisal or consumes a boolean boundary mask, performs optional similarity
  refinement of event boundaries, and updates the ContextManager memories.

Key ideas:
- Local window + global episodic blocks with optional contiguity buffer.
- Surprisal-driven event boundaries, optionally refined via modularity /
  conductance / intra–inter similarity on a token–token similarity matrix.
- Optional Q–μK retrieval (head-aware relevance between current Q and block summaries).
"""

import torch
from torch.nn import CrossEntropyLoss, MSELoss  # MSELoss imported but unused
from typing import List, Optional, Tuple, Union

from .context_manager import ContextManager
from transformers.modeling_outputs import CausalLMOutputWithPast
from .similarity_refinement import events_with_similarity_adjustment


def em_llm_attn_forward(
    model,
    n_local,
    n_init,
    max_block_size,
    max_cached_block,
    exc_block_size,
    repr_topk: int = 1,
    surprisal_threshold_gamma: float = 1.1,
    async_global_stream=True,
    pin_memory=False,
    perhead=False,
    n_mem: int = 2048,
    min_block_size: int = 1,
    block_similarity_topk: int = False,
    similarity_refinement_kwargs: dict = {},
    contiguity_buffer_kwargs: dict = {},
    random_topk_blocks=False,
    infini_attention=False,
    uniform_blocks=False,
    *args, **kwargs
):
    """
    Factory that returns a patched attention forward function integrating EM-LLM memory.

    This wraps a Hugging Face-style attention block so that each call:
      1) Projects Q/K/V (shared or separate projection supported).
      2) Hands them to a ContextManager that maintains local window + global
         memory blocks (episodic cache).
      3) Performs attention over local and (optionally) retrieved global blocks.
      4) Optionally uses surprisal-driven event boundaries and similarity-based
         refinement (see em_llm_causal_lm_forward for how surprisal is produced).

    Parameters
    ----------
    model : nn.Module
        Parent model or module providing config/context (not directly used here,
        but useful for compatibility).
    n_local : int
        Size of the sliding local attention window.
    n_init : int
        Minimum prefix kept before episodic segmentation begins.
    max_block_size : int
        Maximum token span for a single global memory block.
    max_cached_block : int
        Cap on how many global blocks to cache.
    exc_block_size : int
        Execution chunk size (number of new tokens processed per append).
    repr_topk : int, default 1
        Number of representative blocks to retrieve per step (if retrieval used).
    surprisal_threshold_gamma : float, default 1.1
        Multiplier for std when thresholding surprisal spikes to mark boundaries.
    async_global_stream : bool, default True
        If True, stream global K/V asynchronously to overlap compute.
    pin_memory : bool, default False
        Pin host memory for faster H2D transfers (when applicable).
    perhead : bool, default False
        If True, maintain per-head memories/logic in ContextManager.
    n_mem : int, default 2048
        Total memory budget (tokens) for global blocks.
    min_block_size : int, default 1
        Minimum allowed size for a block after refinement.
    block_similarity_topk : int or bool, default False
        If set to int, restrict refinement scoring to top-k candidate blocks.
    similarity_refinement_kwargs : dict, default {}
        Extra args forwarded to similarity refinement (e.g., similarity_metric).
    contiguity_buffer_kwargs : dict, default {}
        Settings for contiguity buffer (reserving tokens between events).
    random_topk_blocks : bool, default False
        If True, sample retrieval candidates randomly (for ablations).
    infini_attention : bool, default False
        If True, enable compatibility mode with Infini-attention style caches.
    uniform_blocks : bool, default False
        If True, disables surprisal-based adaptive boundaries (uniform splits).
    *args, **kwargs
        Additional options forwarded to ContextManager (e.g., qk_retrieval, qk_weight, qk_top_h).

    Returns
    -------
    callable
        A closure `forward(self, query, key_value, position_bias, use_cache, past_key_value, ...)`
        that replaces the attention forward and returns:
            (context_output, None, past_key_value)

    Notes
    -----
    - The returned forward assumes use_cache=True (KV caching enabled).
    - Q–μK retrieval can be toggled via kwargs: qk_retrieval, qk_weight, qk_top_h.
    """

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        position_bias: Optional[torch.Tensor],
        use_cache: bool,
        past_key_value,
        project_q,
        project_k,
        project_v,
        attention_out,
        dim_head,
        num_heads,
        num_heads_kv,
    ):
        """
        Patched attention forward that routes Q/K/V through ContextManager.

        Parameters
        ----------
        query : Tensor, shape (B, T_q, D_in)
            Input hidden states for queries.
        key_value : Tensor, shape (B, T_kv, D_in)
            Input hidden states for keys/values.
        position_bias : Tensor or None
            Optional positional bias/embedding for the layer.
        use_cache : bool
            Must be True; enables KV caching and episodic memory.
        past_key_value : Any or None
            ContextManager instance for this layer; constructed on first call.
        project_q, project_k, project_v : callable or nn.Module
            Linear projections for Q/K/V (or fused qkv if project_k is None).
        attention_out : callable or nn.Module
            Output projection applied to the concatenated head outputs.
        dim_head : int
            Per-head dimensionality.
        num_heads : int
            Number of attention heads for Q.
        num_heads_kv : int
            Number of attention heads for K/V (may be <= num_heads).

        Returns
        -------
        o : Tensor, shape (B, T_q, D_out)
            Attention output after local + global memory attention and output proj.
        None
            Placeholder to match HF attention signature (past_attn_probs, unused).
        past_key_value : ContextManager
            Updated ContextManager carrying caches, blocks, and state.

        Side Effects
        ------------
        - On first call, constructs a ContextManager with the configured memory
          and refinement options.
        - Updates internal episodic/global caches via ContextManager.append().
        """
        batch_size = query.size(0)
        len_q = query.size(1)
        len_k = key_value.size(1)

        assert use_cache
        if project_k is not None:
            h_q = project_q(query)
            h_k = project_k(key_value)
            h_v = project_v(key_value)
        else:
            qkv = project_q(query)
            query_pos = num_heads * dim_head
            h_q = qkv[..., :query_pos]
            h_k = qkv[..., query_pos : query_pos + num_heads_kv * dim_head]
            h_v = qkv[..., query_pos + num_heads_kv * dim_head :]

        h_q = h_q.view(batch_size, len_q, num_heads, dim_head).permute(0, 2, 1, 3).contiguous()
        q_heads_mean = h_q.mean(dim=2).squeeze(0)
        h_k = h_k.view(batch_size, len_k, num_heads_kv, dim_head).permute(0, 2, 1, 3).contiguous()
        h_v = h_v.view(batch_size, len_k, num_heads_kv, dim_head).permute(0, 2, 1, 3).contiguous()

        if past_key_value is None:
            past_key_value = ContextManager(
                layer_idx=self.layer_idx,
                position_embedding=position_bias,
                n_init=n_init,
                n_local=n_local,
                max_block_size=max_block_size,
                max_cached_block=max_cached_block,
                exc_block_size=exc_block_size,
                min_block_size=min_block_size,
                async_global_stream=async_global_stream,
                pin_memory=pin_memory,
                perhead=perhead,
                repr_topk=repr_topk,
                surprisal_threshold_gamma=surprisal_threshold_gamma,
                n_mem=n_mem,
                block_similarity_topk=block_similarity_topk,
                uniform_blocks=uniform_blocks,
                random_topk_blocks=random_topk_blocks,
                infini_attention=infini_attention,
                **similarity_refinement_kwargs,
                **contiguity_buffer_kwargs,
                qk_retrieval=kwargs.pop("qk_retrieval", True),
                qk_weight=float(kwargs.pop("qk_weight", 1.0)),
                qk_top_h=int(kwargs.pop("qk_top_h", 0)),
                **kwargs,
            )
            try:
                past_key_value.set_live_q_heads(q_heads_mean)
            except Exception:
                pass

        local_q, local_k, local_v = h_q, h_k, h_v
        global_q, global_k, global_v = h_q, h_k, h_v

        o = past_key_value.append(
            local_q, local_k, local_v,
            global_q, global_k, global_v,
        )

        o = o.view(batch_size, num_heads, len_q, dim_head).permute(0, 2, 1, 3)
        o = o.reshape(batch_size, len_q, dim_head * num_heads)
        o = attention_out(o)

        return o, None, past_key_value

    return forward


def em_llm_causal_lm_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    em_labels: Optional[torch.Tensor] = None,
    **kwargs,
) -> Union[Tuple, CausalLMOutputWithPast]:
    """
    Causal LM forward with EM-LLM surprisal and optional similarity refinement.

    This drop-in forward computes logits (and optional cross-entropy loss), derives
    surprisal per token (or accepts a boolean boundary mask), optionally refines
    event boundaries using a graph-based similarity objective, and updates the
    per-layer ContextManager memories.

    Parameters
    ----------
    self : PreTrainedModel
        HF-style causal LM (must expose .model (backbone) and .lm_head).
    input_ids : LongTensor, shape (B, T), optional
        Token ids. Mutually exclusive with inputs_embeds.
    attention_mask : Tensor, optional
        Standard attention mask.
    position_ids : LongTensor, optional
        Positional ids when model requires explicit positions.
    past_key_values : list[Any], optional
        Per-layer ContextManager instances (created on first pass if None).
    inputs_embeds : FloatTensor, shape (B, T, D), optional
        Embedded inputs as an alternative to input_ids.
    labels : LongTensor, shape (B, T), optional
        If provided, compute cross-entropy loss against logits.
    use_cache : bool, optional
        Enable KV caching; should be True for EM-LLM memory behavior.
    output_attentions : bool, optional
        Pass-through to backbone.
    output_hidden_states : bool, optional
        Pass-through to backbone.
    return_dict : bool, optional
        If True, return CausalLMOutputWithPast.
    em_labels : Tensor, optional
        If dtype != bool: token ids used to compute surprisal = -log p(token).
        If dtype == bool: treated as a precomputed boundary mask (True at cuts).

    kwargs : dict
        Unused here; accepted for compatibility.

    Returns
    -------
    CausalLMOutputWithPast
        loss : Tensor or None
            Cross-entropy loss if labels provided.
        logits : FloatTensor, shape (B, T, V)
            Unnormalized token logits.
        past_key_values : list[ContextManager]
            Updated per-layer episodic memory state.
        hidden_states, attentions : optional
            Pass-through from backbone if requested.

    Boundary Detection and Refinement
    ---------------------------------
    1) If em_labels is token ids (non-bool), the function computes per-token
       surprisal and detects boundary positions by thresholding against a
       running mean/std (gamma multiplier).
    2) If similarity_refinement is enabled on the ContextManager, the function:
       - stacks K across layers for the recent window,
       - builds a token–token similarity matrix via dot products,
       - refines each tentative boundary within a local window by optimizing a
         graph criterion (modularity / conductance / intra–inter sim),
       - replaces the raw thresholded mask with the refined mask.
    3) The refined boolean boundary mask is passed to each layer’s
       ContextManager.update_memory() to finalize blocks.

    Notes
    -----
    - When uniform_blocks is True, surprisal segmentation is skipped (uniform splits).
    - Surprisal refinement uses only a suffix of the stream (global remainder).
    - All tensor device transfers are best-effort; falls back gracefully if needed.
    """
    """
    Args:
        labels (torch.LongTensor, optional): shape (batch_size, seq_len)
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]
    past_key_values = outputs[1]

    logits = self.lm_head(hidden_states)
    if past_key_values and past_key_values[0].use_hf_acc:
        logits = logits.to(torch.cuda.current_device())
    logits = logits.float()

    loss = None
    if labels is not None:
        if labels.shape[-1] != logits.shape[-2]:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
        else:
            shift_logits = logits
            shift_labels = labels
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1).to(shift_logits.device)
        loss = CrossEntropyLoss()(shift_logits, shift_labels)

    # Build surprisal or pass-through boolean mask
    if em_labels is not None and em_labels.dtype != torch.bool:
        prob = torch.softmax(logits, dim=-1)
        surprisal = -torch.log(torch.gather(prob, dim=-1, index=em_labels.unsqueeze(-1))).squeeze(-1)
    elif em_labels is not None and em_labels.dtype == torch.bool:
        surprisal = em_labels
    else:
        surprisal = None

    torch.cuda.synchronize()

    surprisal_values = None
    if (
        surprisal is not None
        and (em_labels is None or em_labels.dtype != torch.bool)
        and not past_key_values[0].uniform_blocks
    ):
        assert surprisal.shape == (past_key_values[0].batch_size, input_ids.shape[-1]), \
            f"Problem with surprisal shape: {surprisal.shape}"

        if past_key_values[0].similarity_refinement:
            exc_length = input_ids.shape[-1]
            global_remainder_ed = past_key_values[0]._global_remainder_ed + exc_length
            global_remainder_len = global_remainder_ed - past_key_values[0]._global_remainder_st

            if global_remainder_ed <= exc_length:
                divide = surprisal > (
                    past_key_values[0].surprisal_threshold_gamma * torch.std(surprisal, dim=-1)
                    + torch.mean(surprisal, dim=-1)
                )
            else:
                avg_st = max(global_remainder_ed - past_key_values[0].n_local, 0)
                avg_ed = max(global_remainder_ed - exc_length, past_key_values[0].n_init)
                divide = surprisal > (
                    past_key_values[0].surprisal_threshold_gamma
                    * torch.std(past_key_values[0].global_remainder_surprisal[:, avg_st:avg_ed], dim=-1)
                    + torch.mean(past_key_values[0].global_remainder_surprisal[:, avg_st:avg_ed], dim=-1)
                )

            if global_remainder_len >= 2 * exc_length and past_key_values[0].refine_with_buffer:
                last_divide = torch.zeros(past_key_values[0].batch_size)
                for u in range(past_key_values[0].batch_size):
                    last_events = torch.where(
                        past_key_values[0].global_block_divide[u, global_remainder_ed - 2 * exc_length : global_remainder_ed - exc_length] > 0
                    )[0]
                    last_divide[u] = exc_length if len(last_events) == 0 else last_events[-1]
                offsets = exc_length - last_divide.int()
                max_offset = torch.max(offsets)
            else:
                offsets = torch.zeros(past_key_values[0].batch_size).int()
                max_offset = 0

            st_layer = past_key_values[0].refine_from_layer
            assert len(past_key_values) >= st_layer + 1, \
                f"Problem with refine_from_layer param: {st_layer + 1} < {len(past_key_values)}"

            K = torch.clone(
                past_key_values[st_layer].global_remainder[0][:, :, global_remainder_ed - exc_length - max_offset : global_remainder_ed, :]
            ).unsqueeze(dim=1).to(torch.cuda.current_device())

            for l in range(st_layer + 1, len(past_key_values)):
                K = torch.cat(
                    (
                        K,
                        torch.clone(
                            past_key_values[l].global_remainder[0][:, :, global_remainder_ed - exc_length - max_offset : global_remainder_ed, :]
                        ).unsqueeze(dim=1).to(torch.cuda.current_device()),
                    ),
                    dim=1,
                )

            stacked_A = torch.einsum("blhtd,blhTd->btT", K, K).detach()
            del K

            try:
                stacked_A = stacked_A.to(past_key_values[0].global_remainder[0].device)
            except Exception as e:
                print(f"Tried casting stacked_A to GPU, but failed with error: {e}")

            for u in range(past_key_values[0].batch_size):
                events_sur = torch.where(divide[u] > 0)[0]
                if len(events_sur) > 0:
                    events_sur_mod = events_with_similarity_adjustment(
                        events_sur,
                        stacked_A[u][max_offset - offsets[u] :, :][:, max_offset - offsets[u] :],
                        similarity_metric=past_key_values[0].similarity_metric,
                        min_size=past_key_values[0].min_block_size,
                        offset=offsets[u],
                    )
                    divide[u] = torch.zeros_like(divide[u])
                    divide[u][events_sur_mod] = True

            del stacked_A
            surprisal_values = torch.clone(surprisal)
            surprisal = divide
            assert surprisal.dtype == torch.bool, \
                f"Problem with surprisal dtype after refinement: {surprisal.dtype}"

    for pkv in past_key_values:
        pkv.update_memory(input_ids.shape[-1], surprisal, surprisal_values=surprisal_values)

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
