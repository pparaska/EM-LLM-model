import math
from typing import List, Optional, Tuple, Union

import torch
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from ..attention import RotaryEmbeddingESM, ATTN_FORWARD, CAUSAL_LM_FORWARD


def huggingface_forward(forward):
    def hf_forward(
        self,
        hidden_states: torch.Tensor,
        position_ids=None,
        past_key_value=None,
        use_cache: bool = False,
        **kwargs,
    ):
        if hasattr(self, "q_proj"):
            q_proj = self.q_proj
            k_proj = self.k_proj
            v_proj = self.v_proj
        elif hasattr(self, "qkv_proj"):
            q_proj = self.qkv_proj
            k_proj = None
            v_proj = None
        else:
            raise NotImplementedError(
                f"The attention module {self.__class__.__name__} does not appear to have the required projection methods."
            )

        hidden_states, loss, pkv = forward(
            self,
            hidden_states,
            hidden_states,
            position_ids,
            use_cache,
            past_key_value,
            q_proj,
            k_proj,
            v_proj,
            self.o_proj,
            self.head_dim,
            self.num_heads,
            self.num_key_value_heads,
        )

        return hidden_states, loss, pkv

    return hf_forward


def model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    **kwargs,
):
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
    elif input_ids is None and inputs_embeds is None:
        raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
        if hasattr(self, "config") and hasattr(self.config, "scale_emb"):
            inputs_embeds = inputs_embeds * self.config.scale_emb

    pkv = tuple() if use_cache else None
    hidden_states = inputs_embeds

    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    for i, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=self.position_bias,
            past_key_value=past_key_values[i] if past_key_values is not None else None,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        hidden_states = layer_outputs[0]

        if use_cache:
            _cache = layer_outputs[2 if output_attentions else 1]
            pkv = pkv + (_cache,)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    if not return_dict:
        return tuple(v for v in [hidden_states, pkv, all_hidden_states, all_self_attns] if v is not None)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=pkv,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def causal_lm_forward(
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
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, optional):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (ignored).
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
    logits = self.lm_head(hidden_states).float()

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

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def patch_hf(
    model,
    attn_type: str = "em_llm",
    attn_kwargs: dict = None,
    base=None,
    distance_scale=None,
    **kwargs,
):
    attn_kwargs = dict(attn_kwargs or {})
    attn_kwargs.update(kwargs)

    # This approach lacks scalability and will be refactored.
    from transformers import LlamaForCausalLM, MistralForCausalLM, Qwen2ForCausalLM, Phi3ForCausalLM

    forward = huggingface_forward(ATTN_FORWARD[attn_type](model=model, **attn_kwargs))

    if isinstance(model, (LlamaForCausalLM, MistralForCausalLM, Qwen2ForCausalLM, Phi3ForCausalLM)) \
       or model.__class__.__name__ in {"Phi3ForCausalLM", "MiniCPMForCausalLM"}:
        Attention = model.model.layers[0].self_attn.__class__
        Model = model.model.__class__
    else:
        raise ValueError(f"Only supports llama, mistral, phi3, and qwen2 models. {model.__class__.__name__} was passed.")

    hf_rope = model.model.layers[0].self_attn.rotary_emb
    if not hasattr(hf_rope, "dim"):
        # be robust to rope_kwargs["base"] being a float or an object with rope_theta
        base_candidate = None
        if hasattr(hf_rope, "rope_kwargs"):
            base_candidate = hf_rope.rope_kwargs.get("base", None)
        if hasattr(base_candidate, "rope_theta"):
            hf_rope.base = base_candidate.rope_theta
        else:
            hf_rope.base = base_candidate

        if hf_rope.config is not None:
            # infer dim if missing
            if not hasattr(hf_rope, "dim") or hf_rope.dim is None:
                rope_theta = getattr(hf_rope.config, "rope_theta", None)
                if rope_theta is not None and hf_rope.base is None:
                    hf_rope.base = rope_theta
                partial_rotary_factor = getattr(hf_rope.config, "partial_rotary_factor", 1.0)
                head_dim = getattr(hf_rope.config, "head_dim",
                                   hf_rope.config.hidden_size // hf_rope.config.num_attention_heads)
                hf_rope.dim = int(head_dim * partial_rotary_factor)
        if not hasattr(hf_rope, "dim") or hf_rope.dim is None:
            raise NotImplementedError("Could not determine RoPE dim from HF module.")

    base = base if base is not None else hf_rope.base
    distance_scale = 1.0 if distance_scale is None else distance_scale

    if hasattr(hf_rope, "short_factor"):
        new_max_pos_emb = attn_kwargs.get("n_local", 0) + attn_kwargs.get("exc_block_size", 0)
        scale = new_max_pos_emb / getattr(hf_rope, "original_max_position_embeddings", max(new_max_pos_emb, 1))
        if scale <= 1.0:
            ext_factors = torch.tensor(hf_rope.short_factor)
        else:
            print(f"Extending context past original window with scale factor: {scale}")
            ext_factors = torch.tensor(hf_rope.long_factor)
            omp = float(getattr(hf_rope, "original_max_position_embeddings", 1))
            distance_scale = math.sqrt(1 + math.log(max(scale, 1.0000001)) / math.log(max(omp, 2.0)))
    else:
        ext_factors = torch.tensor(1.0)

    rope = RotaryEmbeddingESM(hf_rope.dim, base, distance_scale, ext_factors)
    model.model.position_bias = rope

    def set_forward(m):
        if isinstance(m, Attention):
            m._old_forward = m.forward
            m.forward = forward.__get__(m, Attention)

    model.apply(set_forward)

    model.model._old_forward = model.model.forward
    model.model.forward = model_forward.__get__(model.model, Model)

    model._old_forward = model.model.forward
    if attn_type in CAUSAL_LM_FORWARD:
        model.forward = CAUSAL_LM_FORWARD[attn_type].__get__(model, model.__class__)
    else:
        model.forward = causal_lm_forward.__get__(model, model.__class__)

    # ---- Read-only per-layer head caches ----
    if not hasattr(model, "_last_q_heads"):
        model._last_q_heads = {}  # layer_idx -> (H, Dh)

    def _get_q_heads(layers: list):
        return {L: model._last_q_heads[L] for L in layers if L in model._last_q_heads}

    model._get_q_heads = _get_q_heads

    return model
