import numpy as np
import torch
import gc
import time


class GreedySearch:
    """
    GreedySearch
    -------------
    A minimal greedy decoding loop tailored for EM-LLM-style models that support
    cached key/value states (past_key_values) and optional episodic-memory labels
    (em_labels) for segmentation/surprisal-based event marking during the
    forward pass.

    Responsibilities
    - Prepare model inputs from raw text via a tokenizer.
    - Run chunked forward passes over a long input context to (optionally)
      compute perplexity, update the model cache, and enable offloading knobs.
    - Continue token-by-token greedy generation until an end token is reached or
      max_length is hit.
    - Maintain and reuse past_kv across generate() calls (opt-in caching).

    Notes
    - This class assumes a Hugging Face-like model interface that accepts
      input_ids, attention_mask, past_key_values, labels, em_labels, and returns
      logits, loss, and new past_key_values.
    - em_labels can be provided externally (e.g., sentence boundaries), computed
      via surprisal ("surprisal"), or randomly sampled ("random") for testing.
    - compute_ppl toggles whether loss/perplexity is accumulated per chunk.
    """

    def __init__(self, model, tokenizer, model_type, em_splitter=None
                , compute_ppl=False):
        """
        Initialize the greedy search helper.

        Args:
            model: A language model supporting HF-like generate forward calls.
            tokenizer: Paired tokenizer with encode/decode.
            model_type: String tag to branch model-specific logic ("em-llm").
            em_splitter: One of {None, "surprisal", "random", "sentence"} that
                         controls how episodic-memory labels are built for each
                         processed chunk during the long-context pass.
            compute_ppl: If True, accumulate loss and report chunk/total PPL.
        """
        model.eval()
        self.model = model
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.past_kv = None  # cache persisted across calls to generate()
        self.compute_ppl = compute_ppl
        self.em_splitter = em_splitter

    def clear(self):
        """
        Clear any cached past key/values and trigger Python GC.
        Useful if you want to free GPU/CPU memory between runs.
        """
        self.past_kv = None
        gc.collect()

    def _process_texts(self, input_text):
        """
        Tokenize raw text and build a minimal model_inputs dict compatible with
        the underlying model. Moves tensors to CUDA and adds batch dimension.

        Args:
            input_text: String to be tokenized.

        Returns:
            dict with CUDA tensors: {"input_ids": (1, T), "attention_mask": (1, T)}
        """
        model_inputs = {}
        input_ids = self.tokenizer.encode(input_text)

        model_inputs["input_ids"] = input_ids
        model_inputs["attention_mask"] = [1] * len(model_inputs["input_ids"])

        # to tensor -> int -> add batch dim -> move to GPU
        for key in model_inputs:
            model_inputs[key] = torch.tensor(model_inputs[key]).int().unsqueeze(0).cuda()

        return model_inputs

    def generate(self, text=None, input_ids=None, em_labels=None, **kwargs):
        """
        Public entrypoint for generation. Accepts either raw text or prebuilt
        input_ids. Calls the internal decoding loop under inference mode.

        Args:
            text: Optional raw text prompt. Used when input_ids is None.
            input_ids: Optional pre-tokenized ids of shape (T,) or (1, T).
            em_labels: Optional boolean mask used for episodic marking during
                       the long-context pass (same shape as input_ids). If not
                       provided, the class can synthesize it depending on
                       self.em_splitter.
            **kwargs: Extra configuration passed to _decode (e.g., chunk_size,
                     offload thresholds, max_length, etc.).

        Returns:
            dict with {"pred": decoded text after the original prompt,
                       "chunk_ppl": list[float or None],
                       "total_ppl": float or None}
        """
        if input_ids is None:
            model_inputs = self._process_texts(text)
            input_ids = model_inputs['input_ids']

        with torch.inference_mode():
            result = self._decode(input_ids, em_labels, **kwargs)

        return result

    def _random_splitter(self, ids):
        """
        Create a random boolean mask over input ids to simulate event boundaries.
        Ensures the first position is marked and avoids the last-token boundary.

        Args:
            ids: Tensor of shape (B, T) or (1, T) used only for length.

        Returns:
            torch.BoolTensor of the same shape with randomly set split markers.
        """
        where2split = torch.zeros_like(ids)

        for i, elem_ids in enumerate(ids):
            # choose 10 random split indices, excluding the last token
            splits = np.random.randint(low=0, high=len(elem_ids)-1, size=10)
            where2split[i][0] = 1
            where2split[i][splits] = 1
        return where2split.bool()
    
    def _model_pass(self, input_ids, attention_mask, past_key_values, em_labels=None, labels=None):
        """
        Single forward pass wrapper for the underlying model.

        - If compute_ppl is False, labels are cleared so loss/PPL aren't computed.
        - For model_type == "em-llm", forwards all the EM-LLM-specific fields
          including em_labels and returns a HF-style output object.

        Args:
            input_ids: (B, T_chunk) input token ids for this step/chunk.
            attention_mask: (B, T_total_so_far) attention mask.
            past_key_values: cache from prior steps (can be None on first call).
            em_labels: Optional boolean mask indicating episode boundaries for
                       the chunk. Shape must match input_ids when provided.
            labels: Teacher-forcing labels aligned to input_ids for loss/PPL.

        Returns:
            HF-like output object with fields: logits, loss (optional),
            past_key_values, (optionally) attentions, etc.
        """
        if not self.compute_ppl:
            labels = None

        if self.model_type == "em-llm":
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
                past_key_values=past_key_values,
                em_labels=em_labels,
                labels=labels,
                output_attentions=True,
            )
        else:
            raise NotImplementedError

        return out

    def _decode(self, input_ids, em_labels=None, max_length=100, extra_end_token_ids=[]
                , chunk_size: int = 4096, output=False, **kwargs):
        """
        Core decoding loop.

        Phase A (i == 0):
            - Optionally process the entire prompt in chunks (chunk_size) to
              compute PPL (if enabled), populate/update past_kv and apply
              offloading policies. This acts like a long-context warmup pass.
        Phase B (i > 0):
            - Greedy-generate one token at a time using the updated cache.

        Args:
            input_ids: (1, T) or (T,) prompt token ids. Will be expanded to (1,T).
            em_labels: Optional (1, T) boolean mask aligned to input_ids for the
                       long-context pass (used if em_splitter == "sentence").
            max_length: Max number of new tokens to decode after the prompt.
            extra_end_token_ids: Additional token ids to treat as EOS.
            chunk_size: Chunk length for the warmup pass over the prompt.
            output: If True, stream decoded text to stdout during generation.
            **kwargs: May include disk_offload_threshold, vector_offload_threshold.

        Returns:
            dict with decoded continuation and PPL metrics.
        """
        if input_ids.dim() == 1:
            input_ids = input_ids[None, :]
        if (em_labels is not None) and (em_labels.dim() == 1):
            em_labels = em_labels[None, :]
        input_ids = input_ids.cuda()
        print(f"Context Length: {input_ids.size()}")
        attention_mask = torch.ones_like(input_ids)
        assert input_ids.size(0) == 1  # single-batch path
        length = input_ids.size(1)     # prompt length

        # eos set includes tokenizer.eos_token_id plus any caller-provided ids
        end_token_ids = extra_end_token_ids + [self.tokenizer.eos_token_id]

        logits = None
        past_key_values = self.past_kv  # start from cached kv if available
        if output:
            output_text = ""

        total_loss = 0
        chunk_ppl = []

        # i == 0 handles the initial prompt in chunks; then token-by-token
        for i in range(max_length + 1):
            if i == 0:
                # set default chunk_size to prompt length if None
                if chunk_size is None:
                    chunk_size = input_ids.size(1)
                avg_time = 0

                # iterate over prompt in [st:ed) windows (teacher-forced)
                for st in range(0, input_ids.size(1) - 1, chunk_size):
                    start_time = time.time()
                    ed = min(input_ids.size(1) - 1, st + chunk_size)

                    # Build episodic mask for this chunk according to splitter.
                    if self.em_splitter == "surprisal":
                        # shift by 1 to align labels with next-token prediction
                        em_input = input_ids[:, st+1: ed+1]
                    elif self.em_splitter == "random":
                        em_input = self._random_splitter(input_ids[:, st: ed])
                        assert em_input.dtype == torch.bool
                    elif self.em_splitter == "sentence":
                        em_input = em_labels[:, st: ed]
                        assert em_input.dtype == torch.bool
                    else:
                        em_input = None

                    if em_input is not None:
                        # must match input window shape
                        assert em_input.shape == input_ids[:, st: ed].shape, f"Shape mismatch in em_labels and input_ids"
                    
                    # Apply optional offloading policies on existing cache
                    if past_key_values is not None: 
                        if past_key_values[0].allow_disk_offload is None and input_ids.size(1) > kwargs["disk_offload_threshold"]:
                            print(f"Inputs have length {input_ids.size(1)}: allowing disk offload for past_key_values.")
                            for pkv in past_key_values:
                                pkv.allow_disk_offload = True
                        elif past_key_values[0].vector_offload and input_ids.size(1) > kwargs["vector_offload_threshold"] and past_key_values[0].block_repr_k[0].data.device != torch.device('cpu'):
                            for pkv in past_key_values:
                                pkv._offload_vector()

                    # Forward pass for this prompt chunk (teacher forcing)
                    out = self._model_pass(
                        input_ids=input_ids[:, st: ed],
                        attention_mask=attention_mask[:, :ed],
                        past_key_values=past_key_values,
                        labels=input_ids[:, st+1: ed+1],
                        em_labels=em_input,
                    )

                    logits, past_key_values = out.logits, out.past_key_values
                    
                    # Loss/PPL accumulation (if enabled)
                    try:
                        loss = out.loss.detach().cpu() if out.loss is not None else None
                    except:
                        loss = None
                    if loss is not None:
                        ppl = torch.exp(loss).item() if self.compute_ppl else None
                        total_loss += loss * (ed - st)
                    else:
                        ppl = None
                    
                    chunk_ppl.append(ppl)

                    # Logging and periodic memory/avg-time reporting
                    time_taken = round(time.time() - start_time, 2)
                    avg_time += time_taken
                    log = f"Chunk: {int(st / chunk_size + 1)}/{(input_ids.size(1))//chunk_size}, ppl: {ppl}, time: {time_taken}s"
                    print(log)
                    if int(st / chunk_size + 1) % 100 == 0:
                        print(torch.cuda.memory_summary())
                        print(f"Average time taken per chunk: {round(avg_time/int(st / chunk_size + 1), 2)}s")
                        gc.collect()
 
                # Final one-token step to align the cache with the full prompt
                out = self._model_pass(
                    input_ids = input_ids[:, -1:],
                    attention_mask = attention_mask,
                    past_key_values = past_key_values,
                )
                logits, past_key_values = out.logits, out.past_key_values
            else:
                # Incremental decode: feed the last produced token
                out = self._model_pass(
                    input_ids = input_ids[:, -1:],
                    attention_mask = attention_mask,
                    past_key_values = past_key_values,
                )
                logits, past_key_values = out.logits, out.past_key_values

            # Greedy pick next token
            logits = logits[:, -1, :]
            word = logits.argmax(dim=-1)
           
            # Stop if EOS or max_length reached
            if word.item() in end_token_ids or i == max_length:
                break

            # Append token to the running sequence and extend mask
            input_ids = torch.cat((input_ids, word.view(1, 1)), dim=-1)
            attention_mask = torch.cat(
                (attention_mask, torch.ones((attention_mask.size(0), 1), dtype=torch.int, device=attention_mask.device)),
                dim=-1
            )
            if output:
                # Optional streaming of only the new text beyond the prompt
                tmp = self.tokenizer.decode(input_ids.squeeze(0)[length:])
                if len(tmp) > len(output_text):
                    import sys               
                    sys.stdout.write(tmp[len(output_text):])
                    sys.stdout.flush()
                    output_text = tmp

        # persist cache for potential reuse in next generate() call
        self.past_kv = past_key_values

        if output:
            sys.stdout.write("\n")
            sys.stdout.flush()

        # Finalize perplexity metrics
        if self.compute_ppl:
            chunk_ppl = chunk_ppl
            total_ppl = torch.exp(total_loss).item()
        else:
            chunk_ppl = None
            total_ppl = None          

        return {"pred": self.tokenizer.decode(input_ids.squeeze(0)[length:]), "chunk_ppl": chunk_ppl, "total_ppl": total_ppl}
