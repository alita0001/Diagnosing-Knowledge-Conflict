import copy
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import torch.nn.functional as F

from probe.value_head_probe import ValueHeadProbe


class HallucinationMitigator:
    """
    Hallucination mitigation wrapper that supports:
      - Qwen2.5-VL style inputs (pixel_values + image_grid_thw)
      - Llama-3.2-Vision (mllama) style inputs (pixel_values + aspect_ratio_ids + aspect_ratio_mask + cross_attention_mask)

    Key fix vs your original version:
      1) Never “hand-pick” multimodal keys for mllama. Always forward all processor keys.
      2) For custom KV-cache decoding, extend mllama's cross_attention_mask along seq_len.
      3) For baseline/no-mitigation generation, pass full processor outputs so aspect_ratio_ids are included.
    """

    def __init__(
        self,
        model_name: str,
        probe_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = True,
        device_map: Optional[str] = None,  # e.g. "auto"
    ):
        self.device = device
        self.model_name = model_name
        self.probe_path = probe_path
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        self.device_map = device_map

        self.model = None
        self.processor = None
        self.probe = None

        self._is_mllama = False

        self.load_models()

    # -----------------------------
    # Loading
    # -----------------------------
    def load_models(self):
        """Loads the base multimodal model and the ValueHeadProbe."""
        print(f"Loading model: {self.model_name}")
        from transformers import AutoModelForImageTextToText, AutoProcessor

        print(f"Loading processor: {self.model_name}")
        self.processor = AutoProcessor.from_pretrained(self.model_name)

        print(f"Loading multimodal model: {self.model_name}")
        model_kwargs = dict(
            torch_dtype=self.torch_dtype,
            trust_remote_code=self.trust_remote_code,
        )
        if self.device_map is not None:
            model_kwargs["device_map"] = self.device_map

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            **model_kwargs,
        )
        self.model.eval()

        # If device_map is None, move model to device explicitly.
        if self.device_map is None:
            self.model.to(self.device)

        # Detect mllama (Llama-3.2-Vision) family
        mt = getattr(getattr(self.model, "config", None), "model_type", "") or ""
        cls_name = self.model.__class__.__name__
        self._is_mllama = (mt.lower() == "mllama") or ("mllama" in cls_name.lower())

        print(f"Loading probe from: {self.probe_path}")
        probe_path_obj = Path(self.probe_path)
        self.probe = ValueHeadProbe(self.model, path=probe_path_obj)
        self.probe.to(self.device)
        self.probe.eval()

        print(f"Models loaded successfully. is_mllama={self._is_mllama}")

    # -----------------------------
    # Tokenizer / decoding helpers
    # -----------------------------
    def _get_tokenizer(self):
        """Get an actual tokenizer-like object for decoding."""
        if hasattr(self.processor, "tokenizer") and self.processor.tokenizer is not None:
            return self.processor.tokenizer
        return self.processor

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        """Decode token ids to string. Accepts [T] or [B,T]."""
        tok = self._get_tokenizer()

        if isinstance(token_ids, torch.Tensor):
            if token_ids.dim() == 2:
                # Prefer batch_decode if available
                if hasattr(tok, "batch_decode"):
                    return tok.batch_decode(
                        token_ids,
                        skip_special_tokens=skip_special_tokens,
                        clean_up_tokenization_spaces=False,
                    )[0]
                token_ids = token_ids[0]
            token_ids = token_ids.tolist()

        # token_ids is now List[int]
        if hasattr(tok, "decode"):
            return tok.decode(token_ids, skip_special_tokens=skip_special_tokens)
        # last resort
        return str(token_ids)

    # -----------------------------
    # Input building utilities
    # -----------------------------
    def _to_device(self, inputs: Any) -> Any:
        """Move BatchFeature/dict(tensors) to self.device."""
        if hasattr(inputs, "to"):
            return inputs.to(self.device)

        if isinstance(inputs, dict):
            out = {}
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    out[k] = v.to(self.device)
                else:
                    out[k] = v
            return out

        return inputs

    def _build_inputs(self, prompt: str, image=None) -> Dict[str, torch.Tensor]:
        """
        Build model inputs using processor.
        For mllama, ensure <|image|> token exists if we fall back to plain processor(text=..., images=...).
        """
        processor = self.processor

        # Build chat messages
        if image is not None:
            msg = {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        else:
            msg = {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }

        # For mllama instruct examples: messages should be wrapped as a "conversation list"
        messages_for_template = [[msg]] if self._is_mllama else [msg]

        # Preferred path: processor.apply_chat_template(..., tokenize=True, return_dict=True, return_tensors="pt")
        if hasattr(processor, "apply_chat_template"):
            try:
                inputs = processor.apply_chat_template(
                    messages_for_template,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                return self._to_device(inputs)
            except TypeError:
                # Some processors don't support those args, fall back below
                pass
            except Exception:
                # Fall back below for any processor-specific incompatibility
                pass

        # Fallback: apply_chat_template to string then processor(text=..., images=...)
        text_prompt = None
        if hasattr(processor, "apply_chat_template"):
            try:
                text_prompt = processor.apply_chat_template(
                    messages_for_template,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                text_prompt = None

        if text_prompt is None:
            # Last fallback: raw prompt
            if self._is_mllama and (image is not None) and ("<|image|>" not in prompt):
                text_prompt = "<|image|>" + prompt
            else:
                text_prompt = prompt

        inputs = processor(
            text=[text_prompt] if not isinstance(text_prompt, list) else text_prompt,
            images=image,
            return_tensors="pt",
        )
        return self._to_device(inputs)

    def _split_inputs(
        self, inputs: Dict[str, torch.Tensor]
    ) -> (torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]):
        """Split into (input_ids, attention_mask, extra_mm_kwargs)."""
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        extra = {}
        for k, v in inputs.items():
            if k in ("input_ids", "attention_mask"):
                continue
            extra[k] = v
        return input_ids, attention_mask, extra

    def _extend_seq_mask(self, mask: Optional[torch.Tensor], new_len: int) -> Optional[torch.Tensor]:
        """
        Extend masks that have shape [B, seq_len, ...] to new_len by repeating last row.
        Works for cross_attention_mask (mllama) and any 2D/4D-like seq masks.
        """
        if mask is None or (not torch.is_tensor(mask)):
            return mask
        if mask.dim() < 2:
            return mask

        cur_len = int(mask.size(1))
        if cur_len == new_len:
            return mask
        if cur_len > new_len:
            return mask[:, :new_len, ...]

        pad_len = new_len - cur_len
        last = mask[:, -1:, ...].expand(mask.size(0), pad_len, *mask.shape[2:]).contiguous()
        return torch.cat([mask, last], dim=1)

    def _safe_model_forward(self, **kwargs):
        """
        Some models accept cache_position, others do not.
        Try with cache_position; if TypeError indicates unexpected kwarg, retry without it.
        """
        try:
            return self.model(**kwargs)
        except TypeError as e:
            msg = str(e)
            if "cache_position" in msg and "unexpected keyword" in msg:
                kwargs.pop("cache_position", None)
                return self.model(**kwargs)
            raise

    def _safe_probe_forward(self, inputs: Dict[str, torch.Tensor]):
        """Same retry logic for probe->model forward path."""
        try:
            return self.probe(inputs=inputs)
        except TypeError as e:
            msg = str(e)
            if "cache_position" in msg and "unexpected keyword" in msg:
                inputs = dict(inputs)
                inputs.pop("cache_position", None)
                return self.probe(inputs=inputs)
            raise

    # -----------------------------
    # Probe score
    # -----------------------------
    def predict_hallucination_score(self, model_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Returns hallucination probability for the LAST token in input_ids.
        Score = 1 - P(class0). Assumes class0=Faithful.
        """
        with torch.no_grad():
            out = self._safe_probe_forward(model_inputs)
            logits = out["probe_logits"]  # [B, T, C]
            last_token_logits = logits[:, -1, :]
            probs = torch.softmax(last_token_logits, dim=-1)
            return 1.0 - probs[:, 0]

    # -----------------------------
    # Public API
    # -----------------------------
    def generate(
        self,
        prompt: str,
        image=None,  # PIL image or similar
        method: str = "token_rejection",
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_k: int = 50,
        alpha: float = 0.5,
        num_candidates: int = 5,
        sentence_rejection_batch_size: int = 2,
    ) -> str:
        """
        Generates text using specified mitigation method.
        """
        inputs = self._build_inputs(prompt=prompt, image=image)
        input_ids, attention_mask, mm_kwargs = self._split_inputs(inputs)
        prompt_len = int(input_ids.size(1))

        if method == "token_rejection":
            return self._generate_token_rejection(
                input_ids=input_ids,
                attention_mask=attention_mask,
                mm_kwargs=mm_kwargs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                alpha=alpha,
                num_candidates=num_candidates,
            )

        if method == "sentence_rejection":
            return self._generate_sentence_rejection(
                input_ids=input_ids,
                attention_mask=attention_mask,
                mm_kwargs=mm_kwargs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                num_candidates=num_candidates,
                batch_size=sentence_rejection_batch_size,
            )

        # Fallback: standard generation (no mitigation)
        # Make temperature/top_k effective only when sampling.
        do_sample = (temperature is not None) and (float(temperature) > 0)
        gen_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=int(max_new_tokens),
            do_sample=bool(do_sample),
            use_cache=True,
            return_dict_in_generate=False,
        )
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask

        # forward all multimodal kwargs (IMPORTANT for mllama: aspect_ratio_ids...)
        gen_kwargs.update(mm_kwargs)

        if do_sample:
            gen_kwargs["temperature"] = float(temperature)
            if top_k is not None and int(top_k) > 0:
                gen_kwargs["top_k"] = int(top_k)

        with torch.inference_mode():
            out = self.model.generate(**gen_kwargs)

        generated = out[:, prompt_len:]
        return self.decode(generated)

    # -----------------------------
    # Token-level rejection sampling
    # -----------------------------
    def _generate_token_rejection(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        mm_kwargs: Dict[str, torch.Tensor],
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        alpha: float,
        num_candidates: int,
    ) -> str:
        """
        Token-Level Rejection Sampling.

        Alpha logic:
          - alpha <= 0 : baseline generation via model.generate (fast + correct for mllama keys)
          - alpha > 0  : custom KV-cache decoding, with per-candidate probe scoring
        """
        device = input_ids.device
        prompt_len = int(input_ids.size(1))

        # -------------------------
        # alpha <= 0: baseline
        # -------------------------
        if alpha is None or float(alpha) <= 0.0:
            gen_kwargs = dict(
                input_ids=input_ids,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,  # greedy baseline
                use_cache=True,
                return_dict_in_generate=False,
            )
            if attention_mask is not None:
                gen_kwargs["attention_mask"] = attention_mask
            gen_kwargs.update(mm_kwargs)

            with torch.inference_mode():
                out = self.model.generate(**gen_kwargs)

            return self.decode(out[:, prompt_len:])

        # -------------------------
        # alpha > 0: custom decoding
        # -------------------------
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device, dtype=torch.long)

        max_len = prompt_len + int(max_new_tokens)

        out_ids = input_ids.new_empty((1, max_len))
        out_ids[:, :prompt_len] = input_ids

        out_mask = attention_mask.new_empty((1, max_len))
        out_mask[:, :prompt_len] = attention_mask

        cur_len = prompt_len

        # mllama cross_attention_mask needs to grow with sequence length if provided
        cross_attention_mask_base = mm_kwargs.get("cross_attention_mask", None)

        # Prefill mm kwargs:
        #  - For mllama: keep pixel_values + aspect_ratio_* + cross_attention_mask
        #  - For Qwen2.5-VL: keep pixel_values + image_grid_thw
        mm_prefill = dict(mm_kwargs)

        # Step mm kwargs: avoid re-encoding vision if possible
        mm_step = dict(mm_kwargs)
        if "pixel_values" in mm_step:
            # For both families, we generally do NOT want to resend pixel_values every step.
            # For mllama, pixel_values also forces aspect_ratio_ids requirement, so drop them too.
            mm_step.pop("pixel_values", None)
            if self._is_mllama:
                mm_step.pop("aspect_ratio_ids", None)
                mm_step.pop("aspect_ratio_mask", None)

        # Prefill forward
        with torch.inference_mode():
            model_inputs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
            # cache_position is important for many modern models; safe-forward will drop if unsupported
            model_inputs["cache_position"] = torch.arange(prompt_len, device=device, dtype=torch.long)

            model_inputs.update(mm_prefill)

            outputs = self._safe_model_forward(**model_inputs)
            past = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :]

        eos_id = None
        tok = self._get_tokenizer()
        if hasattr(tok, "eos_token_id"):
            eos_id = tok.eos_token_id

        alpha = float(alpha)
        gen_history: List[int] = []
        recent_window = 64
        rep_penalty = 0.3

        for _ in range(int(max_new_tokens)):
            prefix_len = cur_len

            K = int(num_candidates) if int(num_candidates) > 0 else 1
            if top_k is not None and int(top_k) > 0:
                K = min(K, int(top_k))
            K = max(1, K)

            # temperature scaling
            scaled = next_logits if (temperature is None or float(temperature) <= 0) else (next_logits / float(temperature))
            topk_logits, topk_ids = torch.topk(scaled, k=K, dim=-1)  # [1,K]
            lm_logp = F.log_softmax(topk_logits, dim=-1).view(-1)     # [K]

            cand_tokens = topk_ids.view(K, 1)  # [K,1]

            # candidate scoring with probe (incremental, sequential)
            halluc_scores = self._evaluate_candidates_incremental(
                candidate_tokens=cand_tokens,
                past_key_values=past,
                attention_mask=out_mask[:, :cur_len],
                mm_step_kwargs=mm_step,
                cross_attention_mask_base=cross_attention_mask_base,
                num_candidates=K,
            )

            truth_scores = (1.0 - halluc_scores).clamp(min=1e-6, max=1.0)
            truth_log = torch.log(truth_scores)

            warm = min(1.0, (len(gen_history) + 1) / 16.0)
            alpha_eff = alpha * warm

            combined = lm_logp + alpha_eff * truth_log

            if gen_history:
                recent = set(gen_history[-recent_window:])
                for j in range(K):
                    tid = int(topk_ids[0, j].item())
                    if tid in recent:
                        combined[j] = combined[j] - rep_penalty

            best_j = int(torch.argmax(combined).item())
            best_id = topk_ids[0, best_j].view(1, 1)

            if eos_id is not None and int(best_id.item()) == int(eos_id):
                break

            out_ids[:, cur_len] = best_id
            out_mask[:, cur_len] = 1
            gen_history.append(int(best_id.item()))
            cur_len += 1

            # next step forward with KV-cache
            with torch.inference_mode():
                step_inputs = dict(
                    input_ids=best_id,
                    past_key_values=past,
                    attention_mask=out_mask[:, :cur_len],
                    use_cache=True,
                    return_dict=True,
                )
                step_inputs["cache_position"] = torch.tensor([prefix_len], device=device, dtype=torch.long)

                # If mllama cross_attention_mask exists, extend it to current length
                if cross_attention_mask_base is not None:
                    step_inputs["cross_attention_mask"] = self._extend_seq_mask(
                        cross_attention_mask_base, new_len=cur_len
                    )

                # add other step mm kwargs (image_grid_thw for qwen, etc.)
                # NOTE: do not overwrite cross_attention_mask we just extended
                for k, v in mm_step.items():
                    if k == "cross_attention_mask":
                        continue
                    step_inputs[k] = v

                step_out = self._safe_model_forward(**step_inputs)
                past = step_out.past_key_values
                next_logits = step_out.logits[:, -1, :]

        return self.decode(out_ids[:, prompt_len:cur_len])

    def _evaluate_candidates_incremental(
        self,
        candidate_tokens: torch.Tensor,
        past_key_values,
        attention_mask: torch.Tensor,
        mm_step_kwargs: Dict[str, torch.Tensor],
        cross_attention_mask_base: Optional[torch.Tensor],
        num_candidates: int = 5,
    ) -> torch.Tensor:
        """
        Score candidates by running the probe on a single next-token step using prefix KV-cache.
        Sequential evaluation to avoid batch-expanding Cache objects.

        Important for mllama:
          - If cross_attention_mask is present, it must match seq_len; we extend it by repeating last row.
          - Do NOT pass pixel_values at this stage (we already dropped it in mm_step_kwargs).
        """
        K = int(num_candidates) if int(num_candidates) > 0 else int(candidate_tokens.size(0))
        device = candidate_tokens.device

        if past_key_values is None:
            return torch.zeros((K,), device=device, dtype=torch.float32)

        prefix_len = int(attention_mask.size(1))

        scores = []
        for i in range(K):
            tok_i = candidate_tokens[i:i + 1]  # [1,1]

            # Defensive copy: cache updates are in-place
            try:
                pkv_i = past_key_values.copy() if hasattr(past_key_values, "copy") else copy.deepcopy(past_key_values)
            except Exception:
                pkv_i = copy.deepcopy(past_key_values)

            probe_inputs = dict(
                input_ids=tok_i,
                past_key_values=pkv_i,
                use_cache=True,
                return_dict=True,
            )

            # Extend attention mask to include new token
            am = torch.ones((1, prefix_len + 1), device=device, dtype=attention_mask.dtype)
            if prefix_len > 0:
                am[:, :prefix_len] = attention_mask[:, :prefix_len]
            probe_inputs["attention_mask"] = am

            # cache_position for new token
            probe_inputs["cache_position"] = torch.tensor([prefix_len], device=device, dtype=torch.long)

            # mllama cross_attention_mask (if present) must match seq_len=prefix_len+1
            if cross_attention_mask_base is not None:
                probe_inputs["cross_attention_mask"] = self._extend_seq_mask(
                    cross_attention_mask_base, new_len=prefix_len + 1
                )

            # other multimodal step kwargs (e.g., image_grid_thw)
            for k, v in mm_step_kwargs.items():
                if k == "cross_attention_mask":
                    continue
                probe_inputs[k] = v

            with torch.inference_mode():
                out = self._safe_probe_forward(probe_inputs)

            # Extract p_truth = P(class0)
            p_truth_t = None
            if isinstance(out, dict):
                if "probs" in out and torch.is_tensor(out["probs"]):
                    probs_t = out["probs"]
                    if probs_t.dim() == 3:
                        probs_t = probs_t[:, -1, :]
                    p_truth_t = probs_t[:, 0]
                elif "probe_logits" in out and torch.is_tensor(out["probe_logits"]):
                    lg = out["probe_logits"]
                    if lg.dim() == 3:
                        lg = lg[:, -1, :]
                    p_truth_t = torch.softmax(lg.float(), dim=-1)[:, 0]
                elif "logits" in out and torch.is_tensor(out["logits"]):
                    lg = out["logits"]
                    if lg.dim() == 3:
                        lg = lg[:, -1, :]
                    p_truth_t = torch.softmax(lg.float(), dim=-1)[:, 0]

            if p_truth_t is None:
                p_truth = torch.tensor(0.5, device=device, dtype=torch.float32)
            else:
                p_truth = p_truth_t[0].float().clamp(0, 1)

            scores.append((1.0 - p_truth).detach())

        return torch.stack(scores, dim=0).to(device=device)

    # -----------------------------
    # Sentence-level rejection sampling
    # -----------------------------
    def _repeat_inputs(self, base: Dict[str, torch.Tensor], n: int) -> Dict[str, torch.Tensor]:
        """Repeat a single-sample inputs dict along batch dim to size n."""
        out = {}
        for k, v in base.items():
            if not torch.is_tensor(v):
                out[k] = v
                continue
            if v.size(0) == 1:
                out[k] = v.repeat(n, *([1] * (v.dim() - 1)))
            else:
                # already batched; leave as-is
                out[k] = v
        return out

    def _generate_sentence_rejection(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        mm_kwargs: Dict[str, torch.Tensor],
        max_new_tokens: int,
        temperature: float,
        num_candidates: int,
        batch_size: int,
    ) -> str:
        """
        Sentence-Level Rejection Sampling.
        Generate N candidates, score each by average P(class0) over generated tokens, choose best.
        """
        device = input_ids.device
        prompt_len = int(input_ids.size(1))

        all_generated_ids: List[torch.Tensor] = []
        all_scores: List[float] = []

        base_inputs = dict(input_ids=input_ids)
        if attention_mask is not None:
            base_inputs["attention_mask"] = attention_mask
        base_inputs.update(mm_kwargs)

        num_batches = (int(num_candidates) + int(batch_size) - 1) // int(batch_size)

        for b in range(num_batches):
            curr_bs = min(int(batch_size), int(num_candidates) - b * int(batch_size))
            batch_inputs = self._repeat_inputs(base_inputs, curr_bs)

            # sampling generation
            gen_kwargs = dict(
                **batch_inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=True,
                temperature=float(temperature) if (temperature is not None and float(temperature) > 0) else 1.0,
                top_k=50,
                use_cache=True,
                return_dict_in_generate=False,
            )

            with torch.inference_mode():
                outputs = self.model.generate(**gen_kwargs)  # [B, T_out]

            # Build attention mask for outputs
            tok = self._get_tokenizer()
            pad_id = getattr(tok, "pad_token_id", None)
            if pad_id is None:
                pad_id = getattr(tok, "eos_token_id", 0)
            out_attn = (outputs != pad_id).long().to(device)

            # mllama: cross_attention_mask must match output seq_len
            probe_inputs = dict(
                input_ids=outputs,
                attention_mask=out_attn,
            )

            if "cross_attention_mask" in mm_kwargs and mm_kwargs["cross_attention_mask"] is not None:
                cam_base = mm_kwargs["cross_attention_mask"]
                cam_full = self._extend_seq_mask(cam_base, new_len=int(outputs.size(1)))
                if cam_full.size(0) == 1 and curr_bs > 1:
                    cam_full = cam_full.repeat(curr_bs, *([1] * (cam_full.dim() - 1)))
                probe_inputs["cross_attention_mask"] = cam_full

            # For scoring full sequences, include vision inputs (pixel_values/aspect ratios) if present.
            # This is the simplest correct path for mllama.
            for k, v in mm_kwargs.items():
                if k == "cross_attention_mask":
                    continue
                if torch.is_tensor(v) and v.size(0) == 1 and curr_bs > 1:
                    probe_inputs[k] = v.repeat(curr_bs, *([1] * (v.dim() - 1)))
                else:
                    probe_inputs[k] = v

            with torch.inference_mode():
                probe_out = self._safe_probe_forward(probe_inputs)

            logits = probe_out["probe_logits"]  # [B, T_out, C]
            probs = torch.softmax(logits, dim=-1)

            # class0 faithfulness over generated part
            gen_probs = probs[:, prompt_len:, 0]  # [B, T_gen]
            if gen_probs.numel() > 0 and gen_probs.size(1) > 0:
                avg_faith = gen_probs.mean(dim=1)
            else:
                avg_faith = torch.zeros((curr_bs,), device=device)

            for i in range(curr_bs):
                all_generated_ids.append(outputs[i])
                all_scores.append(float(avg_faith[i].item()))

        best_idx = int(torch.argmax(torch.tensor(all_scores, device=device)).item())
        best_seq = all_generated_ids[best_idx].unsqueeze(0)

        return self.decode(best_seq[:, prompt_len:])

