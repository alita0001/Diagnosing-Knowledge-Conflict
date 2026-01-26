import torch
import copy
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
from transformers import PreTrainedTokenizer

from probe.value_head_probe import ValueHeadProbe
from utils.model_utils import load_model_and_tokenizer
from probe.config import ProbeConfig

class HallucinationMitigator:
    """
    Handles hallucination mitigation using a trained ValueHeadProbe.
    """
    def __init__(
        self,
        model_name: str,
        probe_path: str,
        device: str = "cuda"
    ):
        self.device = device
        self.model_name = model_name
        self.probe_path = probe_path
        
        self.model = None
        self.processor = None
        self.probe = None
        
        self.load_models()

    def load_models(self):
        """Loads the base LLM and the ValueHeadProbe."""
        # ===== 3、加载模型和processor/tokenizer =====
        print(f"Loading model: {self.model_name}")
        from transformers import AutoModelForImageTextToText, AutoProcessor
        # 检查是否是多模态模型
        
        # 使用 AutoProcessor 代替 AutoTokenizer
        print(f"Loading multimodal model: {self.model_name}")
        processor = AutoProcessor.from_pretrained(self.model_name)
        
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            # device_map="auto",  
            trust_remote_code=True
        )
        self.processor = processor
        self.model.eval()
        # Ensure model is on the correct device
        self.model.to(self.device)

        print(f"Loading probe from: {self.probe_path}")
        # We need to construct a dummy ProbeConfig or load from the path
        # ValueHeadProbe handles loading from path internally if path is provided
        from pathlib import Path
        probe_path_obj = Path(self.probe_path)
        
        # Initialize probe with the loaded model
        self.probe = ValueHeadProbe(self.model, path=probe_path_obj)
        self.probe.to(self.device)
        self.probe.eval()
        print("Models loaded successfully.")

    def _get_tokenizer(self):
        """Helper to get the actual tokenizer object from processor if needed."""
        # if hasattr(self.processor, "tokenizer"):
        #     return self.processor.tokenizer
        return self.processor
        
    def decode(self, token_ids: torch.Tensor, skip_special_tokens=True):
        """Helper to decode token ids."""
        tokenizer = self._get_tokenizer()
        if isinstance(token_ids, torch.Tensor) and token_ids.dim() > 1:
            token_ids = token_ids[0]
        return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def predict_hallucination_score(self, input_ids: torch.Tensor, pixel_values: Optional[torch.Tensor] = None, image_grid_thw: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Returns the hallucination probability for the LAST token in the input_ids.
        Score = Sum of Probabilities of Classes 1, 2, 3.
        Class 0 is 'Faithful'.
        """
        with torch.no_grad():
            inputs = {"input_ids": input_ids}
            if pixel_values is not None:
                inputs["pixel_values"] = pixel_values
            if image_grid_thw is not None:
                inputs["image_grid_thw"] = image_grid_thw
            if attention_mask is not None:
                inputs["attention_mask"] = attention_mask
                
            outputs = self.probe(inputs=inputs)
            logits = outputs["probe_logits"] # [Batch, Seq, NumClasses]
            
            # We are interested in the last token's prediction
            last_token_logits = logits[:, -1, :] # [Batch, NumClasses]
            probs = torch.softmax(last_token_logits, dim=-1)
            
            # Hallucination Score = 1 - P(Faithful) = P(Class 1) + P(Class 2) + P(Class 3)
            # Assuming Class 0 is Faithful.
            hallucination_score = 1.0 - probs[:, 0]
            
            return hallucination_score

    def generate(
        self,
        prompt: str,
        image = None, # PIL Image or similar
        method: str = "token_rejection",
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_k: int = 5,
        alpha: float = 0.5, # Weight for combining scores (used in token_rejection)
        num_candidates: int = 5, # N for sentence rejection, or K for token rejection
        sentence_rejection_batch_size: int = 2
    ) -> str:
        """
        Generates text using the specified mitigation method.
        Accepts optional `image` for multimodal models.
        """
        pixel_values = None
        image_grid_thw = None
        attention_mask = None

        # (Chat template logic omitted for brevity, keeping existing flow but adding attention_mask extraction)
        # ... [Same chat template logic as before] ...
        processor = self.processor
        
        # Construct messages for chat template
        if image is not None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
        
        # Apply chat template to get formatted text
        text_prompt = None
        text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = processor(text=[text_prompt], images=image, return_tensors="pt")

        input_ids = inputs["input_ids"].to(self.device)
        if "pixel_values" in inputs:
            pixel_values = inputs["pixel_values"].to(self.device)
        if "image_grid_thw" in inputs:
            image_grid_thw = inputs["image_grid_thw"].to(self.device)
        if "attention_mask" in inputs:
            attention_mask = inputs["attention_mask"].to(self.device)

        
        if method == "token_rejection":
            return self._generate_token_rejection(
                input_ids, max_new_tokens, temperature, top_k, alpha, num_candidates, pixel_values, image_grid_thw, attention_mask
            )
        elif method == "sentence_rejection":
            return self._generate_sentence_rejection(
                input_ids, max_new_tokens, temperature, num_candidates, sentence_rejection_batch_size, pixel_values, image_grid_thw, attention_mask
            )
        else:
            # Fallback to standard generation
            gen_kwargs = {"input_ids": input_ids, "max_new_tokens": max_new_tokens, "temperature": temperature, "top_k": top_k}
            if pixel_values is not None:
                gen_kwargs["pixel_values"] = pixel_values
            if image_grid_thw is not None:
                gen_kwargs["image_grid_thw"] = image_grid_thw
            if attention_mask is not None:
                gen_kwargs["attention_mask"] = attention_mask
                
            output_ids = self.model.generate(**gen_kwargs)
            # print(processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
            return self.decode(output_ids)
            # return self.processor.decode(output_ids[0][input_ids.size(1):], skip_special_tokens=True)

    def _generate_token_rejection(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        alpha: float,
        num_candidates: int,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> str:
        """
        Token-Level Rejection Sampling (FAST, Qwen2.5-VL friendly).

        关键修复：
        - **alpha==0**：直接走 `model.generate`（等价 baseline，避免自写 decode path 触发 cache_position/position 问题）
        - **alpha>0**：自写 KV-cache decoding，但为 Qwen2.5-VL 显式传 `cache_position`
        - 只返回生成部分（不包含 prompt/system/user）
        """
        device = input_ids.device

        # -------- alpha==0 => baseline generation --------
        if alpha is None or float(alpha) <= 0.0:
            gen_kwargs = {
                "input_ids": input_ids,
                "max_new_tokens": int(max_new_tokens),
                "do_sample": False,  # greedy
                "use_cache": True,
                "return_dict_in_generate": False,
            }
            if attention_mask is not None:
                gen_kwargs["attention_mask"] = attention_mask
            if pixel_values is not None:
                gen_kwargs["pixel_values"] = pixel_values
            if image_grid_thw is not None:
                gen_kwargs["image_grid_thw"] = image_grid_thw

            with torch.inference_mode():
                out = self.model.generate(**gen_kwargs)
            prompt_len = int(input_ids.size(1))
            return self.decode(out[:, prompt_len:])

        # -------- alpha>0 => token rejection decoding --------
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device, dtype=torch.long)

        prompt_len = int(input_ids.size(1))
        max_len = prompt_len + int(max_new_tokens)

        out_ids = input_ids.new_empty((1, max_len))
        out_ids[:, :prompt_len] = input_ids
        out_mask = attention_mask.new_empty((1, max_len))
        out_mask[:, :prompt_len] = attention_mask

        cur_len = prompt_len

        # ---- Prefill ----
        with torch.inference_mode():
            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "use_cache": True,
                "return_dict": True,
            }
            if pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values
            if image_grid_thw is not None:
                model_inputs["image_grid_thw"] = image_grid_thw

            # Qwen2.5-VL / 新 cache：需要 cache_position 才能正确对齐 RoPE/Cache 更新
            cache_pos = torch.arange(prompt_len, device=device, dtype=torch.long)
            model_inputs["cache_position"] = cache_pos
            outputs = self.model(**model_inputs)

            past = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :]

        # 多模态：后续 steps 不再传 pixel_values（避免重复视觉编码）
        step_pixel_values = None if (pixel_values is not None) else pixel_values

        # decoding loop
        eos_id = getattr(self.processor.tokenizer, "eos_token_id", None)
        alpha = float(alpha)

        # 生成历史（用于轻度 repetition 控制）
        gen_history: List[int] = []
        recent_window = 64
        rep_penalty = 0.3  # log-space penalty

        for _ in range(int(max_new_tokens)):
            prefix_len = cur_len  # 当前 prefix 长度
            # candidates: respect top_k and num_candidates
            K = int(num_candidates) if int(num_candidates) > 0 else 1
            if top_k is not None and int(top_k) > 0:
                K = min(K, int(top_k))
            K = max(1, K)

            # scale logits
            if temperature is None or float(temperature) <= 0:
                scaled = next_logits
            else:
                scaled = next_logits / float(temperature)

            topk_logits, topk_ids = torch.topk(scaled, k=K, dim=-1)  # [1,K]
            lm_logp = F.log_softmax(topk_logits, dim=-1).view(-1)     # [K]
            cand_tokens = topk_ids.view(K, 1)                          # [K,1]

            # probe truth scores (sequential for Qwen cache)
            halluc_scores = self._evaluate_candidates_incremental(
                candidate_tokens=cand_tokens,
                past_key_values=past,
                pixel_values=step_pixel_values,
                image_grid_thw=image_grid_thw,
                attention_mask=out_mask[:, :cur_len],
                num_candidates=K,
            )  # [K] larger => more halluc
            truth_scores = (1.0 - halluc_scores).clamp(min=1e-6, max=1.0)
            truth_log = torch.log(truth_scores)

            # alpha warm-up: 前 16 token 逐步增强
            warm = min(1.0, (len(gen_history) + 1) / 16.0)
            alpha_eff = alpha * warm

            combined = lm_logp + alpha_eff * truth_log  # [K]

            # light repetition penalty (only when we already started generating)
            if gen_history:
                recent = set(gen_history[-recent_window:])
                for j in range(K):
                    tid = int(topk_ids[0, j].item())
                    if tid in recent:
                        combined[j] = combined[j] - rep_penalty

            best_j = int(torch.argmax(combined).item())
            best_id = topk_ids[0, best_j].view(1, 1)

            # stop on eos
            if eos_id is not None and int(best_id.item()) == int(eos_id):
                break

            # append to buffer
            out_ids[:, cur_len] = best_id
            out_mask[:, cur_len] = 1
            gen_history.append(int(best_id.item()))
            cur_len += 1

            # next step forward: only 1 token + cache
            with torch.inference_mode():
                step_inputs = {
                    "input_ids": best_id,
                    "past_key_values": past,
                    "attention_mask": out_mask[:, :cur_len],
                    "use_cache": True,
                    "return_dict": True,
                }
                # IMPORTANT: cache_position for this new token
                step_inputs["cache_position"] = torch.tensor([prefix_len], device=device, dtype=torch.long)

                # 多模态一般不需要再传 pixel_values；如果某些模型需要，保留兜底
                if step_pixel_values is not None:
                    step_inputs["pixel_values"] = step_pixel_values
                if image_grid_thw is not None:
                    step_inputs["image_grid_thw"] = image_grid_thw

                step_out = self.model(**step_inputs)

                past = step_out.past_key_values
                next_logits = step_out.logits[:, -1, :]

        return self.decode(out_ids[:, prompt_len:cur_len])


    def _evaluate_candidates_incremental(
        self,
        candidate_tokens: torch.Tensor,
        past_key_values,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        num_candidates: int = 5,
    ) -> torch.Tensor:
        """
        Score candidates by running the **probe** on a single next-token step using the prefix KV-cache.

        Qwen2.5-VL (Transformers Cache/DynamicCache) 注意点：
        - `past_key_values` 是 Cache 对象，forward 会 **in-place update**
        - batch 扩展 Cache 很容易踩坑，因此这里走 **逐候选 sequential**，并对 cache 做 copy/deepcopy
        - 为了避免 position/rope 对齐错误，显式传 `cache_position = [prefix_len]`
        """
        K = int(num_candidates) if int(num_candidates) > 0 else int(candidate_tokens.size(0))
        device = candidate_tokens.device

        if past_key_values is None:
            return torch.zeros((K,), device=device, dtype=torch.float32)

        # prefix length from attention mask
        prefix_len = int(attention_mask.size(1)) if attention_mask is not None else None
        if prefix_len is None:
            # best effort: fall back to 0 (should rarely happen)
            prefix_len = 0

        scores = []
        for i in range(K):
            tok_i = candidate_tokens[i:i+1]  # [1,1]

            # defensive copy (Cache updates in-place)
            pkv_i = past_key_values

            if hasattr(past_key_values, "copy"):
                pkv_i = past_key_values.copy()
            else:
                pkv_i = copy.deepcopy(past_key_values)

            probe_inputs = {
                "input_ids": tok_i,
                "past_key_values": pkv_i,
                "use_cache": True,
                "return_dict": True,
            }
            if attention_mask is not None:
                # need include the new token position as attended
                am = torch.ones((1, prefix_len + 1), device=device, dtype=attention_mask.dtype)
                if prefix_len > 0:
                    am[:, :prefix_len] = attention_mask[:, :prefix_len]
                probe_inputs["attention_mask"] = am

            # critical: cache_position for this new token
            probe_inputs["cache_position"] = torch.tensor([prefix_len], device=device, dtype=torch.long)

            # multimodal kwargs: usually not needed after prefill; keep only if provided explicitly
            if pixel_values is not None:
                probe_inputs["pixel_values"] = pixel_values
            if image_grid_thw is not None:
                probe_inputs["image_grid_thw"] = image_grid_thw

            with torch.inference_mode():
                out = self.probe(inputs=probe_inputs)

            # --- extract "truth" probability from probe output ---
            p_truth_t = None
            if isinstance(out, dict):
                if "probs" in out:
                    probs_t = out["probs"]
                    # allow [B,C] or [B,T,C]
                    if torch.is_tensor(probs_t) and probs_t.dim() == 3:
                        probs_t = probs_t[:, -1, :]
                    if torch.is_tensor(probs_t):
                        p_truth_t = probs_t[:, 0]
                elif "probe_logits" in out:
                    probe_logits = out["probe_logits"]
                    if torch.is_tensor(probe_logits) and probe_logits.dim() == 3:
                        probe_logits = probe_logits[:, -1, :]
                    if torch.is_tensor(probe_logits):
                        p_truth_t = torch.softmax(probe_logits.float(), dim=-1)[:, 0]
                elif "logits" in out:
                    lg = out["logits"]
                    if torch.is_tensor(lg) and lg.dim() == 3:
                        lg = lg[:, -1, :]
                    if torch.is_tensor(lg):
                        p_truth_t = torch.softmax(lg.float(), dim=-1)[:, 0]
            elif torch.is_tensor(out):
                probs_t = out
                if probs_t.dim() == 3:
                    probs_t = probs_t[:, -1, :]
                p_truth_t = probs_t[:, 0]

            if p_truth_t is None:
                # fallback: neutral score
                p_truth = torch.tensor(0.5, device=device, dtype=torch.float32)
            else:
                p_truth = p_truth_t[0].float().clamp(0, 1)

            # hallucination score = 1 - P(truth)
            scores.append((1.0 - p_truth).detach())

        return torch.stack(scores, dim=0).to(device=device)

    def _evaluate_candidates_with_cache(
        self,
        current_ids: torch.Tensor,
        candidate_tokens: torch.Tensor,
        past_key_values,
        pixel_values: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        num_candidates: int
    ) -> torch.Tensor:
        """
        Backward-compatible wrapper.

        - 当 past_key_values 可用时：走 _evaluate_candidates_incremental（KV-Cache, FAST）
        - 否则：退化到全序列评估（兼容性更强但更慢）
        """
        if past_key_values is not None:
            return self._evaluate_candidates_incremental(
                candidate_tokens=candidate_tokens,
                past_key_values=past_key_values,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
                num_candidates=num_candidates,
            )
        return self._evaluate_candidates_full_sequence(
            current_ids=current_ids,
            candidate_tokens=candidate_tokens,
            past_key_values=None,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            num_candidates=num_candidates,
        )

    def _evaluate_candidates_full_sequence(
        self,
        current_ids: torch.Tensor,
        candidate_tokens: torch.Tensor,
        past_key_values,  # Not used in this version - kept for API compatibility
        pixel_values: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        num_candidates: int
    ) -> torch.Tensor:
        """
        Evaluate hallucination scores for candidate tokens.
        
        Note: KV-Cache batching is complex for models like Qwen2.5-VL that use 
        DynamicCache. We use full sequence evaluation for probe which is simpler
        and more compatible. The main LM forward pass still benefits from KV-Cache.
        
        Args:
            current_ids: Current generated sequence [1, Seq]
            candidate_tokens: Candidate next tokens [K, 1]
            past_key_values: (Unused) kept for API compatibility
            pixel_values: Image features
            image_grid_thw: Image grid info
            attention_mask: Current attention mask [1, Seq]
            num_candidates: Number of candidates K
            
        Returns:
            Hallucination scores for each candidate [K]
        """
        with torch.no_grad():
            # Build full sequences for each candidate: [K, Seq+1]
            batch_current_ids = current_ids.repeat(num_candidates, 1)  # [K, Seq]
            batch_input_ids = torch.cat([batch_current_ids, candidate_tokens], dim=1)  # [K, Seq+1]
            
            # Prepare batch inputs (without KV-Cache to avoid compatibility issues)
            probe_inputs = {
                "input_ids": batch_input_ids,  # [K, Seq+1] - full sequences
            }
            
            # Only pass pixel_values on first step (when they are provided)
            if pixel_values is not None:
                if isinstance(pixel_values, torch.Tensor):
                    orig_shape = pixel_values.shape
                    repeat_dims = [num_candidates] + [1] * (len(orig_shape) - 1)
                    probe_inputs["pixel_values"] = pixel_values.repeat(*repeat_dims)
                else:
                    probe_inputs["pixel_values"] = pixel_values
            
            if image_grid_thw is not None:
                orig_shape = image_grid_thw.shape
                repeat_dims = [num_candidates] + [1] * (len(orig_shape) - 1)
                probe_inputs["image_grid_thw"] = image_grid_thw.repeat(*repeat_dims)
            
            # Attention mask for full sequence including new token
            if attention_mask is not None:
                batch_mask = attention_mask.repeat(num_candidates, 1)  # [K, Seq]
                new_token_mask = torch.ones((num_candidates, 1), device=self.device, dtype=attention_mask.dtype)
                batch_mask = torch.cat([batch_mask, new_token_mask], dim=1)  # [K, Seq+1]
                probe_inputs["attention_mask"] = batch_mask
            
            # Run probe forward (which internally runs model forward with hook)
            outputs = self.probe(inputs=probe_inputs)
            logits = outputs["probe_logits"]  # [K, Seq+1, NumClasses]
            
            # Get the last token's prediction
            last_token_logits = logits[:, -1, :]  # [K, NumClasses]
            probs = torch.softmax(last_token_logits, dim=-1)
            
            # Hallucination Score = 1 - P(Faithful) = 1 - P(Class 0)
            hallucination_scores = 1.0 - probs[:, 0]
            
            return hallucination_scores
    
    
    def _expand_past_key_values(self, past_key_values, num_candidates: int):
        """Try to expand KV cache from batch=1 to batch=K.

        Important for Qwen2.5-VL:
          - HF Qwen2.5-VL expects `past_key_values` to be a `Cache` object (has `.get_seq_length()`).
          - Converting to legacy tuple will crash (`tuple` has no `get_seq_length`).

        Strategy:
          - If past_key_values is a tuple (legacy): expand tensors and return tuple cache.
          - If past_key_values looks like a HF Cache object: return None (caller should use sequential
            scoring with per-candidate cache copies).
        """
        if past_key_values is None:
            return None

        # HF Cache / DynamicCache objects typically have get_seq_length/update methods.
        if not isinstance(past_key_values, tuple) and hasattr(past_key_values, "get_seq_length"):
            return None

        if not isinstance(past_key_values, tuple):
            # Unknown non-tuple cache type: safest to fall back to sequential.
            return None

        # Legacy tuple cache: tuple[(k,v), ...]
        expanded = []
        for layer_past in past_key_values:
            if layer_past is None:
                expanded.append(None)
                continue
            if not (isinstance(layer_past, tuple) and len(layer_past) == 2):
                expanded.append(layer_past)
                continue
            key, value = layer_past

            if torch.is_tensor(key) and key.size(0) == 1:
                key_k = key.expand(num_candidates, *key.shape[1:]).contiguous()
            else:
                key_k = key.repeat(num_candidates, *([1] * (key.dim() - 1)))

            if torch.is_tensor(value) and value.size(0) == 1:
                value_k = value.expand(num_candidates, *value.shape[1:]).contiguous()
            else:
                value_k = value.repeat(num_candidates, *([1] * (value.dim() - 1)))

            expanded.append((key_k, value_k))
        return tuple(expanded)

    def _generate_sentence_rejection(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        num_candidates: int,
        batch_size: int,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> str:
        """
        Sentence-Level Rejection Sampling.
        """
        all_generated_ids = []
        all_scores = []
        
        num_batches = (num_candidates + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            curr_batch_size = min(batch_size, num_candidates - i*batch_size)
            
            # Replicate input_ids for the batch
            batch_input = input_ids.repeat(curr_batch_size, 1)
            
            # Prepare batch inputs
            gen_kwargs = {
                "input_ids": batch_input,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": True,
                "top_k": 50
            }
            
            if pixel_values is not None:
                orig_shape = pixel_values.shape
                repeat_dims = [curr_batch_size] + [1] * (len(orig_shape) - 1)
                gen_kwargs["pixel_values"] = pixel_values.repeat(*repeat_dims)
            
            if image_grid_thw is not None:
                 orig_shape = image_grid_thw.shape
                 repeat_dims = [curr_batch_size] + [1] * (len(orig_shape) - 1)
                 gen_kwargs["image_grid_thw"] = image_grid_thw.repeat(*repeat_dims)
            
            if attention_mask is not None:
                gen_kwargs["attention_mask"] = attention_mask.repeat(curr_batch_size, 1)

            outputs = self.model.generate(**gen_kwargs)
            
            # Score each generated sequence
            probe_inputs = {"input_ids": outputs}
            if "pixel_values" in gen_kwargs:
                probe_inputs["pixel_values"] = gen_kwargs["pixel_values"]
            if "image_grid_thw" in gen_kwargs:
                probe_inputs["image_grid_thw"] = gen_kwargs["image_grid_thw"]
            # To be precise, we need attention mask for the full outputs too.
            # But generate usually handles padding. The outputs will have pad tokens.
            # We can create attention mask from pad tokens if needed,/
            # or rely on probe/model to handle it (usually they ignore pad if specified).
            # For simplicity, if we pass attention mask during generate, outputs are usually valid.
            
            probe_out = self.probe(inputs=probe_inputs)
            logits = probe_out["probe_logits"] # [Batch, Seq, 4]
            probs = torch.softmax(logits, dim=-1)
            
            input_len = input_ids.shape[1]
            gen_probs = probs[:, input_len:, 0] # Class 0 = Faithful
            
            if gen_probs.shape[1] > 0:
                avg_faith_scores = gen_probs.mean(dim=1) 
            else:
                avg_faith_scores = torch.zeros(curr_batch_size).to(self.device)
            
            all_generated_ids.extend([out for out in outputs])
            all_scores.extend(avg_faith_scores.tolist())
            
        best_idx = torch.argmax(torch.tensor(all_scores))
        best_sequence = all_generated_ids[best_idx]
        
        return self.decode(best_sequence)