import torch
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
        self.tokenizer = None
        self.probe = None
        
        self.load_models()

    def load_models(self):
        """Loads the base LLM and the ValueHeadProbe."""
        # ===== 3、加载模型和processor/tokenizer =====
        print(f"Loading model: {self.model_name}")
        from transformers import AutoModelForImageTextToText, AutoProcessor
        # 检查是否是多模态模型
        self.is_multimodal = 'vision' in self.model_name.lower() or 'onevision' in self.model_name.lower()
        
        if self.is_multimodal:
            # 使用 AutoProcessor 代替 AutoTokenizer
            print(f"Loading multimodal model: {self.model_name}")
            processor = AutoProcessor.from_pretrained(self.model_name)
            
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                # device_map="auto",  
                trust_remote_code=True
            )
            self.tokenizer = processor
        else:
            # 原有的文本模型加载逻辑
            print("加载原来的文本模型")    
            self.model, self.tokenizer = load_model_and_tokenizer(self.model_name)
            processor = None  # 非多模态模型不需要 processor
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
        if hasattr(self.tokenizer, "tokenizer"):
            return self.tokenizer.tokenizer
        return self.tokenizer

    def encode(self, text: str):
        """Helper to encode text."""
        tokenizer = self._get_tokenizer()
        return tokenizer.encode(text, return_tensors="pt").to(self.device)
        
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
        
        if self.is_multimodal:
            # (Chat template logic omitted for brevity, keeping existing flow but adding attention_mask extraction)
            # ... [Same chat template logic as before] ...
            processor = self.tokenizer
            tokenizer = self._get_tokenizer()
            
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
            try:
                text_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception as e:
                # Fallback logic
                if image is not None:
                     content_str = "<image>\n" + prompt
                else:
                     content_str = prompt
                
                messages_simple = [{"role": "user", "content": content_str}]
                try:
                    text_prompt = self.tokenizer.apply_chat_template(messages_simple, tokenize=False, add_generation_prompt=True)
                except Exception as e2:
                    print(f"String-based chat template failed: {e2}. Using raw prompt.")
                    text_prompt = f"User: {content_str}\nAssistant:"
            
            try:
                inputs = processor(text=[text_prompt], images=image, return_tensors="pt")
            except Exception as e:
                print(f"Processor encoding failed: {e}. Fallback to direct encode.")
                inputs = {"input_ids": self.encode(prompt)}

            input_ids = inputs["input_ids"].to(self.device)
            if "pixel_values" in inputs:
                pixel_values = inputs["pixel_values"].to(self.device)
            if "image_grid_thw" in inputs:
                image_grid_thw = inputs["image_grid_thw"].to(self.device)
            if "attention_mask" in inputs:
                attention_mask = inputs["attention_mask"].to(self.device)

        else:
            # Text-only model
            input_ids = self.encode(prompt)
            # Encode helper doesn't usually return attention_mask unless we ask, but it's good practice.
            # self.encode calls tokenizer.encode(return_tensors="pt"). 
            # Actually self.encode implementation: return tokenizer.encode(text, return_tensors="pt")
            # This returns just ids tensor if not return_dict=True? 
            # tokenizer.encode returns Tensor directly. tokenizer(text, return_tensors="pt") returns dict.
            # Our self.encode helper: tokenizer.encode(text, return_tensors="pt") -> Tensor.
            # So attention_mask is lost there.
            # Let's check self.encode. It does: tokenizer.encode(..., return_tensors="pt").
            # To get mask we should use tokenizer(...) instead.
            # For now, let's assume if it's not multimodal, we might not have mask or need to create one.
            # But let's try to get it if we can.
            pass
        
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
            return self.decode(output_ids)

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
        Token-Level Rejection Sampling.
        
        This version does NOT use KV-Cache to ensure correctness and compatibility
        with all models. It's slower but reliable.
        """
        # input_ids: [1, Seq]
        current_ids = input_ids.clone()
        current_mask = attention_mask.clone() if attention_mask is not None else None
        
        for step in range(max_new_tokens):
            # 1. Get LM logits for the next token (full forward pass)
            with torch.no_grad():
                model_inputs = {"input_ids": current_ids}
                if pixel_values is not None:
                    model_inputs["pixel_values"] = pixel_values
                if image_grid_thw is not None:
                    model_inputs["image_grid_thw"] = image_grid_thw
                if current_mask is not None:
                    model_inputs["attention_mask"] = current_mask
                    
                outputs = self.model(**model_inputs)
                next_token_logits = outputs.logits[:, -1, :]  # [1, Vocab]
                
            # 2. Sample candidates (Top-K from LM)
            if temperature > 0:
                probs = F.softmax(next_token_logits / temperature, dim=-1)
            else:
                probs = F.softmax(next_token_logits, dim=-1) 
                
            if num_candidates > 1:
                candidate_probs, candidate_ids = torch.topk(probs, num_candidates, dim=-1)
            else:
                candidate_probs, candidate_ids = torch.topk(probs, max(5, num_candidates), dim=-1)
                candidate_ids = candidate_ids[:, :num_candidates]
                candidate_probs = candidate_probs[:, :num_candidates]

            # 3. Evaluate each candidate with the probe (skip if alpha=0)
            if alpha > 0:
                batch_candidate_tokens = candidate_ids.view(-1, 1)  # [K, 1]
                hallucination_scores = self._evaluate_candidates_with_cache(
                    current_ids=current_ids,
                    candidate_tokens=batch_candidate_tokens,
                    past_key_values=None,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    attention_mask=current_mask,
                    num_candidates=num_candidates
                )  # [K]
                truth_scores = 1.0 - hallucination_scores
            else:
                # Skip probe evaluation if alpha=0 (pure LM mode)
                truth_scores = torch.zeros(num_candidates, device=self.device)
            
            # 4. Calculate Combined Score
            lm_scores = candidate_probs.view(-1)  # [K]
            combined_scores = (1 - alpha) * lm_scores + alpha * truth_scores
            
            # Select best candidate
            best_idx = torch.argmax(combined_scores)
            best_token = candidate_ids[0, best_idx]
            
            # Append the best token
            current_ids = torch.cat([current_ids, best_token.view(1, 1)], dim=1)
            if current_mask is not None:
                current_mask = torch.cat([current_mask, torch.ones((1, 1), device=self.device, dtype=current_mask.dtype)], dim=1)
            
            # Stop if EOS
            if best_token == self._get_tokenizer().eos_token_id:
                break
                
        return self.decode(current_ids)
    
    def _evaluate_candidates_with_cache(
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
        """
        Expand past_key_values for batch evaluation of K candidates.
        
        past_key_values structure varies by model:
        - For tuple format: tuple of (key, value) tensors per layer
        - For DynamicCache format (Qwen2.5-VL, etc.): Cache object with key_cache/value_cache
        - key/value shape: [batch, num_heads, seq_len, head_dim]
        """
        if past_key_values is None:
            return None
        
        # Check if it's a DynamicCache or similar Cache object (used by Qwen2.5-VL)
        if hasattr(past_key_values, 'key_cache') and hasattr(past_key_values, 'value_cache'):
            # It's a Cache object (DynamicCache, etc.)
            from transformers.cache_utils import DynamicCache
            
            expanded_cache = DynamicCache()
            
            for layer_idx in range(len(past_key_values.key_cache)):
                key = past_key_values.key_cache[layer_idx]
                value = past_key_values.value_cache[layer_idx]
                
                # Expand batch dimension: [1, num_heads, seq_len, head_dim] -> [K, num_heads, seq_len, head_dim]
                expanded_key = key.repeat(num_candidates, 1, 1, 1)
                expanded_value = value.repeat(num_candidates, 1, 1, 1)
                
                # Update the cache (this appends to key_cache and value_cache lists)
                expanded_cache.update(expanded_key, expanded_value, layer_idx)
            
            return expanded_cache
        
        # Fallback: tuple format (legacy models)
        if isinstance(past_key_values, tuple):
            expanded = []
            for layer_past in past_key_values:
                if isinstance(layer_past, tuple):
                    # Standard format: (key, value)
                    key, value = layer_past
                    # Expand batch dimension
                    expanded_key = key.repeat(num_candidates, 1, 1, 1)
                    expanded_value = value.repeat(num_candidates, 1, 1, 1)
                    expanded.append((expanded_key, expanded_value))
                else:
                    # Some models use different formats
                    # Try to handle gracefully
                    if hasattr(layer_past, 'repeat'):
                        expanded.append(layer_past.repeat(num_candidates, *([1] * (layer_past.dim() - 1))))
                    else:
                        expanded.append(layer_past)
            
            return tuple(expanded)
        
        # If we can't handle it, return as-is and hope for the best
        return past_key_values

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
