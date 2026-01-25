import os
import json
import argparse
from typing import Any, Dict, List, Tuple, Optional

import torch
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import login, HfApi
from transformers import AutoProcessor, AutoModelForImageTextToText


# ---------------------- 1) Steering Manager ----------------------
class SteeringManager:
    def __init__(self, model, layer_idx: int, steering_vector: torch.Tensor, coefficient: float = 1.0):
        self.model = model
        self.layer_idx = layer_idx
        self.steering_vector = steering_vector
        self.coefficient = float(coefficient)
        self.hook_handle = None
        self._vec_cached = None

    def _find_language_layers(self):
        candidates = [
            lambda m: m.model.language_model.layers,
            lambda m: m.model.language_model.model.layers,
            lambda m: m.model.model.language_model.layers,
            lambda m: m.language_model.model.layers,
            lambda m: m.language_model.layers,
            lambda m: m.model.layers,
        ]
        last = None
        for getter in candidates:
            try:
                layers = getter(self.model)
                if layers is not None and len(layers) > 0:
                    return list(layers)
            except Exception as e:
                last = e
        raise RuntimeError(f"Cannot locate language model layers. Last error: {last}")

    def _steering_hook(self, module, inputs, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(hidden_states):
            return output

        if (self._vec_cached is None
            or self._vec_cached.device != hidden_states.device
            or self._vec_cached.dtype != hidden_states.dtype):
            self._vec_cached = self.steering_vector.to(hidden_states.device, dtype=hidden_states.dtype)

        hidden_states = hidden_states + self.coefficient * self._vec_cached  # [H] broadcast -> [B,T,H]
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states

    def register(self):
        layers = self._find_language_layers()
        if not (0 <= self.layer_idx < len(layers)):
            raise ValueError(f"layer_idx={self.layer_idx} out of range (0..{len(layers)-1})")
        self.hook_handle = layers[self.layer_idx].register_forward_hook(self._steering_hook)
        print(f"--> Steering hook registered at layer {self.layer_idx}")

    def remove(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
            print("--> Steering hook removed")


# ---------------------- 2) Build steering vector from HDF5 ----------------------
def read_h5_(name):
    import h5py
    hidden_list_img = []
    hidden_list_txt = []
    with h5py.File(name, "r") as f:
        data_group = f["data"]
        for entry_name in data_group:
            entry = data_group[entry_name]
            hidden_state_img1 = torch.tensor(entry["result_image_hidden_states_1"][()], dtype=torch.bfloat16)
            hidden_state_text1 = torch.tensor(entry["result_text_hidden_states_1"][()], dtype=torch.bfloat16)
            hidden_list_img.append(hidden_state_img1)
            hidden_list_txt.append(hidden_state_text1)
    return hidden_list_img, hidden_list_txt


def extract_last_token_per_layer(h: torch.Tensor) -> torch.Tensor:
    # 兼容多余前缀维度：一直取第0个直到变成 [L,S,H]
    while h.dim() > 3:
        h = h[0]
    if h.dim() != 3:
        raise ValueError(f"Unexpected hidden shape: {tuple(h.shape)}")
    return h[1:, -1, :]  # [L-1, H]


def build_steering_vector(h5_path: str, layer_idx: int) -> torch.Tensor:
    img_list, txt_list = read_h5_(h5_path)
    img = torch.stack(img_list).to(torch.float32)
    txt = torch.stack(txt_list).to(torch.float32)

    img_layers = torch.stack([extract_last_token_per_layer(x) for x in img], dim=0)  # [N,L-1,H]
    txt_layers = torch.stack([extract_last_token_per_layer(x) for x in txt], dim=0)  # [N,L-1,H]

    image_vs_text = img_layers - txt_layers
    mean_vecs = image_vs_text.mean(dim=0)  # [L-1,H]

    if not (0 <= layer_idx < mean_vecs.shape[0]):
        raise ValueError(f"layer_idx={layer_idx} out of range (0..{mean_vecs.shape[0]-1})")
    return mean_vecs[layer_idx].contiguous()  # [H]


# ---------------------- 3) Helpers: build batched inputs ----------------------
def make_chat_text(processor, prompt_text: str, with_image: bool) -> str:
    if with_image:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},  # 只放占位符
                {"type": "text", "text": prompt_text},
            ],
        }]
    else:
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
            ],
        }]

    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def normalize_image_obj(img):
    """
    datasets.Image 可能返回 PIL.Image，也可能是 dict(None/path/bytes) 取决于加载方式
    - None: 表示无图
    - PIL: 直接用
    - dict: 尝试把 'path' 交给 processor（它支持路径/URL）
    """
    if img is None:
        return None
    # PIL.Image.Image
    try:
        from PIL import Image as PILImage
        if isinstance(img, PILImage.Image):
            return img.convert("RGB") if img.mode != "RGB" else img
    except Exception:
        pass

    if isinstance(img, dict):
        # datasets 有时会给 {"path": "..."} 或 {"bytes": ...}
        if img.get("path", None) is not None:
            return img["path"]
        if img.get("bytes", None) is not None:
            return img  # 让 HF image processor 自己处理
    return img


@torch.inference_mode()
def run_generate_batch(model, processor, tok, texts: List[str], images: Optional[List[Any]],
                       max_new_tokens: int, do_sample: bool, temperature: float, top_p: float) -> List[str]:
    if images is None:
        inputs = processor(
            text=texts,
            return_tensors="pt",
            padding=True,
        )
    else:
        inputs = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )

    inputs = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    gen_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        use_cache=True,
        pad_token_id=tok.pad_token_id,
    )

    # 注意：padding 后所有样本 input_len 一致，所以可以统一切片
    input_len = inputs["input_ids"].shape[1]
    new_tokens = gen_ids[:, input_len:]
    outs = processor.batch_decode(new_tokens, skip_special_tokens=True)
    return outs


# ---------------------- 4) Main: TRUE batching (split with/without image) ----------------------
@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    # model
    # parser.add_argument("--model_path", type=str, default="/new_disk/lhl1/models/Llama-3.2V-11B-cot")
    # parser.add_argument("--model_path", type=str, default="/new_disk/lhl1/models/R1-Onevision-7B")
    parser.add_argument("--model_path", type=str, default="/new_disk/lhl1/models/Ocean_R1_7B_Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])

    # steering
    parser.add_argument("--h5_path", type=str, default="/new_disk/lhl1/Modality-Preference/results1/hidden_states_single_template1/icc/base_multi/R1-Onevision-7B.h5")
    parser.add_argument("--layer_idx", type=int, default=22)
    parser.add_argument("--coefficient", type=float, default=-1.0)

    # generation
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)

    # datasets
    # parser.add_argument("--source_repo", type=str, default="1Jin1/TriConflict-hallucination-annotations-80_20")
    # parser.add_argument("--source_subset", type=str, default="Llama-3.2V-11B-Cot")
    parser.add_argument("--source_repo", type=str, default="alita01/TriConflict-hallucination-annotations-80_20")
    parser.add_argument("--source_subset", type=str, default="Ocean_R1_7B_Instruct")
    parser.add_argument("--source_split", type=str, default="test")

    parser.add_argument("--target_repo", type=str, default="alita01/TriConflict-Mitigation")
    parser.add_argument("--target_subset", type=str, default="Ocean_R1_7B_Instruct")
    parser.add_argument("--target_split", type=str, default="steering-neg")

    # resume/checkpoint
    parser.add_argument("--save_jsonl", type=str, default="steering_ckpt_Ocean_R1_7B_Instruct-neg.jsonl")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save_every", type=int, default=1)

    # hf
    parser.add_argument("--hf_token", type=str, default=os.getenv("HF_TOKEN", ""))

    args = parser.parse_args()

    if args.hf_token:
        login(token=args.hf_token)

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    ).to(args.device).eval()

    # tokenizer padding（batch 必需）
    tok = processor.tokenizer
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    # tok.padding_side = "left"

    steering_vec = build_steering_vector(args.h5_path, args.layer_idx)  # float32 [H]
    steer_manager = SteeringManager(model, args.layer_idx, steering_vec, args.coefficient)

    ds = load_dataset(args.source_repo, args.source_subset, split=args.source_split)

    # resume map
    output_map: Dict[str, str] = {}
    if args.resume and os.path.exists(args.save_jsonl):
        with open(args.save_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                output_map[obj["question_id"]] = obj["model_output"]
        print(f"[Resume] loaded {len(output_map)} outputs from {args.save_jsonl}")

    def flush_ckpt():
        with open(args.save_jsonl, "w", encoding="utf-8") as f:
            for qid, out_text in output_map.items():
                f.write(json.dumps({"question_id": qid, "model_output": out_text}, ensure_ascii=False) + "\n")

    # ---- TRUE batching + split with/without image ----
    steer_manager.register()
    newly_done = 0
    try:
        indices = list(range(len(ds)))
        pbar = tqdm(range(0, len(indices), args.batch_size), desc=f"Batch steering (bs={args.batch_size})")

        for start in pbar:
            batch_idx = indices[start:start + args.batch_size]
            batch_ex = [ds[i] for i in batch_idx]

            # 过滤已完成
            todo = [ex for ex in batch_ex if ex["question_id"] not in output_map]
            if len(todo) == 0:
                continue

            # 分组：有图 / 无图
            with_img = []
            no_img = []
            for ex in todo:
                img = normalize_image_obj(ex.get("image", None))
                if img is None:
                    no_img.append(ex)
                else:
                    with_img.append((ex, img))

            # 1) 有图组 batch
            if len(with_img) > 0:
                texts = []
                images = []
                qids = []
                for ex, img in with_img:
                    prompt_text = ex.get("detailed_prompt") or ex.get("question") or ""
                    texts.append(make_chat_text(processor, prompt_text, with_image=True))
                    images.append(img)
                    qids.append(ex["question_id"])

                outs = run_generate_batch(
                    model, processor, tok,
                    texts=texts,
                    images=images,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p
                )
                for qid, out_text in zip(qids, outs):
                    output_map[qid] = out_text
                newly_done += len(qids)

            # 2) 无图组 batch（关键：processor 不传 images）
            if len(no_img) > 0:
                texts = []
                qids = []
                for ex in no_img:
                    prompt_text = ex.get("detailed_prompt") or ex.get("question") or ""
                    texts.append(make_chat_text(processor, prompt_text, with_image=False))
                    qids.append(ex["question_id"])

                outs = run_generate_batch(
                    model, processor, tok,
                    texts=texts,
                    images=None,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p
                )
                for qid, out_text in zip(qids, outs):
                    output_map[qid] = out_text
                newly_done += len(qids)

            if newly_done >= args.save_every:
                flush_ckpt()
                newly_done = 0

        flush_ckpt()

    finally:
        steer_manager.remove()

    # ---- attach outputs & align schema & push ----
    def attach_output(ex):
        qid = ex["question_id"]
        ex["model_output"] = output_map.get(qid, "")
        return ex

    ds_out = ds.map(attach_output, desc="Attach steering outputs")

    keep_cols = [
        "question_id",
        "question",
        "model_output",
        "ground_truth",
        "image",
        "image_name",
        "detailed_prompt",
        "model_name",
        "conflict_type",
    ]
    drop_cols = [c for c in ds_out.column_names if c not in keep_cols]
    if drop_cols:
        ds_out = ds_out.remove_columns(drop_cols)

    if args.hf_token:
        api = HfApi(token=args.hf_token)
        api.create_repo(repo_id=args.target_repo, repo_type="dataset", exist_ok=True)

        ds_out.push_to_hub(
            args.target_repo,
            config_name=args.target_subset,
            split=args.target_split,
            token=args.hf_token,
        )
        print(f"[Pushed] {args.target_repo} / {args.target_subset} / split={args.target_split}")
    else:
        print("[Skip Push] No HF_TOKEN provided. Set --hf_token or env HF_TOKEN to push.")


if __name__ == "__main__":
    main()
