from typing import Optional, List, Tuple, Dict, Any
import torch, math
from transformers import AutoModelForCausalLM, AutoTokenizer
from . import activate_local_venv, get_hf_token, load_project_env


activate_local_venv()
load_project_env()

def load_model_and_tokenizer(model_name: str, device: str = "cuda",
                             load_in_4bit: bool = False, load_in_8bit: bool = False,
                             dtype: Optional[str] = None):
    lower_name = model_name.lower()
    if lower_name.endswith(".gguf") or "gguf" in lower_name:
        raise ValueError(
            f"Model '{model_name}' appears to be a GGUF/llama.cpp checkpoint. "
            "These weights cannot be loaded with transformers; please choose a PyTorch- or safetensors-based repo."
        )

    kwargs = dict()
    if load_in_4bit:
        kwargs["load_in_4bit"] = True
    if load_in_8bit:
        kwargs["load_in_8bit"] = True
    if dtype:
        if dtype.lower() == "float16" or dtype.lower() == "fp16":
            kwargs["torch_dtype"] = torch.float16
        elif dtype.lower() == "bfloat16" or dtype.lower() == "bf16":
            kwargs["torch_dtype"] = torch.bfloat16

    hf_token = get_hf_token()
    token_kwargs: Dict[str, Any] = {}
    if hf_token:
        token_kwargs["token"] = hf_token

    try:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, **token_kwargs)
    except ValueError as err:
        message = str(err)
        needs_slow = "SentencePiece" in message or "tokenizer.model" in message
        if not needs_slow:
            raise
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=False, **token_kwargs)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model_kwargs: Dict[str, Any] = dict(device_map="auto", **kwargs)
    if hf_token:
        model_kwargs["token"] = hf_token

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    except (OSError, ValueError) as err:
        raise ValueError(
            f"Failed to load model '{model_name}' with transformers. "
            "Ensure this repo provides PyTorch or safetensors weights compatible with `transformers`."
        ) from err

    model.eval()
    return model, tok

def find_subtoken_span(tok, full_ids, sub_ids):
    # Return the start, end indices (inclusive-exclusive) of sub_ids inside full_ids
    n, m = len(full_ids), len(sub_ids)
    for i in range(n - m + 1):
        if full_ids[i:i+m] == sub_ids:
            return i, i+m
    return None

def evenly_spaced_layers(n_layers: int, fractions: List[float]) -> List[int]:
    lay = []
    for f in fractions:
        idx = max(0, min(n_layers-1, int(round(f * (n_layers-1)))))
        lay.append(idx)
    # dedupe while preserving order
    seen = set(); out = []
    for x in lay:
        if x not in seen:
            out.append(x); seen.add(x)
    return out
