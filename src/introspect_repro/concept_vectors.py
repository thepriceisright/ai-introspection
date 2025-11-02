
from typing import List, Dict, Optional, Tuple
import torch, numpy as np
from .models import load_model_and_tokenizer
from .activation_utils import ResidualCapture
from .prompts import INJECTED_THOUGHTS_BASE
from .word_lists import BASELINE_WORDS

def _prompt_for_concept(word: str) -> str:
    # Appendix p.36: record on final ":" in
    # "Human: Tell me about {word}\nAssistant:"
    return f"Human: Tell me about {word.lower()}\nAssistant:"

def _token_index_of_last_colon(tok, text: str) -> int:
    # token position of last ":" in the prompt (the "Assistant:" colon)
    ids = tok.encode(text, add_special_tokens=False)
    colon_id = tok.encode(":", add_special_tokens=False)[-1]
    # find last occurrence
    for i in range(len(ids)-1, -1, -1):
        if ids[i] == colon_id:
            return i
    return len(ids)-1

def get_residual_at_token(model, tok, text: str, layer_idx: int, token_index: int) -> torch.Tensor:
    device = next(model.parameters()).device
    ids = tok(text, return_tensors="pt").to(device)
    with ResidualCapture(model, layer_idx, capture_output=False) as cap:
        _ = model(**ids, use_cache=False)
    hs = cap.buffer  # [B, T, D]
    return hs[0, token_index, :].detach().cpu()

def compute_baseline_mean(model, tok, layer_idx: int) -> torch.Tensor:
    vecs = []
    for w in BASELINE_WORDS:
        text = _prompt_for_concept(w)
        idx = _token_index_of_last_colon(tok, text)
        v = get_residual_at_token(model, tok, text, layer_idx, idx)
        vecs.append(v)
    return torch.stack(vecs, dim=0).mean(dim=0)

def compute_concept_vector(model, tok, word: str, layer_idx: int,
                           cached_baseline: Optional[torch.Tensor] = None) -> torch.Tensor:
    text = _prompt_for_concept(word)
    idx = _token_index_of_last_colon(tok, text)
    v = get_residual_at_token(model, tok, text, layer_idx, idx)
    if cached_baseline is None:
        return v
    return v - cached_baseline

