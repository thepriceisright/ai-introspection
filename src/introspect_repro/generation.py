
from typing import Optional, Tuple
import torch
from .activation_utils import ResidualInjector

def generate_with_optional_injection(model, tok, prompt: str, layer_idx: Optional[int] = None,
                                     vec=None, strength: float = 0.0,
                                     token_range: Optional[Tuple[int,int]] = None,
                                     max_new_tokens: int = 128, temperature: float = 0.0):
    device = next(model.parameters()).device
    enc = tok(prompt, return_tensors="pt").to(device)
    inj_ctx = (ResidualInjector(model, layer_idx, vec, strength, token_range)
               if (layer_idx is not None and vec is not None and strength != 0.0)
               else nullcontext())
    with inj_ctx:
        out = model.generate(**enc, do_sample=(temperature>0.0),
                             temperature=max(temperature,1e-6),
                             max_new_tokens=max_new_tokens,
                             pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
    text = tok.decode(out[0], skip_special_tokens=True)
    # return only the assistant continuation after the last "Assistant:" if present
    if "Assistant:" in prompt:
        split_on = text.rsplit("Assistant:", 1)
        return split_on[-1].strip()
    else:
        return text[len(prompt):].strip()

from contextlib import contextmanager

@contextmanager
def nullcontext():
    yield
