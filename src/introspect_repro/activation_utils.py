
from typing import Optional, Tuple, Callable, Dict, Any, List
import torch

class ResidualCapture:
    """
    Captures residual stream at a given decoder layer index.
    Works with HF decoder layers where the forward signature is (hidden_states, ...).
    """
    def __init__(self, model, layer_idx: int, capture_output: bool = False):
        self.model = model
        self.layer_idx = layer_idx
        self.capture_output = capture_output
        self.buffer = None
        self.hook = None

    def _hook_pre(self, module, inputs):
        # inputs[0] is hidden_states [B, T, D]
        self.buffer = inputs[0].detach().clone()

    def _hook_post(self, module, inputs, outputs):
        if isinstance(outputs, tuple):
            hs = outputs[0]
        else:
            hs = outputs
        self.buffer = hs.detach().clone()

    def __enter__(self):
        layer = self.model.model.layers[self.layer_idx]
        if self.capture_output:
            self.hook = layer.register_forward_hook(self._hook_post, with_kwargs=False)
        else:
            self.hook = layer.register_forward_pre_hook(self._hook_pre, with_kwargs=False)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.hook is not None:
            self.hook.remove()
            self.hook = None

class ResidualInjector:
    """
    Adds a vector (broadcast across batch, restricted to token span) into the residual stream
    at a chosen layer with a scalar strength.
    """
    def __init__(self, model, layer_idx: int, vec: torch.Tensor, strength: float,
                 token_range: Optional[Tuple[int,int]] = None):
        self.model = model
        self.layer_idx = layer_idx
        self.vec = vec  # [D]
        self.strength = strength
        self.token_range = token_range
        self.hook = None

    def _hook_pre(self, module, inputs):
        hs = inputs[0]
        start, end = (0, hs.shape[1]) if self.token_range is None else self.token_range
        add = self.vec.to(hs.device) * self.strength
        hs[:, start:end, :] = hs[:, start:end, :] + add
        return (hs,)+tuple(inputs[1:])  # return modified inputs

    def __enter__(self):
        layer = self.model.model.layers[self.layer_idx]
        self.hook = layer.register_forward_pre_hook(self._hook_pre, with_kwargs=False)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.hook is not None:
            self.hook.remove()
            self.hook = None
