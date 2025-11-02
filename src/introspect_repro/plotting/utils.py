
import os, re, json
from glob import glob

LAYER_RE = re.compile(r"layer(\d+)")
STRENGTH_RE = re.compile(r"strength([0-9.]+)")

def _extract_layer_strength(fname: str):
    layer = None; strength = None
    m = LAYER_RE.search(fname)
    if m: layer = int(m.group(1))
    m = STRENGTH_RE.search(fname)
    if m:
        try:
            strength = float(m.group(1))
        except:
            strength = None
    return layer, strength

def load_results(dirpath: str):
    files = sorted(glob(os.path.join(dirpath, "*.json")))
    out = []
    for f in files:
        try:
            with open(f, "r") as fh:
                j = json.load(fh)
            layer, strength = _extract_layer_strength(os.path.basename(f))
            if "layer" in j: layer = j["layer"]
            if "strength" in j and strength is None: strength = j["strength"]
            out.append((layer, strength, j, f))
        except Exception as e:
            print(f"Skipping {f}: {e}")
    return out
