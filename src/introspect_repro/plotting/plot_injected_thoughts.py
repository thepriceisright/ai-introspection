
import argparse, os, numpy as np
import matplotlib.pyplot as plt
from .utils import load_results

def rate(xs):
    if not xs: return 0.0
    return sum(1 for x in xs if x) / float(len(xs))

def compute_metrics(trials):
    """
    Returns a dict of metrics computed from a trials list.
    We expect each trial to contain: response (str), coherence (bool/None), affirmative (bool/None), correct_identification (bool/None)
    Optional: injection (bool) for control trials.
    """
    coh = [t.get("coherence") for t in trials if t.get("coherence") is not None]
    aff = [t.get("affirmative") for t in trials if t.get("affirmative") is not None]
    cor = [t.get("correct_identification") for t in trials if t.get("correct_identification") is not None]
    # Introspective awareness per paper: coherence & affirmative & correct identification (grader enforces "detect before saying")
    aware = [ (t.get("coherence") and t.get("affirmative") and t.get("correct_identification")) for t in trials ]
    # naive "mentions word" check
    mword = []
    for t in trials:
        w = (t.get("word") or "").lower()
        resp = (t.get("response") or "").lower()
        mword.append(w and (w in resp))
    # false positive: control trials with affirmative True
    ctrls = [t for t in trials if t.get("injection") is False]
    ctrl_aff = [t.get("affirmative") for t in ctrls if t.get("affirmative") is not None]

    return dict(
        n=len(trials),
        coherence=rate(coh),
        affirmative=rate(aff),
        correct_id=rate(cor),
        aware=rate(aware),
        mention=rate(mword),
        false_positive=rate(ctrl_aff) if ctrl_aff else 0.0
    )

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="Path to a single run dir, e.g. runs/<ts>/injected_thoughts")
    p.add_argument("--strength", type=float, default=None, help="If given, restrict to this strength")
    p.add_argument("--save", default=None, help="Path to save PNG")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    results = load_results(args.run_dir)
    # aggregate by layer, optionally filtered by strength
    layers = sorted(set([layer for (layer, strength, j, f) in results if layer is not None and (args.strength is None or strength==args.strength)]))
    metrics_by_layer = {L: [] for L in layers}
    for (layer, strength, j, f) in results:
        if layer is None: continue
        if args.strength is not None and strength != args.strength: continue
        m = compute_metrics(j["trials"])
        metrics_by_layer[layer].append(m)

    # average across files per layer
    xs = []; aware = []; affirmative = []; mention = []; false_pos = []
    for L in sorted(metrics_by_layer.keys()):
        xs.append(L)
        arr = metrics_by_layer[L]
        if not arr: 
            aware.append(0.0); affirmative.append(0.0); mention.append(0.0); false_pos.append(0.0)
        else:
            aware.append(np.mean([a["aware"] for a in arr]))
            affirmative.append(np.mean([a["affirmative"] for a in arr]))
            mention.append(np.mean([a["mention"] for a in arr]))
            false_pos.append(np.mean([a["false_positive"] for a in arr]))

    plt.figure()
    plt.plot(xs, aware, label="Introspective awareness (coh ∧ yes ∧ correct id)")
    plt.plot(xs, affirmative, label="Affirmative (yes)")
    plt.plot(xs, mention, label="Mentions injected word")
    if any(false_pos):
        plt.plot(xs, false_pos, label="False positive rate (control)")
    plt.xlabel("Layer")
    plt.ylabel("Rate")
    if args.strength is not None:
        plt.title(f"Injected thoughts – Layer-wise at strength {args.strength}")
    else:
        plt.title("Injected thoughts – Layer-wise")
    plt.legend()
    if args.save:
        plt.savefig(args.save, bbox_inches="tight", dpi=160)
    if args.show:
        plt.show()

if __name__ == "__main__":
    main()
