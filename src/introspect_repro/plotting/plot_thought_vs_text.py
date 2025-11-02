
import argparse, os, numpy as np
import matplotlib.pyplot as plt
from .utils import load_results

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="Path to runs/<ts>/thought_vs_text")
    p.add_argument("--strength", type=float, default=None)
    p.add_argument("--save", default=None)
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    results = load_results(args.run_dir)
    layers = sorted(set([layer for (layer, strength, j, f) in results if layer is not None and (args.strength is None or strength==args.strength)]))
    xs = []; think_rates = []; repeat_rates = []
    for L in layers:
        # gather all files for this layer & strength
        files = [(layer,strength,j,f) for (layer,strength,j,f) in results if layer==L and (args.strength is None or strength==args.strength)]
        if not files:
            continue
        think = []; rep = []
        for (_, strength, j, f) in files:
            think.extend([1 if t.get("think_about") else 0 for t in j["trials"]])
            rep.extend([1 if t.get("exact_repeat") else 0 for t in j["trials"]])
        xs.append(L)
        think_rates.append(np.mean(think) if think else 0.0)
        repeat_rates.append(np.mean(rep) if rep else 0.0)

    plt.figure()
    plt.plot(xs, think_rates, label="Identify injected thought (judge yes)")
    plt.plot(xs, repeat_rates, label="Repeat sentence exactly")
    plt.xlabel("Layer")
    plt.ylabel("Rate")
    title = "Thought vs Text – Layer-wise"
    if args.strength is not None:
        title += f" (strength {args.strength})"
    plt.title(title)
    plt.legend()
    if args.save:
        plt.savefig(args.save, bbox_inches="tight", dpi=160)
    if args.show:
        plt.show()

if __name__ == "__main__":
    main()
