
import argparse, numpy as np
import matplotlib.pyplot as plt
from .utils import load_results

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="Path to runs/<ts>/prefill_intention")
    p.add_argument("--strength", type=float, default=None)
    p.add_argument("--save", default=None)
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    results = load_results(args.run_dir)
    layers = sorted(set([layer for (layer, strength, j, f) in results if layer is not None and (args.strength is None or strength==args.strength)]))
    xs = []; apology_rates = []
    for L in layers:
        files = [(layer,strength,j,f) for (layer,strength,j,f) in results if layer==L and (args.strength is None or strength==args.strength)]
        if not files:
            continue
        intended = []
        for (_, strength, j, f) in files:
            intended.extend([1 if t.get("intended") else 0 for t in j["trials"]])
        xs.append(L)
        # "apology rate" = 1 - intended rate (paper p.24)
        ar = 1.0 - (np.mean(intended) if intended else 0.0)
        apology_rates.append(ar)

    plt.figure()
    plt.plot(xs, apology_rates, label="Apology rate (lower is better)")
    plt.xlabel("Layer")
    plt.ylabel("Apology Rate")
    title = "Prefill Intention – Layer-wise"
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
