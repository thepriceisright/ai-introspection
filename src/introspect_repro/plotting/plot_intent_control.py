
import argparse, os, json, numpy as np
import matplotlib.pyplot as plt
from glob import glob
from .utils import load_results, _extract_layer_strength

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="Path to runs/<ts>/intentional_control")
    p.add_argument("--save", default=None)
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    files = sorted(glob(os.path.join(args.run_dir, "layer*.json")))
    layers = []
    think_means = []
    dont_means = []
    for f in files:
        with open(f, "r") as fh:
            j = json.load(fh)
        layer, _ = _extract_layer_strength(os.path.basename(f))
        # average cosine across tokens and trials
        sims_think = []
        sims_dont = []
        for t in j["trials"]:
            if "sims_think" in t:
                sims_think.extend(t["sims_think"])
            if "sims_dont" in t:
                sims_dont.extend(t["sims_dont"])
        if sims_think and sims_dont:
            layers.append(layer)
            think_means.append(np.mean(sims_think))
            dont_means.append(np.mean(sims_dont))

    # sort by layer
    pairs = sorted(zip(layers, think_means, dont_means), key=lambda x: x[0])
    xs = [p[0] for p in pairs]
    tmeans = [p[1] for p in pairs]
    dmeans = [p[2] for p in pairs]

    plt.figure()
    plt.plot(xs, tmeans, label="Think (mean cosine)")
    plt.plot(xs, dmeans, label="Don't think (mean cosine)")
    plt.xlabel("Layer")
    plt.ylabel("Cosine similarity")
    plt.title("Intentional control – Layer-wise mean cosine")
    plt.legend()
    if args.save:
        plt.savefig(args.save, bbox_inches="tight", dpi=160)
    if args.show:
        plt.show()

if __name__ == "__main__":
    main()
