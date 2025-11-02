
import argparse, json, os, time, random
import torch, numpy as np
from ..models import load_model_and_tokenizer
from ..prompts import THINK_PROMPT, DONT_THINK_PROMPT
from ..word_lists import CONCEPT_WORDS, SENTENCES
from ..concept_vectors import compute_baseline_mean, compute_concept_vector
from ..generation import generate_with_optional_injection
from ..activation_utils import ResidualCapture

def cosine(a, b):
    a = a / (a.norm() + 1e-8); b = b / (b.norm() + 1e-8)
    return (a * b).sum().item()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--layer", type=int, required=True, help="layer to *measure* activation similarity")
    p.add_argument("--n-trials", type=int, default=16)
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--load-in-8bit", action="store_true")
    p.add_argument("--dtype", default=None)
    p.add_argument("--outdir", default=None)
    args = p.parse_args()

    model, tok = load_model_and_tokenizer(args.model, device=args.device,
                                          load_in_4bit=args.load_in_4bit, load_in_8bit=args.load_in_8bit,
                                          dtype=args.dtype)
    ts = time.strftime("%Y%m%d_%H%M%S")
    outdir = args.outdir or os.path.join("runs", ts, "intentional_control")
    os.makedirs(outdir, exist_ok=True)

    baseline = compute_baseline_mean(model, tok, args.layer)

    trials = []
    for i in range(args.n_trials):
        w = random.choice(CONCEPT_WORDS)
        s = random.choice(SENTENCES)
        vec = compute_concept_vector(model, tok, w, args.layer, cached_baseline=baseline)
        # THINK
        prompt_think = THINK_PROMPT.format(sentence=s, word=w.lower())
        ids_think = tok(prompt_think, return_tensors="pt").to(next(model.parameters()).device)

        with ResidualCapture(model, args.layer, capture_output=True) as cap:
            out = model.generate(**ids_think, do_sample=False, max_new_tokens= len(tok(s).input_ids)+5,
                                 pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
        hs = cap.buffer[0]  # [T, D]
        # compute cosine across token positions (assistant output tokens appended)
        sims_think = [cosine(hs[t], vec.to(hs.device)) for t in range(hs.shape[0])]

        # DONT THINK
        prompt_dont = DONT_THINK_PROMPT.format(sentence=s, word=w.lower())
        ids_dont = tok(prompt_dont, return_tensors="pt").to(next(model.parameters()).device)
        with ResidualCapture(model, args.layer, capture_output=True) as cap2:
            out2 = model.generate(**ids_dont, do_sample=False, max_new_tokens=len(tok(s).input_ids)+5,
                                  pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
        hs2 = cap2.buffer[0]
        sims_dont = [cosine(hs2[t], vec.to(hs2.device)) for t in range(hs2.shape[0])]

        trials.append(dict(word=w, sentence=s, sims_think=sims_think, sims_dont=sims_dont))

    with open(os.path.join(outdir, f"layer{args.layer}.json"), "w") as f:
        json.dump(dict(layer=args.layer, trials=trials), f, indent=2)
    print(f"Saved {len(trials)} trials to {outdir}")

if __name__ == "__main__":
    main()
