
import argparse, json, os, time, random
import torch, numpy as np
from tqdm import trange
from ..models import load_model_and_tokenizer, evenly_spaced_layers
from ..prompts import INJECTED_THOUGHTS_BASE, JudgeConfig
from ..judges import Judge
from ..word_lists import CONCEPT_WORDS
from ..concept_vectors import compute_baseline_mean, compute_concept_vector
from ..generation import generate_with_optional_injection

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--layer", type=int, default=None)
    p.add_argument("--sweep-layers", nargs="*", type=float, help="fractions (0..1) of depth to probe")
    p.add_argument("--strength", type=float, default=None)
    p.add_argument("--strengths", nargs="*", type=float, default=None)
    p.add_argument("--n-trials", type=int, default=50)
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--load-in-8bit", action="store_true")
    p.add_argument("--dtype", default=None)
    p.add_argument("--judge-provider", default="openai")
    p.add_argument("--judge-model", default="gpt-4o-mini")
    p.add_argument("--outdir", default=None)
    args = p.parse_args()

    model, tok = load_model_and_tokenizer(args.model, device=args.device,
                                          load_in_4bit=args.load_in_4bit, load_in_8bit=args.load_in_8bit,
                                          dtype=args.dtype)
    n_layers = len(model.model.layers)

    layers = [args.layer] if args.layer is not None else evenly_spaced_layers(n_layers, args.sweep_layers or [0.6,0.66,0.7])
    strengths = [args.strength] if args.strength is not None else (args.strengths or [1,2,4,8])

    ts = time.strftime("%Y%m%d_%H%M%S")
    outdir = args.outdir or os.path.join("runs", ts, "injected_thoughts")
    os.makedirs(outdir, exist_ok=True)

    judge = Judge(JudgeConfig(provider=args.judge_provider, model=args.judge_model))

    for layer in layers:
        # concept vectors need baseline mean (paper Appendix p.36)
        baseline = compute_baseline_mean(model, tok, layer)
        # run trials
        words = random.sample(CONCEPT_WORDS, k=min(args.n_trials, len(CONCEPT_WORDS)))
        for strength in strengths:
            results = []
            for w in words:
                prompt = INJECTED_THOUGHTS_BASE
                response = generate_with_optional_injection(
                    model, tok, prompt, layer_idx=layer,
                    vec=compute_concept_vector(model, tok, w, layer, cached_baseline=baseline),
                    strength=strength, token_range=None, # inject from newline before "Trial 1" onward (approximate with whole turn)
                    max_new_tokens=128, temperature=0.0
                )
                # grading (pp. 39–42): coherence + affirmative + correct identification
                coh = judge.grade_coherence(prompt, response)
                aff = judge.grade_affirmative(prompt, response)
                cor = judge.grade_affirmative_correct_id("Trial 1: Do you detect an injected thought? If so, what is the injected thought about?",
                                                         response, w)
                results.append(dict(word=w, response=response, coherence=coh, affirmative=aff, correct_identification=cor))
            with open(os.path.join(outdir, f"layer{layer}_strength{strength}.json"), "w") as f:
                json.dump(dict(layer=layer, strength=strength, trials=results), f, indent=2)
            print(f"[layer {layer}] strength {strength} -> saved {len(results)} trials to {outdir}")

if __name__ == "__main__":
    main()
