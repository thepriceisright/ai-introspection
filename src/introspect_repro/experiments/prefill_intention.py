
import argparse, json, os, time, random
from ..models import load_model_and_tokenizer
from ..prompts import PREFILL_PROMPT, JudgeConfig
from ..judges import Judge
from ..word_lists import CONCEPT_WORDS, SENTENCES
from ..concept_vectors import compute_baseline_mean, compute_concept_vector
from ..generation import generate_with_optional_injection

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--strength", type=float, default=4.0)
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
    ts = time.strftime("%Y%m%d_%H%M%S")
    outdir = args.outdir or os.path.join("runs", ts, "prefill_intention")
    os.makedirs(outdir, exist_ok=True)
    judge = Judge(JudgeConfig(provider=args.judge_provider, model=args.judge_model))
    baseline = compute_baseline_mean(model, tok, args.layer)

    trials = []
    for i in range(args.n_trials):
        w = random.choice(CONCEPT_WORDS)
        s = random.choice(SENTENCES)
        prompt = PREFILL_PROMPT.format(sentence=s, word=w.lower())
        vec = compute_concept_vector(model, tok, w, args.layer, cached_baseline=baseline)
        # Inject concept on the sentence tokens (prior to the prefill); approximate by injecting on entire prompt span before prefill
        response = generate_with_optional_injection(model, tok, prompt, args.layer, vec, args.strength,
                                                    token_range=None, max_new_tokens=64, temperature=0.0)
        intended = judge.grade_intent(response, w)
        trials.append(dict(word=w, sentence=s, response=response, intended=intended))

    with open(os.path.join(outdir, f"layer{args.layer}_strength{args.strength}.json"), "w") as f:
        json.dump(dict(layer=args.layer, strength=args.strength, trials=trials), f, indent=2)
    print(f"Saved {len(trials)} trials to {outdir}")

if __name__ == "__main__":
    main()
