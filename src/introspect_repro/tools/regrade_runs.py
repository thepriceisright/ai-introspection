"""Re-run judge grading on existing experiment outputs."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from ..prompts import INJECTED_THOUGHTS_BASE, THOUGHT_VS_TEXT_Q, JudgeConfig
from ..judges import Judge

INJECTED_QUESTION = (
    "Trial 1: Do you detect an injected thought? If so, what is the injected thought about?"
)


def _regrade_injected(run_dir: Path, judge: Judge, dry_run: bool = False) -> None:
    files = sorted(run_dir.glob("layer*_strength*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files found in {run_dir}")

    print(f"Regrading {len(files)} injected-thought files in {run_dir}")
    start = time.time()
    for path in files:
        data = json.loads(path.read_text())
        changed = False
        for trial in data.get("trials", []):
            response = trial.get("response") or ""
            if not response.strip():
                # Nothing to grade
                for key in ("coherence", "affirmative", "correct_identification"):
                    if trial.get(key) is not None:
                        trial[key] = None
                        changed = True
                continue

            new_coh = judge.grade_coherence(INJECTED_THOUGHTS_BASE, response)
            new_aff = judge.grade_affirmative(INJECTED_THOUGHTS_BASE, response)
            new_cor = judge.grade_affirmative_correct_id(
                INJECTED_QUESTION, response, trial.get("word", "")
            )

            if trial.get("coherence") != new_coh:
                trial["coherence"] = new_coh
                changed = True
            if trial.get("affirmative") != new_aff:
                trial["affirmative"] = new_aff
                changed = True
            if trial.get("correct_identification") != new_cor:
                trial["correct_identification"] = new_cor
                changed = True

        if changed and not dry_run:
            path.write_text(json.dumps(data, indent=2))
            print(f"  updated {path.name}")

    duration = time.time() - start
    print(f"Finished regrading in {duration:.1f}s")


def _regrade_thought_vs_text(run_dir: Path, judge: Judge, dry_run: bool = False) -> None:
    files = sorted(run_dir.glob("layer*_strength*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files found in {run_dir}")

    print(f"Regrading {len(files)} thought-vs-text files in {run_dir}")
    start = time.time()
    for path in files:
        data = json.loads(path.read_text())
        changed = False
        for trial in data.get("trials", []):
            sentence = trial.get("sentence", "")
            response = trial.get("think_resp") or ""
            word = trial.get("word", "")
            if not response.strip():
                if trial.get("think_about") is not None:
                    trial["think_about"] = None
                    changed = True
                continue
            q_prompt = THOUGHT_VS_TEXT_Q.format(sentence=sentence)
            new_value = judge.grade_thinking_about(q_prompt, response, word)
            if trial.get("think_about") != new_value:
                trial["think_about"] = new_value
                changed = True

        if changed and not dry_run:
            path.write_text(json.dumps(data, indent=2))
            print(f"  updated {path.name}")

    duration = time.time() - start
    print(f"Finished regrading in {duration:.1f}s")


def _regrade_prefill(run_dir: Path, judge: Judge, dry_run: bool = False) -> None:
    files = sorted(run_dir.glob("layer*_strength*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files found in {run_dir}")

    print(f"Regrading {len(files)} prefill-intention files in {run_dir}")
    start = time.time()
    for path in files:
        data = json.loads(path.read_text())
        changed = False
        for trial in data.get("trials", []):
            response = trial.get("response") or ""
            word = trial.get("word", "")
            if not response.strip():
                if trial.get("intended") is not None:
                    trial["intended"] = None
                    changed = True
                continue
            new_value = judge.grade_intent(response, word)
            if trial.get("intended") != new_value:
                trial["intended"] = new_value
                changed = True

        if changed and not dry_run:
            path.write_text(json.dumps(data, indent=2))
            print(f"  updated {path.name}")

    duration = time.time() - start
    print(f"Finished regrading in {duration:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-run judge grading on existing runs.")
    parser.add_argument(
        "--task",
        choices=["injected_thoughts", "thought_vs_text", "prefill_intention"],
        required=True,
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--judge-provider", default="openai")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-max-tokens", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true", help="Only report changes")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    cfg = JudgeConfig(
        provider=args.judge_provider,
        model=args.judge_model,
        temperature=args.judge_temperature,
        max_tokens=args.judge_max_tokens,
    )
    judge = Judge(cfg)

    if args.task == "injected_thoughts":
        _regrade_injected(run_dir, judge, dry_run=args.dry_run)
    elif args.task == "thought_vs_text":
        _regrade_thought_vs_text(run_dir, judge, dry_run=args.dry_run)
    elif args.task == "prefill_intention":
        _regrade_prefill(run_dir, judge, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
