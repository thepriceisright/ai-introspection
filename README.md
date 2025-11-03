

## Overview

This repository reproduces the key experiments for *Emergent Introspective Awareness in LLMs*.
Experiments live in Jupyter notebooks under the repo root, with reusable code in `src/introspect_repro/`.

## Quick Start

```bash
# 1) Create the local virtual environment the repo expects
python -m venv .venv
source .venv/bin/activate

# 2) Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3) Create a .env file with your API credentials (see below)
touch .env  # edit and add the keys you need
```

### Environment variables

The notebooks and Python modules automatically load environment variables from `.env`. Populate the file with the credentials you plan to use:

```bash
ANTHROPIC_API_KEY="sk-ant-..."
OPENAI_API_KEY="sk-proj-..."
OPENROUTER_API_KEY="sk-or-..."
HUGGINGFACEHUB_API_TOKEN="hf_..."
```

Any of the following keys can be used for the Hugging Face token: `HUGGINGFACEHUB_API_TOKEN`, `HUGGINGFACE_TOKEN`, `HF_TOKEN`, `HF_API_TOKEN`. The loader normalises whichever one you set so `HUGGINGFACEHUB_API_TOKEN` is always populated, and the token is passed to `transformers` when downloading gated checkpoints (e.g. Llama-3).

> **Tip:** If you add new keys to `.env`, rerun the “Load environment variables” cell in the notebooks so they are picked up by the kernel.

### Using the notebooks

Launch Jupyter (or VS Code) with the `.venv` interpreter selected. The first notebook cell primes the environment:

1. Activates `.venv` so local dependencies are in `sys.path`.
2. Loads `.env` variables while preserving any values already set in the shell.
3. Prints whether the Hugging Face token, Anthropic, OpenAI, and OpenRouter keys were detected.

> **Note:** The experiment runners expect `transformers`-compatible checkpoints (PyTorch/safetensors). GGUF/llama.cpp quantisations are not supported.

If the kernel is not using `.venv`, the cell prints a warning and still adds the `.venv` site-packages directory to the Python path.

### Regrading existing runs

If you change judge settings (e.g. swap models or tweak prompts) you can re-run the grading stage on already-generated trials without regenerating completions:

```bash
python -m src.introspect_repro.tools.regrade_runs \
  --task injected_thoughts \
  --run-dir runs/<timestamp>/injected_thoughts \
  --judge-provider openai \
  --judge-model gpt-5-mini \
  --judge-temperature 1.0
```

## Plotting layer-wise lines

The `plotting/` module recreates the layer-wise curves shown in the paper (e.g., Fig. on p.15 for injected thoughts; pp. 21, 24, 28–29 for other tasks). Save outputs from your runs in `runs/<timestamp>/...` and then call:

```bash
# Injected thoughts (choose a strength, e.g. 2)
python -m introspect_repro.plotting.plot_injected_thoughts       --run-dir runs/<timestamp>/injected_thoughts --strength 2 --save injected_thoughts_layerwise.png

# Thoughts vs text (plots both lines: "think-about" judge rate and "exact repeat")
python -m introspect_repro.plotting.plot_thought_vs_text       --run-dir runs/<timestamp>/thought_vs_text --strength 2 --save thought_vs_text_layerwise.png

# Prefill intention (apology rate vs layer)
python -m introspect_repro.plotting.plot_prefill_intention       --run-dir runs/<timestamp>/prefill_intention --strength 4 --save prefill_layerwise.png

# Intentional control (collects multiple files: runs/<ts>/intentional_control/layer*.json)
python -m introspect_repro.plotting.plot_intent_control       --run-dir runs/<timestamp>/intentional_control --save intent_control_layerwise.png
```

**Notes:** 
- For injected thoughts, if you include control (no-injection) trials in your JSON files (field `"injection": false`), the plot will show the **false positive** line.  
- Our *introspective awareness* line is computed as **coherent ∧ affirmative ∧ correct-identification**, matching the Appendix grader (pp. 39–42).  
- The *apology rate* is 1 minus the judge-rated “intended” rate, matching the definition on p.24.  
