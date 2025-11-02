

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
