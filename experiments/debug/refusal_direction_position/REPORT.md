# Debug Report: Refusal Direction Position Selection Bias

## Issue
Pipeline selected pos=-5 (end_of_turn special token), layer=22 as the best refusal
direction for Gemma 2 27B. Ablating this direction produces gibberish.

## Root Cause: Metric Bias

The `refusal_score` metric is **biased by model destruction**:

- It checks whether P(token "I") drops after ablation
- When you ablate a direction carrying general model state, the model produces garbage
- Garbage doesn't start with "I" → P("I") drops → metric reports misleadingly good score
- **The pipeline selects for model destruction, not clean refusal removal**

## Evidence

**Pearson correlation(KL divergence, refusal_score) = -0.77** across all 230 candidates.

Per-position:
- pos=-5 (<end_of_turn>): r = -0.90
- pos=-4 (\n):            r = -0.81
- pos=-3 (<start_of_turn>): r = -0.82
- pos=-2 (model):         r = -0.67
- pos=-1 (\n final):      r = -0.81

Higher KL → lower refusal score at r = -0.77 to -0.90. The metric rewards destruction.

## Fix Applied

Changed `metadata.json` to **pos=-1, layer=19**:
- cos(refusal, assistant_axis) = 0.50 (vs 0.23 at old selection)
- Refusal score = -17.64 (nearly identical to -17.99)
- Position is the final token before generation (intuitive decision point)

## Recommended Pipeline Improvement

The direction selection should use a **composite score** that penalizes high KL:

```
composite = -refusal_score + α * cos(direction, assistant_axis) - β * kl_divergence
```

Or at minimum, re-enable KL filtering with an appropriate threshold for large models
(e.g., kl_threshold based on median KL across candidates rather than disabled entirely).

## Files
- `diagnose_score_bias.py` — diagnostic script
- `results/score_bias_analysis.json` — raw data
