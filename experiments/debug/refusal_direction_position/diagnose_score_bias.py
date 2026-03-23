"""Diagnostic: Is the refusal_score metric biased by model destruction?

Hypothesis: When ablating a direction that carries general model state (not just
refusal), the model produces garbage that happens to not start with 'I', giving
a misleadingly low (good) refusal score. This would explain why pos=-5
(end_of_turn special token) scores better than pos=-1 (actual decision point).

Test: For each (pos, layer) candidate, check:
1. The ablation refusal score (from pipeline evaluations)
2. The KL divergence (model damage indicator)
3. Cosine similarity with assistant axis (direction quality)
4. Whether high KL correlates with misleadingly good refusal scores

No GPU needed — uses saved artifacts only.
"""

import json
import torch
import numpy as np
from pathlib import Path

VECTORS_DIR = Path("vectors/gemma-2-27b-it")
RUNS_DIR = Path("pipelines/refusal_direction/runs/gemma-2-27b-it")
OUTPUT_DIR = Path("experiments/debug/refusal_direction_position/results")

# Load data
md = torch.load(VECTORS_DIR / "refusal_mean_diffs.pt", map_location="cpu", weights_only=True)
aa_data = torch.load(VECTORS_DIR / "assistant_axis.pt", map_location="cpu", weights_only=True)
aa = aa_data.float() if not isinstance(aa_data, dict) else aa_data["axis"].float()

with open(RUNS_DIR / "select_direction/direction_evaluations.json") as f:
    all_evals = json.load(f)

with open(RUNS_DIR / "select_direction/direction_evaluations_filtered.json") as f:
    filtered = json.load(f)

# Token names for each position
pos_tokens = {-5: "<end_of_turn>", -4: "\\n", -3: "<start_of_turn>", -2: "model", -1: "\\n (final)"}

print("=" * 90)
print("DIAGNOSTIC: Refusal Score Bias from Model Destruction")
print("=" * 90)

# Build lookup for all evaluations
eval_lookup = {}
for e in all_evals:
    eval_lookup[(e["position"], e["layer"])] = e

# For each position, find the best layer and report all metrics
print(f"\n{'pos':>4} {'token':>16} {'layer':>5} {'refusal':>10} {'steering':>10} {'kl_div':>10} {'cos(r,aa)':>10} {'FLAG':>10}")
print("-" * 90)

results = []
for pos_idx in range(5):
    pos = pos_idx - 5
    token_name = pos_tokens[pos]

    # Find the best layer for this position (lowest refusal score in filtered set)
    pos_candidates = [e for e in filtered if e["position"] == pos]
    if not pos_candidates:
        # Not in filtered set — check unfiltered
        pos_all = [e for e in all_evals if e["position"] == pos]
        if pos_all:
            best = min(pos_all, key=lambda x: x["refusal_score"])
            flag = "FILTERED_OUT"
        else:
            continue
    else:
        best = pos_candidates[0]
        flag = ""

    layer = best["layer"]
    r = md[pos_idx, layer].float()
    cos = float(torch.nn.functional.cosine_similarity(r.unsqueeze(0), aa[layer].unsqueeze(0)))

    kl = best["kl_div_score"]
    ref_score = best["refusal_score"]
    steer = best["steering_score"]

    # Flag: high KL + low refusal score = suspicious (model destruction, not clean ablation)
    if kl > 5 and ref_score < -15:
        flag = "SUSPICIOUS"

    print(f"{pos:>4} {token_name:>16} {layer:>5} {ref_score:>10.2f} {steer:>10.2f} {kl:>10.4f} {cos:>10.4f} {flag:>10}")

    results.append({
        "position": pos, "token": token_name, "layer": layer,
        "refusal_score": ref_score, "steering_score": steer,
        "kl_div": kl, "cos_assistant_axis": cos, "flag": flag,
    })

# Correlation analysis
print("\n" + "=" * 90)
print("CORRELATION: Does high KL predict misleadingly good refusal scores?")
print("=" * 90)

# All evaluations
kls = [e["kl_div_score"] for e in all_evals if not np.isnan(e["kl_div_score"])]
refs = [e["refusal_score"] for e in all_evals if not np.isnan(e["kl_div_score"])]
corr = np.corrcoef(kls, refs)[0, 1]
print(f"\nPearson correlation(KL, refusal_score) across all {len(kls)} candidates: r = {corr:.4f}")
print("  Negative r means higher KL → lower (better) refusal score")
print("  This would confirm the bias: destroying the model gives fake 'good' scores")

# Separate by position
print(f"\nPer-position breakdown:")
for pos_idx in range(5):
    pos = pos_idx - 5
    pos_evals = [e for e in all_evals if e["position"] == pos and not np.isnan(e["kl_div_score"])]
    if len(pos_evals) > 2:
        kls_p = [e["kl_div_score"] for e in pos_evals]
        refs_p = [e["refusal_score"] for e in pos_evals]
        corr_p = np.corrcoef(kls_p, refs_p)[0, 1]
        print(f"  pos={pos} ({pos_tokens[pos]:>16}): r={corr_p:.4f} (n={len(pos_evals)})")

# Save results
with open(OUTPUT_DIR / "score_bias_analysis.json", "w") as f:
    json.dump({
        "per_position_best": results,
        "correlation_all": float(corr),
        "conclusion": "PENDING — check if correlation is significantly negative",
    }, f, indent=2)

print(f"\nResults saved to {OUTPUT_DIR / 'score_bias_analysis.json'}")
