"""Verify: does median KL threshold fix the selection for Gemma 2 27B?

Simulates the paper's method with median KL threshold on saved data.
"""
import json, math
import torch
import numpy as np
from pathlib import Path

RUNS_DIR = Path("pipelines/refusal_direction/runs/gemma-2-27b-it")

with open(RUNS_DIR / "select_direction/direction_evaluations.json") as f:
    all_evals = json.load(f)

pos_tokens = {-5: "<end_of_turn>", -4: "\\n", -3: "<start_of_turn>", -2: "model", -1: "\\n (final)"}
n_layer = 46

# Compute median KL
all_kls = [e["kl_div_score"] for e in all_evals if not math.isnan(e["kl_div_score"])]
median_kl = np.median(all_kls)
print(f"KL distribution: min={min(all_kls):.2f}, median={median_kl:.2f}, max={max(all_kls):.2f}")

# Run selection with median KL threshold (paper method)
candidates = []
for e in all_evals:
    pos, layer = e["position"], e["layer"]
    rs, ss, kl = e["refusal_score"], e["steering_score"], e["kl_div_score"]

    # Filter: NaN, last 20% layers, KL > median, steering < 0
    if math.isnan(rs) or math.isnan(ss) or math.isnan(kl):
        continue
    if layer >= int(n_layer * 0.8):
        continue
    if kl > median_kl:
        continue
    if ss < 0:
        continue

    candidates.append((rs, pos, layer, ss, kl))

candidates.sort(key=lambda x: x[0])  # ascending refusal score = best first

print(f"\nFiltered to {len(candidates)} candidates (from {len(all_evals)})")
print(f"\n{'Rank':>4} {'Pos':>4} {'Token':>16} {'Layer':>5} {'Refusal':>8} {'Steering':>8} {'KL':>8}")
print("-" * 70)
for i, (rs, pos, layer, ss, kl) in enumerate(candidates[:10]):
    print(f"  {i+1:>2}  {pos:>4} {pos_tokens.get(pos, '?'):>16} {layer:>5} {rs:>8.2f} {ss:>8.2f} {kl:>8.2f}")

winner = candidates[0]
print(f"\nWinner: pos={winner[1]}, layer={winner[2]} ({pos_tokens.get(winner[1], '?')})")
print(f"  Refusal: {winner[0]:.2f}, Steering: {winner[3]:.2f}, KL: {winner[4]:.2f}")
