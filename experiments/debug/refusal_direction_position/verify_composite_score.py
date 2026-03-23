"""Verify: does the composite score select pos=-1 over pos=-5?

Simulates the new selection logic on saved evaluation data.
No GPU needed.
"""
import json
import math
import torch
from pathlib import Path

RUNS_DIR = Path("pipelines/refusal_direction/runs/gemma-2-27b-it")
VECTORS_DIR = Path("vectors/gemma-2-27b-it")

# Load saved evaluations
with open(RUNS_DIR / "select_direction/direction_evaluations.json") as f:
    all_evals = json.load(f)

# Load assistant axis
aa_data = torch.load(VECTORS_DIR / "assistant_axis.pt", map_location="cpu", weights_only=True)
aa = aa_data.float() if not isinstance(aa_data, dict) else aa_data["axis"].float()

# Load mean_diffs for cosine sim computation
md = torch.load(VECTORS_DIR / "refusal_mean_diffs.pt", map_location="cpu", weights_only=True)

n_layer = 46
pos_tokens = {-5: "<end_of_turn>", -4: "\\n", -3: "<start_of_turn>", -2: "model", -1: "\\n (final)"}

def filter_fn(refusal_score, steering_score, kl_div_score, layer, n_layer,
              kl_threshold=None, prune_layer_percentage=0.20):
    if math.isnan(refusal_score) or math.isnan(steering_score) or math.isnan(kl_div_score):
        return True
    if prune_layer_percentage is not None and layer >= int(n_layer * (1.0 - prune_layer_percentage)):
        return True
    if kl_threshold is not None and kl_div_score > kl_threshold:
        return True
    return False

print("=" * 100)
print("OLD SCORING (pure -refusal_score) vs NEW SCORING (composite)")
print("=" * 100)

old_candidates = []
new_candidates = []

for e in all_evals:
    pos, layer = e["position"], e["layer"]
    rs, ss, kl = e["refusal_score"], e["steering_score"], e["kl_div_score"]

    if filter_fn(rs, ss, kl, layer, n_layer, kl_threshold=None, prune_layer_percentage=0.20):
        continue

    # Old: pure -refusal_score
    old_score = -rs
    old_candidates.append((old_score, pos, layer, rs, kl))

    # New: composite
    pos_idx = 5 + pos
    direction = md[pos_idx, layer].float()
    aa_layer = aa[layer].float()
    cos = torch.nn.functional.cosine_similarity(direction.unsqueeze(0), aa_layer.unsqueeze(0)).item()

    new_score = -rs
    if kl > 0 and not math.isnan(kl):
        new_score -= 0.5 * kl
    new_score += 5.0 * max(cos, 0)

    new_candidates.append((new_score, pos, layer, rs, kl, cos))

old_candidates.sort(key=lambda x: x[0], reverse=True)
new_candidates.sort(key=lambda x: x[0], reverse=True)

print(f"\n{'Rank':>4} {'Score':>8} {'Pos':>4} {'Token':>16} {'Layer':>5} {'Refusal':>8} {'KL':>8}")
print("-" * 70)
print("OLD TOP 5 (pure -refusal_score):")
for i, (sc, pos, layer, rs, kl) in enumerate(old_candidates[:5]):
    print(f"  {i+1:>2}  {sc:>8.2f} {pos:>4} {pos_tokens.get(pos, '?'):>16} {layer:>5} {rs:>8.2f} {kl:>8.2f}")

print(f"\n{'Rank':>4} {'Score':>8} {'Pos':>4} {'Token':>16} {'Layer':>5} {'Refusal':>8} {'KL':>8} {'cos(aa)':>8}")
print("-" * 80)
print("NEW TOP 5 (composite: -refusal - 0.5*KL + 5*cos(aa)):")
for i, (sc, pos, layer, rs, kl, cos) in enumerate(new_candidates[:5]):
    print(f"  {i+1:>2}  {sc:>8.2f} {pos:>4} {pos_tokens.get(pos, '?'):>16} {layer:>5} {rs:>8.2f} {kl:>8.2f} {cos:>8.4f}")

# Verify the winner changed
old_winner = old_candidates[0]
new_winner = new_candidates[0]
print(f"\nOLD winner: pos={old_winner[1]}, layer={old_winner[2]} ({pos_tokens.get(old_winner[1], '?')})")
print(f"NEW winner: pos={new_winner[1]}, layer={new_winner[2]} ({pos_tokens.get(new_winner[1], '?')})")
print(f"Changed: {'YES' if (old_winner[1], old_winner[2]) != (new_winner[1], new_winner[2]) else 'NO'}")
