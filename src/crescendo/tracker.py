"""Activation tracker — projects activations onto refusal direction and assistant axis."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


class ActivationTracker:
    """Projects activations onto refusal direction and assistant axis.

    Args:
        vectors_dir: Path to the model's vector directory containing:
            - assistant_axis.pt (dict with "axis" key, shape: n_layers x d_model)
            - refusal_mean_diffs.pt (shape: n_positions x n_layers x d_model)
            - metadata.json ({"pos": int, "layer": int})
    """

    def __init__(self, vectors_dir: str):
        vectors_path = Path(vectors_dir)

        # Check if vectors exist
        axis_path = vectors_path / "assistant_axis.pt"
        mean_diffs_path = vectors_path / "refusal_mean_diffs.pt"
        metadata_path = vectors_path / "metadata.json"

        if not all(p.exists() for p in [axis_path, mean_diffs_path, metadata_path]):
            print(f"  WARNING: Missing vectors in {vectors_dir}")
            print(f"  Activation tracking disabled.")
            self.enabled = False
            return

        self.enabled = True
        print(f"Loading vectors from {vectors_dir}")

        # Load refusal direction (per-layer)
        mean_diffs = torch.load(mean_diffs_path, map_location="cpu", weights_only=True)
        with open(metadata_path) as f:
            self.refusal_meta = json.load(f)

        n_positions, n_layers, d_model = mean_diffs.shape
        pos_idx = n_positions + self.refusal_meta["pos"]
        self.refusal_per_layer = mean_diffs[pos_idx].float()
        self.refusal_layer = self.refusal_meta["layer"]

        # Load assistant axis
        axis_data = torch.load(axis_path, map_location="cpu", weights_only=True)
        if isinstance(axis_data, dict):
            self.assistant_axis = axis_data["axis"].float()
        else:
            self.assistant_axis = axis_data.float()

        self.n_layers = n_layers
        self.d_model = d_model

        # Peak assistant-axis layer (highest norm)
        axis_norms = self.assistant_axis.norm(dim=1)
        self.assistant_layer = axis_norms.argmax().item()

        # Pre-compute normalized directions
        self.refusal_normed = {}
        self.assistant_normed = {}
        for l in range(n_layers):
            r, a = self.refusal_per_layer[l], self.assistant_axis[l]
            rn, an = r.norm(), a.norm()
            self.refusal_normed[l] = r / rn if rn > 0 else r
            self.assistant_normed[l] = a / an if an > 0 else a

        cos_r = float(self.refusal_normed[self.refusal_layer] @ self.assistant_normed[self.refusal_layer])
        cos_a = float(self.refusal_normed[self.assistant_layer] @ self.assistant_normed[self.assistant_layer])
        print(f"  Refusal peak layer={self.refusal_layer}, Assistant peak layer={self.assistant_layer}")
        print(f"  cos(refusal, assistant): {cos_r:.4f} @ L{self.refusal_layer}, {cos_a:.4f} @ L{self.assistant_layer}")

    @property
    def key_layers(self) -> List[int]:
        if not self.enabled:
            return []
        return sorted(set([self.refusal_layer, self.assistant_layer, self.n_layers - 1]))

    def compute_projections(
        self, activations: Dict[int, torch.Tensor]
    ) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float], Dict[int, float]]:
        refusal_proj, assistant_proj, cos_refusal, cos_assistant = {}, {}, {}, {}
        if not self.enabled:
            return refusal_proj, assistant_proj, cos_refusal, cos_assistant

        for layer_idx, act in activations.items():
            act_f = act.float()
            act_norm = act_f.norm()
            r_dir, a_dir = self.refusal_per_layer[layer_idx], self.assistant_axis[layer_idx]
            r_norm, a_norm = r_dir.norm(), a_dir.norm()

            refusal_proj[layer_idx] = float(act_f @ r_dir / r_norm) if r_norm > 0 else 0.0
            assistant_proj[layer_idx] = float(act_f @ a_dir / a_norm) if a_norm > 0 else 0.0
            cos_refusal[layer_idx] = float(act_f @ r_dir / (act_norm * r_norm)) if (act_norm > 0 and r_norm > 0) else 0.0
            cos_assistant[layer_idx] = float(act_f @ a_dir / (act_norm * a_norm)) if (act_norm > 0 and a_norm > 0) else 0.0

        return refusal_proj, assistant_proj, cos_refusal, cos_assistant
