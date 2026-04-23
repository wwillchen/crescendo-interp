"""Merge successful conversations from the original run + reruns into one directory."""

import json
import shutil
from pathlib import Path
from collections import defaultdict

RUNS_DIR = Path("experiments/crescendo_runs")
ORIGINAL = RUNS_DIR / "20260405_233409_crescendo"
RERUN_PREFIX = "20260407_"
OUTPUT = RUNS_DIR / "merged_gemma3_12b"

# Map objective text -> category (for --objective runs that show "custom")
OBJECTIVES_FILE = Path("objectives/harmful_9.json")


def load_category_map():
    with open(OBJECTIVES_FILE) as f:
        data = json.load(f)
    return {entry["objective"]: entry["category"] for entry in data}


def collect_conversations(run_dir, category_map):
    """Return list of (json_path, index, category) for non-errored conversations."""
    results = []
    for p in sorted(run_dir.glob("conversation_*.json")):
        idx = int(p.stem.split("_")[1])
        with open(p) as f:
            data = json.load(f)
        if data.get("error") is not None:
            continue
        # Fix "custom" category from --objective runs
        cat = data.get("category", "custom")
        if cat == "custom":
            cat = category_map.get(data["objective"], "custom")
        results.append((p, idx, cat))
    return results


def main():
    category_map = load_category_map()

    # Collect from original run
    print(f"Scanning original: {ORIGINAL.name}")
    original_convs = collect_conversations(ORIGINAL, category_map)
    print(f"  {len(original_convs)} non-errored conversations")

    # Collect categories already covered by original
    original_obj_cats = defaultdict(int)
    for _, _, cat in original_convs:
        original_obj_cats[cat] += 1

    # Collect from reruns
    rerun_convs = []
    rerun_dirs = sorted(d for d in RUNS_DIR.iterdir()
                        if d.is_dir() and d.name.startswith(RERUN_PREFIX))
    for rd in rerun_dirs:
        print(f"Scanning rerun: {rd.name}")
        convs = collect_conversations(rd, category_map)
        print(f"  {len(convs)} non-errored conversations")
        rerun_convs.extend([(p, idx, cat, rd) for p, idx, cat in convs])

    # Build merged list: original successes + rerun successes, sorted by category
    all_convs = []
    for p, idx, cat in original_convs:
        all_convs.append((cat, p, idx, ORIGINAL))
    for p, idx, cat, rd in rerun_convs:
        all_convs.append((cat, p, idx, rd))

    # Sort by category then by source dir then by original index
    all_convs.sort(key=lambda x: (x[0], x[3].name, x[2]))

    # Create output directory
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # Copy and renumber
    cat_counts = defaultdict(int)
    total = 0
    for new_idx, (cat, json_path, old_idx, source_dir) in enumerate(all_convs):
        # Read and fix category in JSON
        with open(json_path) as f:
            data = json.load(f)
        data["category"] = cat
        data["_source_dir"] = source_dir.name
        data["_source_index"] = old_idx

        # Write conversation JSON
        out_json = OUTPUT / f"conversation_{new_idx:03d}.json"
        with open(out_json, "w") as f:
            json.dump(data, f, indent=2)

        # Copy activation files if they exist
        for prefix in ["activations", "response_activations"]:
            src_pt = source_dir / f"{prefix}_{old_idx:03d}.pt"
            dst_pt = OUTPUT / f"{prefix}_{new_idx:03d}.pt"
            if src_pt.exists():
                shutil.copy2(src_pt, dst_pt)

        cat_counts[cat] += 1
        total += 1

    # Write merged config
    with open(ORIGINAL / "config.json") as f:
        orig_config = json.load(f)
    merged_config = {
        **orig_config,
        "run_id": "merged_gemma3_12b",
        "n_conversations_total": total,
        "source_dirs": [ORIGINAL.name] + [rd.name for rd in rerun_dirs],
        "merge_note": "Merged non-errored conversations from original + reruns",
    }
    with open(OUTPUT / "config.json", "w") as f:
        json.dump(merged_config, f, indent=2)

    # Summary
    print(f"\n{'='*50}")
    print(f"Merged {total} conversations into {OUTPUT}")
    print(f"{'='*50}")
    print(f"{'Category':<25} {'Count':>5}")
    print(f"{'-'*25} {'-'*5}")
    for cat in sorted(cat_counts):
        print(f"{cat:<25} {cat_counts[cat]:>5}")
    print(f"{'-'*25} {'-'*5}")
    print(f"{'TOTAL':<25} {total:>5}")


if __name__ == "__main__":
    main()
