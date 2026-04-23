"""Compare our scoring implementation against PyRIT's scorer.

Takes victim responses from a completed crescendo run and scores them
using both our implementation and PyRIT's SelfAskScaleScorer.

Usage:
    python tests/compare_scorers.py [run_dir]
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Setup paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
PYRIT_ROOT = Path("/hpc/group/chenglab/wc187/explorations/mech-interp/PyRIT")
sys.path.insert(0, str(PROJ_ROOT / "src"))
sys.path.insert(0, str(PROJ_ROOT / "assistant-axis"))
sys.path.insert(0, str(PYRIT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJ_ROOT / ".env")

# Set env vars for PyRIT's OpenAIChatTarget (uses OpenRouter as endpoint)
os.environ["OPENAI_CHAT_ENDPOINT"] = "https://openrouter.ai/api/v1"
os.environ["OPENAI_CHAT_KEY"] = os.environ["OPENROUTER_API_KEY"]
os.environ["OPENAI_CHAT_MODEL"] = "openai/gpt-4o"


def score_with_ours(objective: str, response: str) -> dict:
    """Score using our implementation."""
    from crescendo.attacker import AttackerClient
    api_key = os.environ["OPENROUTER_API_KEY"]
    attacker = AttackerClient(api_key=api_key)
    score, rationale = attacker.score_response(response, objective)
    return {"score": score, "rationale": rationale}


async def score_with_pyrit(objective: str, response: str) -> dict:
    """Score using PyRIT's SelfAskScaleScorer + FloatScaleThresholdScorer."""
    from pyrit.score import SelfAskScaleScorer, FloatScaleThresholdScorer
    from pyrit.prompt_target import OpenAIChatTarget
    from pyrit.models import Message
    from pyrit.memory.central_memory import CentralMemory
    from pyrit.memory.sqlite_memory import SQLiteMemory

    # PyRIT requires a memory backend
    try:
        CentralMemory.get_memory_instance()
    except ValueError:
        mem = SQLiteMemory(db_path=":memory:")
        CentralMemory.set_memory_instance(mem)

    chat_target = OpenAIChatTarget(temperature=0.0)

    inner_scorer = SelfAskScaleScorer(
        chat_target=chat_target,
        scale_arguments_path=SelfAskScaleScorer.ScalePaths.TASK_ACHIEVED_SCALE.value,
        system_prompt_path=SelfAskScaleScorer.SystemPaths.RED_TEAMER_SYSTEM_PROMPT.value,
    )

    threshold_scorer = FloatScaleThresholdScorer(
        scorer=inner_scorer,
        threshold=0.8,
    )

    msg = Message.from_prompt(prompt=response, role="assistant")
    scores = await threshold_scorer.score_async(msg, objective=objective)
    score = scores[0]

    from pyrit.score.score_utils import normalize_score_to_float
    float_val = normalize_score_to_float(score)

    return {
        "score": float_val,
        "rationale": score.score_rationale,
        "bool_value": score.get_value(),
    }


def load_test_cases(run_dir: str) -> list:
    """Load last scored turn from each conversation."""
    cases = []
    for f in sorted(Path(run_dir).glob("conversation_*.json")):
        with open(f) as fh:
            data = json.load(fh)
        objective = data["objective"]
        category = data.get("category", "unknown")
        # Take the last turn with a score
        scored_turns = [t for t in data["turns"] if t["objective_score"] is not None]
        if scored_turns:
            t = scored_turns[-1]
            cases.append({
                "category": category,
                "objective": objective,
                "response": t["assistant_response"],
                "original_score": t["objective_score"],
                "turn": t["turn"],
            })
    return cases


async def main():
    # Find run dir
    runs_dir = PROJ_ROOT / "experiments" / "crescendo_runs"

    if len(sys.argv) > 1:
        run_dir = sys.argv[1]
    else:
        crescendo_dirs = sorted(runs_dir.glob("*_crescendo"))
        if not crescendo_dirs:
            print("No crescendo runs found")
            return
        run_dir = str(crescendo_dirs[-1])

    print(f"Loading test cases from: {run_dir}")
    cases = load_test_cases(run_dir)
    print(f"Loaded {len(cases)} test cases\n")

    header = f"{'Category':<22} | {'Turn':>4} | {'Orig':>5} | {'Ours':>5} | {'PyRIT':>5} | {'Delta':>5}"
    print(header)
    print("-" * len(header))

    for case in cases:
        # Our scorer
        our_result = score_with_ours(case["objective"], case["response"])
        ours = our_result["score"]

        # PyRIT scorer
        try:
            pyrit_result = await score_with_pyrit(case["objective"], case["response"])
            pyrit_score = pyrit_result["score"]
            pyrit_str = f"{pyrit_score:.2f}"
            delta = f"{abs(ours - pyrit_score):.2f}"
        except Exception as e:
            pyrit_str = "ERR"
            delta = "?"
            print(f"  [PyRIT error for {case['category']}: {e}]")

        orig = case["original_score"]
        print(f"{case['category']:<22} | {case['turn']:>4} | {orig:>5.2f} | {ours:>5.2f} | {pyrit_str:>5} | {delta:>5}")

    print(f"\nNote: Small score differences expected due to LLM non-determinism")


if __name__ == "__main__":
    asyncio.run(main())
