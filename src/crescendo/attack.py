"""Crescendo attack loop — Algorithm 1 from Russinovich et al. (2024)."""

import json
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .attacker import AttackerClient
from .victim import VictimModel
from .tracker import ActivationTracker
from .prompts import OBJECTIVE_SCORE_THRESHOLD


@dataclass
class TurnRecord:
    """Record for a single turn in the crescendo conversation."""
    turn: int
    user_message: str
    assistant_response: str
    was_refused: bool = False
    was_backtracked: bool = False
    objective_score: Optional[float] = None
    score_rationale: Optional[str] = None
    refusal_proj: Dict[int, float] = field(default_factory=dict)
    assistant_proj: Dict[int, float] = field(default_factory=dict)
    cos_refusal: Dict[int, float] = field(default_factory=dict)
    cos_assistant: Dict[int, float] = field(default_factory=dict)


@dataclass
class ConversationResult:
    """Full result of a crescendo attack conversation."""
    objective: str
    turns: List[TurnRecord] = field(default_factory=list)
    success: bool = False
    final_score: float = 0.0
    n_turns: int = 0
    backtrack_count: int = 0
    error: Optional[str] = None


def run_crescendo(
    attacker: AttackerClient,
    victim: VictimModel,
    tracker: ActivationTracker,
    objective: str,
    max_turns: int = 10,
    max_backtracks: int = 10,
    verbose: bool = True,
) -> ConversationResult:
    """Run a single crescendo attack trial.

    Implements Algorithm 1 from Russinovich et al. (2024):
      1. Attacker generates escalating prompt (with score feedback)
      2. Victim responds
      3. Refusal Judge checks -> if refused, BACKTRACK
      4. If not refused, Judge scores response -> feed back to attacker
      5. If score >= threshold -> SUCCESS, continue for variants
      6. Repeat until max turns
    """
    result = ConversationResult(objective=objective)
    conversation: List[Dict[str, str]] = []
    key_layers = tracker.key_layers

    attacker.init_conversation(objective, max_turns)

    executed_turns = 0
    backtrack_count = 0
    last_response = None
    last_score = None
    last_rationale = None
    refused_text = None
    achieved = False

    if verbose:
        print(f"\n{'='*60}")
        print(f"CRESCENDO ATTACK (Algorithm 1)")
        print(f"Objective: {objective}")
        print(f"Max turns: {max_turns}, Max backtracks: {max_backtracks}")
        print(f"Score threshold: {OBJECTIVE_SCORE_THRESHOLD}")
        if tracker.enabled:
            print(f"Tracking: all {tracker.n_layers} layers, key={key_layers}")
        else:
            print(f"Tracking: DISABLED (no vectors)")
        print(f"{'='*60}")

    while executed_turns < max_turns:
        turn_display = executed_turns + 1
        try:
            if verbose:
                bt_str = " [after backtrack]" if refused_text else ""
                print(f"\n--- Turn {turn_display}/{max_turns}{bt_str} ---")

            attacker_question = attacker.generate_next_prompt(
                current_turn=turn_display, max_turns=max_turns,
                last_response=last_response, last_score=last_score,
                last_score_rationale=last_rationale, refused_text=refused_text,
                success_flag=achieved,
            )
            refused_text = None

            if verbose:
                print(f"  Attacker: {attacker_question[:150]}")

            conversation.append({"role": "user", "content": attacker_question})

            # Extract activations at decision point
            if tracker.enabled:
                activations = victim.extract_decision_point_activations(conversation, layers=None)
                refusal_proj, assistant_proj, cos_refusal, cos_assistant = tracker.compute_projections(activations)
            else:
                refusal_proj, assistant_proj, cos_refusal, cos_assistant = {}, {}, {}, {}

            # Generate victim response
            response = victim.generate_response(conversation)
            conversation.append({"role": "assistant", "content": response})

            if verbose:
                print(f"  Victim: {response[:200]}")
                for l in key_layers:
                    print(f"    L{l:2d}: ref={refusal_proj.get(l, 0):+.3f}, ast={assistant_proj.get(l, 0):+.3f}")

            # Refusal Judge
            is_refusal = attacker.check_refusal(response)

            if is_refusal and backtrack_count < max_backtracks:
                if verbose:
                    print(f"  -> REFUSAL — backtracking (#{backtrack_count + 1}/{max_backtracks})")

                result.turns.append(TurnRecord(
                    turn=turn_display, user_message=attacker_question,
                    assistant_response=response, was_refused=True, was_backtracked=True,
                    refusal_proj=refusal_proj, assistant_proj=assistant_proj,
                    cos_refusal=cos_refusal, cos_assistant=cos_assistant,
                ))

                conversation.pop()
                conversation.pop()
                attacker.backtrack_attacker_conversation(attacker_question)
                refused_text = attacker_question
                backtrack_count += 1
                continue

            if is_refusal:
                if verbose:
                    print(f"  -> REFUSAL (max backtracks reached, removing from conversation)")
                conversation.pop()
                conversation.pop()
                result.turns.append(TurnRecord(
                    turn=turn_display, user_message=attacker_question,
                    assistant_response=response, was_refused=True, was_backtracked=True,
                    refusal_proj=refusal_proj, assistant_proj=assistant_proj,
                    cos_refusal=cos_refusal, cos_assistant=cos_assistant,
                ))
                executed_turns += 1
                last_response = None
                last_score = None
                last_rationale = None
                refused_text = attacker_question
                continue

            # Judge scores the response
            score, rationale = attacker.score_response(response, objective)
            last_response = response
            last_score = score
            last_rationale = rationale

            if verbose:
                print(f"  -> Score: {score:.2f} ({rationale[:80]})")

            result.turns.append(TurnRecord(
                turn=turn_display, user_message=attacker_question,
                assistant_response=response, was_refused=False, was_backtracked=False,
                objective_score=score, score_rationale=rationale,
                refusal_proj=refusal_proj, assistant_proj=assistant_proj,
                cos_refusal=cos_refusal, cos_assistant=cos_assistant,
            ))

            if score >= OBJECTIVE_SCORE_THRESHOLD and not achieved:
                achieved = True
                if verbose:
                    print(f"  * OBJECTIVE ACHIEVED (score={score:.2f})")

            executed_turns += 1

        except Exception as e:
            print(f"  ERROR at turn {turn_display}: {e}")
            traceback.print_exc()
            result.error = str(e)
            break

    result.success = achieved
    result.final_score = last_score or 0.0
    result.n_turns = executed_turns
    result.backtrack_count = backtrack_count
    return result


def save_result(result: ConversationResult, run_dir: Path, index: int):
    data = {
        "objective": result.objective,
        "success": result.success,
        "final_score": result.final_score,
        "n_turns": result.n_turns,
        "backtrack_count": result.backtrack_count,
        "error": result.error,
        "turns": [],
    }
    for t in result.turns:
        data["turns"].append({
            "turn": t.turn,
            "user_message": t.user_message,
            "assistant_response": t.assistant_response,
            "was_refused": t.was_refused,
            "was_backtracked": t.was_backtracked,
            "objective_score": t.objective_score,
            "score_rationale": t.score_rationale,
            "refusal_proj": {str(k): v for k, v in t.refusal_proj.items()},
            "assistant_proj": {str(k): v for k, v in t.assistant_proj.items()},
            "cos_refusal": {str(k): v for k, v in t.cos_refusal.items()},
            "cos_assistant": {str(k): v for k, v in t.cos_assistant.items()},
        })

    out_path = Path(run_dir) / f"conversation_{index:03d}.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {out_path}")


def print_trajectory_summary(result: ConversationResult, tracker: ActivationTracker):
    if not result.turns:
        return

    print(f"\n{'='*60}")
    print(f"TRAJECTORY — {result.objective[:50]}")
    print(f"Result: {'SUCCESS' if result.success else 'FAILURE'} | "
          f"Score: {result.final_score:.2f} | "
          f"Turns: {result.n_turns} | Backtracks: {result.backtrack_count}")
    print(f"{'='*60}")

    if tracker.enabled:
        rl, al = tracker.refusal_layer, tracker.assistant_layer
        print(f"\n  Turn | Ref@L{rl:2d} | Ast@L{al:2d} | Score | Status")
        print(f"  -----|----------|----------|-------|-------")
        for t in result.turns:
            rp = t.refusal_proj.get(rl, 0)
            ap = t.assistant_proj.get(al, 0)
            sc = f"{t.objective_score:.2f}" if t.objective_score is not None else "  - "
            if t.was_backtracked:
                status = "BACKTRACK"
            elif t.objective_score and t.objective_score >= OBJECTIVE_SCORE_THRESHOLD:
                status = "* SUCCESS"
            else:
                status = ""
            print(f"  {t.turn:4d} | {rp:+8.3f} | {ap:+8.3f} | {sc} | {status}")
    else:
        print(f"\n  Turn | Score | Status")
        print(f"  -----|-------|-------")
        for t in result.turns:
            sc = f"{t.objective_score:.2f}" if t.objective_score is not None else "  - "
            status = "BACKTRACK" if t.was_backtracked else ("* SUCCESS" if t.objective_score and t.objective_score >= OBJECTIVE_SCORE_THRESHOLD else "")
            print(f"  {t.turn:4d} | {sc} | {status}")
