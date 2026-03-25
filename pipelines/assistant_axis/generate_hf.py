#!/usr/bin/env python3
"""
Generate model responses for all roles using HuggingFace (not vLLM).

This script is designed for models like Qwen 1.8B that have compatibility issues
with vLLM's tokenizer/chat template handling. It uses ProbingModel.generate() or
a manual autoregressive loop as fallback.

Outputs JSONL files per role in the same format as the standard pipeline Stage 1,
so Stages 2-5 can be reused unchanged.

Usage:
    python pipelines/assistant_axis/generate_hf.py \
        --model Qwen/Qwen-1_8B-Chat \
        --roles_dir assistant-axis/data/roles/instructions \
        --questions_file assistant-axis/data/extraction_questions.jsonl \
        --output_dir outputs/qwen-1_8b-chat/responses \
        --question_count 240 --max_new_tokens 256
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional

import jsonlines
import torch
from tqdm import tqdm

# Add project paths
PROJ_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJ_ROOT / "assistant-axis"))

from assistant_axis.internals import ProbingModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_questions(questions_file: str, question_count: int = 240) -> List[str]:
    """Load extraction questions from JSONL file."""
    questions = []
    with jsonlines.open(questions_file) as reader:
        for item in reader:
            questions.append(item["question"])
    return questions[:question_count]


def load_role(role_file: Path) -> dict:
    """Load a role JSON file."""
    with open(role_file) as f:
        return json.load(f)


def get_role_prompts(role_data: dict) -> List[str]:
    """Extract system prompt variants from role data."""
    return [entry["pos"] for entry in role_data.get("instruction", [])]


def format_instruction(instruction: str, model_short_name: str) -> str:
    """Format instruction, replacing {model_name} placeholder."""
    return instruction.replace("{model_name}", model_short_name)


def generate_with_manual_loop(
    model,
    tokenizer,
    conversation: List[Dict[str, str]],
    max_new_tokens: int = 256,
) -> str:
    """
    Manual autoregressive generation for Qwen 1.x (no KV cache).
    Pattern from victim.py:121-146.
    """
    formatted = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]

    # EOS tokens to stop on
    eos_tokens = set()
    for tok_str in ["<|im_end|>", "<|endoftext|>"]:
        tok_id = tokenizer.convert_tokens_to_ids(tok_str)
        if tok_id is not None and tok_id != tokenizer.unk_token_id:
            eos_tokens.add(tok_id)
    if tokenizer.eos_token_id is not None:
        eos_tokens.add(tokenizer.eos_token_id)

    generated_ids = []
    for _ in range(max_new_tokens):
        with torch.inference_mode():
            outputs = model(input_ids)
            logits = outputs.logits[0, -1, :]

        probs = torch.softmax(logits.float(), dim=-1)
        next_token = torch.multinomial(probs, 1).item()

        if next_token in eos_tokens:
            break

        generated_ids.append(next_token)
        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_token]], device=input_ids.device)], dim=1
        )

    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def generate_response(
    pm: ProbingModel,
    conversation: List[Dict[str, str]],
    max_new_tokens: int = 256,
    use_manual_loop: bool = False,
) -> str:
    """Generate a single response."""
    if use_manual_loop:
        return generate_with_manual_loop(
            pm.model, pm.tokenizer, conversation, max_new_tokens
        )

    # Try ProbingModel.generate() first
    try:
        # Build the prompt with system message included
        formatted = pm.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        return pm.generate(
            formatted,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            chat_format=False,  # Already formatted
        )
    except Exception as e:
        logger.warning(f"ProbingModel.generate() failed: {e}. Falling back to manual loop.")
        return generate_with_manual_loop(
            pm.model, pm.tokenizer, conversation, max_new_tokens
        )


def process_role(
    pm: ProbingModel,
    role_name: str,
    role_data: dict,
    questions: List[str],
    output_dir: Path,
    model_short_name: str,
    prompt_indices: List[int],
    max_new_tokens: int = 256,
    use_manual_loop: bool = False,
) -> int:
    """Generate responses for a single role and save to JSONL."""
    output_file = output_dir / f"{role_name}.jsonl"

    # Check existing progress
    existing_keys = set()
    if output_file.exists():
        with jsonlines.open(output_file) as reader:
            for entry in reader:
                key = f"pos_p{entry['prompt_index']}_q{entry['question_index']}"
                existing_keys.add(key)

    prompts = get_role_prompts(role_data)
    if not prompts:
        return 0

    count = 0
    with jsonlines.open(output_file, mode="a") as writer:
        for prompt_idx in prompt_indices:
            if prompt_idx >= len(prompts):
                continue

            sys_prompt = format_instruction(prompts[prompt_idx], model_short_name)

            for q_idx, question in enumerate(questions):
                key = f"pos_p{prompt_idx}_q{q_idx}"
                if key in existing_keys:
                    continue

                # Build conversation
                conversation = []
                if sys_prompt:
                    conversation.append({"role": "system", "content": sys_prompt})
                conversation.append({"role": "user", "content": question})

                # Generate
                response = generate_response(
                    pm, conversation, max_new_tokens, use_manual_loop
                )

                # Save in pipeline-compatible format
                result = {
                    "system_prompt": sys_prompt,
                    "prompt_index": prompt_idx,
                    "question_index": q_idx,
                    "question": question,
                    "label": "pos",
                    "conversation": conversation
                    + [{"role": "assistant", "content": response}],
                }
                writer.write(result)
                count += 1

    return count


def main():
    parser = argparse.ArgumentParser(
        description="Generate role responses using HuggingFace"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="HuggingFace model name"
    )
    parser.add_argument(
        "--roles_dir",
        type=str,
        default="assistant-axis/data/roles/instructions",
        help="Directory containing role JSON files",
    )
    parser.add_argument(
        "--questions_file",
        type=str,
        default="assistant-axis/data/extraction_questions.jsonl",
        help="Path to questions JSONL file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for JSONL files",
    )
    parser.add_argument(
        "--question_count",
        type=int,
        default=240,
        help="Number of questions per role",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate per response",
    )
    parser.add_argument(
        "--prompt_indices",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated prompt indices to use (default: 0-4)",
    )
    parser.add_argument(
        "--manual_loop",
        action="store_true",
        help="Force manual autoregressive loop (for Qwen 1.x KV cache issues)",
    )
    parser.add_argument(
        "--roles",
        nargs="+",
        help="Specific role names to process (default: all)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    roles_dir = Path(args.roles_dir)
    prompt_indices = [int(x.strip()) for x in args.prompt_indices.split(",")]

    # Detect model short name
    model_lower = args.model.lower()
    if "qwen" in model_lower:
        model_short_name = "Qwen"
    elif "gemma" in model_lower:
        model_short_name = "Gemma"
    else:
        model_short_name = args.model.split("/")[-1]

    # Load questions
    questions = load_questions(args.questions_file, args.question_count)
    logger.info(f"Loaded {len(questions)} questions")

    # Load model
    logger.info(f"Loading model: {args.model}")
    pm = ProbingModel(args.model)
    logger.info(
        f"Model loaded. Layers: {len(pm.get_layers())}, Hidden: {pm.hidden_size}"
    )

    # Detect if we need manual loop (Qwen 1.x)
    use_manual_loop = args.manual_loop
    if not use_manual_loop and "qwen" in model_lower:
        # Test if model.generate() works
        try:
            test_ids = pm.tokenizer("Hello", return_tensors="pt").to(pm.device)
            with torch.inference_mode():
                pm.model.generate(**test_ids, max_new_tokens=5)
            logger.info("model.generate() works — using standard generation")
        except Exception as e:
            logger.warning(f"model.generate() failed: {e}")
            logger.info("Falling back to manual autoregressive loop")
            use_manual_loop = True

    # Get role files
    role_files = sorted(roles_dir.glob("*.json"))
    logger.info(f"Found {len(role_files)} role files")

    if args.roles:
        role_files = [f for f in role_files if f.stem in args.roles]

    # Count total work
    total_roles = len(role_files)
    total_per_role = len(prompt_indices) * len(questions)
    logger.info(
        f"Processing {total_roles} roles × {len(prompt_indices)} prompts × {len(questions)} questions = {total_roles * total_per_role} generations"
    )

    # Process roles
    total_generated = 0
    for role_file in tqdm(role_files, desc="Roles"):
        role_name = role_file.stem
        try:
            role_data = load_role(role_file)
            if "instruction" not in role_data:
                continue

            count = process_role(
                pm=pm,
                role_name=role_name,
                role_data=role_data,
                questions=questions,
                output_dir=output_dir,
                model_short_name=model_short_name,
                prompt_indices=prompt_indices,
                max_new_tokens=args.max_new_tokens,
                use_manual_loop=use_manual_loop,
            )
            total_generated += count
            if count > 0:
                logger.info(f"  {role_name}: {count} new responses")

        except Exception as e:
            logger.error(f"Error processing {role_name}: {e}")

    logger.info(f"\nDone! Generated {total_generated} total responses.")


if __name__ == "__main__":
    main()
