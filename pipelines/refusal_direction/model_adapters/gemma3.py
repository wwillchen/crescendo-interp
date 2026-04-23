
import torch
import functools

from torch import Tensor
from transformers import AutoTokenizer, Gemma3ForCausalLM
from typing import List
from jaxtyping import Float

from utils.utils import get_orthogonalized_matrix
from model_adapters.base import ModelBase

# Gemma 3 uses the same chat markers as Gemma 2 (<start_of_turn>/<end_of_turn>).
# Tokenizer vocab differs: the token id for "I" is 236777 in Gemma 3 (235285 in Gemma 2).
GEMMA3_CHAT_TEMPLATE = """<start_of_turn>user
{instruction}<end_of_turn>
<start_of_turn>model
"""

GEMMA3_REFUSAL_TOKS = [236777]  # ["I"]


def format_instruction_gemma3_chat(
    instruction: str,
    output: str = None,
    system: str = None,
    include_trailing_whitespace: bool = True,
):
    if system is not None:
        raise ValueError("System prompts are not supported for Gemma models.")
    formatted_instruction = GEMMA3_CHAT_TEMPLATE.format(instruction=instruction)

    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()

    if output is not None:
        formatted_instruction += output

    return formatted_instruction


def tokenize_instructions_gemma3_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str] = None,
    system: str = None,
    include_trailing_whitespace=True,
):
    if outputs is not None:
        prompts = [
            format_instruction_gemma3_chat(
                instruction=instruction, output=output, system=system,
                include_trailing_whitespace=include_trailing_whitespace,
            )
            for instruction, output in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_gemma3_chat(
                instruction=instruction, system=system,
                include_trailing_whitespace=include_trailing_whitespace,
            )
            for instruction in instructions
        ]

    return tokenizer(prompts, padding=True, truncation=False, return_tensors="pt")


def orthogonalize_gemma3_weights(model, direction: Float[Tensor, "d_model"]):
    model.model.embed_tokens.weight.data = get_orthogonalized_matrix(
        model.model.embed_tokens.weight.data, direction,
    )

    for block in model.model.layers:
        block.self_attn.o_proj.weight.data = get_orthogonalized_matrix(
            block.self_attn.o_proj.weight.data.T, direction,
        ).T
        block.mlp.down_proj.weight.data = get_orthogonalized_matrix(
            block.mlp.down_proj.weight.data.T, direction,
        ).T


def act_add_gemma3_weights(model, direction: Float[Tensor, "d_model"], coeff, layer):
    dtype = model.model.layers[layer - 1].mlp.down_proj.weight.dtype
    device = model.model.layers[layer - 1].mlp.down_proj.weight.device
    bias = (coeff * direction).to(dtype=dtype, device=device)
    model.model.layers[layer - 1].mlp.down_proj.bias = torch.nn.Parameter(bias)


class Gemma3Model(ModelBase):

    # 12B fits comfortably on an H200; mirror Gemma 2's conservative size.
    pipeline_batch_size = 8
    # Median-based KL threshold for large models (see gemma.py note).
    kl_threshold = "median"

    def _load_model(self, model_path, dtype=torch.bfloat16):
        # Load text-only Gemma3ForCausalLM. The 12B/27B HF checkpoints are
        # multimodal (Gemma3ForConditionalGeneration); Gemma3ForCausalLM exposes
        # the language model alone, keeping module paths identical to Gemma 2
        # (model.model.layers, model.model.embed_tokens).
        model = Gemma3ForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="cuda",
        ).eval()
        model.requires_grad_(False)
        return model

    def _load_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.padding_side = "left"
        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(
            tokenize_instructions_gemma3_chat,
            tokenizer=self.tokenizer, system=None, include_trailing_whitespace=True,
        )

    def _get_eoi_toks(self):
        return self.tokenizer.encode(
            GEMMA3_CHAT_TEMPLATE.split("{instruction}")[-1],
            add_special_tokens=False,
        )

    def _get_refusal_toks(self):
        return GEMMA3_REFUSAL_TOKS

    def _get_model_block_modules(self):
        return self.model.model.layers

    def _get_attn_modules(self):
        # Gemma 3 keeps post_attention_layernorm between self_attn and the residual add
        # (same as Gemma 2). Ablation hooks must go AFTER this norm.
        return torch.nn.ModuleList(
            [block.post_attention_layernorm for block in self.model_block_modules]
        )

    def _get_mlp_modules(self):
        return torch.nn.ModuleList(
            [block.post_feedforward_layernorm for block in self.model_block_modules]
        )

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        return functools.partial(orthogonalize_gemma3_weights, direction=direction)

    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
        return functools.partial(
            act_add_gemma3_weights, direction=direction, coeff=coeff, layer=layer,
        )
