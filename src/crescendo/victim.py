"""Model-agnostic victim wrapper for Crescendo attacks.

Supports Qwen 1.x, Gemma 2/3, Llama, and other HuggingFace models.
Handles generation (with KV cache or manual loop) and activation extraction.
"""

import sys
import types
from typing import Dict, List, Optional

import torch


class VictimModel:
    """Wraps a HuggingFace model for generation + activation extraction."""

    def __init__(self, model_name: str, device: str = "auto"):
        # Qwen 1.x compatibility stub
        if "qwen" in model_name.lower() and "transformers_stream_generator" not in sys.modules:
            stub = types.ModuleType("transformers_stream_generator")
            stub.init_stream_support = lambda: None
            sys.modules["transformers_stream_generator"] = stub

        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.model_name = model_name
        print(f"Loading victim model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
        )
        self.model.eval()

        # Padding
        if self.tokenizer.pad_token is None:
            if hasattr(self.tokenizer, "eod_id"):
                self.tokenizer.pad_token = "<|extra_0|>"
                self.tokenizer.pad_token_id = self.tokenizer.eod_id
            else:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # Detect Qwen 1.x (no chat template)
        self._is_qwen1x = ("qwen" in model_name.lower() and
                           not getattr(self.tokenizer, "chat_template", None))
        self._can_use_generate = not self._is_qwen1x

        self.layers = self._get_layers()
        self.n_layers = len(self.layers)

        self.hidden_size = self.model.config.hidden_size

        gen_mode = "model.generate()" if self._can_use_generate else "manual loop (Qwen 1.x)"
        print(f"  Loaded: {self.n_layers} layers, hidden_size={self.hidden_size}, generation={gen_mode}")

    def _get_layers(self):
        paths = [
            lambda m: m.model.layers,       # Llama, Gemma 2, Qwen 2+
            lambda m: m.transformer.h,      # Qwen 1.x, GPT-style
        ]
        for path_fn in paths:
            try:
                layers = path_fn(self.model)
                if layers is not None and len(layers) > 0:
                    return layers
            except AttributeError:
                continue
        raise ValueError(f"Cannot find layers for {self.model_name}")

    @property
    def device(self):
        return next(self.model.parameters()).device

    def format_conversation(
        self, conversation: List[Dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        if self._is_qwen1x:
            parts = []
            for msg in conversation:
                parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
            prompt = "\n".join(parts)
            if add_generation_prompt:
                prompt += "\n<|im_start|>assistant\n"
            return prompt
        return self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=add_generation_prompt,
        )

    def _tokenize(self, text: str) -> torch.Tensor:
        encoded = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        return encoded["input_ids"].to(self.device)

    def generate_response(
        self,
        conversation: List[Dict[str, str]],
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Generate assistant response.

        Uses model.generate() with KV cache for modern models.
        Falls back to manual autoregressive loop for Qwen 1.x.
        """
        prompt = self.format_conversation(conversation, add_generation_prompt=True)
        input_ids = self._tokenize(prompt)

        if self._can_use_generate:
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )
            new_ids = output_ids[0, input_ids.shape[1]:]
            return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        # Manual autoregressive loop (Qwen 1.x compat)
        eos_ids = set()
        for tok_str in ["<|im_end|>", "<end_of_turn>", "<|eot_id|>"]:
            ids = self.tokenizer.encode(tok_str, add_special_tokens=False)
            eos_ids.update(ids)
        if self.tokenizer.eos_token_id is not None:
            eos_ids.add(self.tokenizer.eos_token_id)

        generated_ids = []
        for _ in range(max_new_tokens):
            with torch.inference_mode():
                outputs = self.model(input_ids)
                logits = outputs.logits[0, -1, :]
            if temperature > 0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
            else:
                next_token = logits.argmax().item()
            if next_token in eos_ids:
                break
            generated_ids.append(next_token)
            next_tensor = torch.tensor([[next_token]], device=input_ids.device)
            input_ids = torch.cat([input_ids, next_tensor], dim=1)

        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def extract_decision_point_activations(
        self,
        conversation: List[Dict[str, str]],
        layers: Optional[List[int]] = None,
    ) -> Dict[int, torch.Tensor]:
        """Extract activations at the decision point (last token before generation)."""
        if layers is None:
            layers = list(range(self.n_layers))

        prompt = self.format_conversation(conversation, add_generation_prompt=True)
        input_ids = self._tokenize(prompt)
        decision_pos = input_ids.shape[1] - 1

        activations = {}
        handles = []

        def make_hook(layer_idx):
            def hook_fn(module, inp, out):
                act = out[0] if isinstance(out, tuple) else out
                activations[layer_idx] = act[0, decision_pos, :].detach().cpu().float()
            return hook_fn

        for layer_idx in layers:
            h = self.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
            handles.append(h)

        try:
            with torch.inference_mode():
                self.model(input_ids)
        finally:
            for h in handles:
                h.remove()

        return activations
