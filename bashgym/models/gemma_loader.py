"""
Gemma 4 Model Loader — Loads base and fine-tuned models for evaluation.

Uses a single model in memory with PEFT adapter toggle to switch between
base and fine-tuned inference without doubling memory usage.
"""

import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def load_models(
    base_model: str = "unsloth/gemma-4-E4B-it",
    adapter_path: str | None = None,
    max_seq_length: int = 4096,
    load_in_4bit: bool = True,
) -> dict[str, Any]:
    """Load base model with optional LoRA adapter for A/B evaluation.

    Returns dict with:
        model: The PEFT-wrapped model (or base model if no adapter)
        tokenizer: The tokenizer
        base_generate: Callable that generates with adapter disabled
        ft_generate: Callable that generates with adapter enabled
        has_adapter: Whether an adapter was loaded
    """
    from unsloth import FastLanguageModel

    logger.info(f"Loading base model: {base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=load_in_4bit,
        device_map="sequential",
    )

    has_adapter = False
    if adapter_path:
        adapter_path = Path(adapter_path).expanduser()
        if (adapter_path / "adapter_config.json").exists():
            from peft import PeftModel

            logger.info(f"Loading LoRA adapter from: {adapter_path}")
            model = PeftModel.from_pretrained(model, str(adapter_path))
            has_adapter = True
            logger.info("Adapter loaded successfully")
        else:
            logger.warning(f"No adapter_config.json found at {adapter_path}")

    FastLanguageModel.for_inference(model)

    def _generate(
        prompt: str | list[dict],
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        use_adapter: bool = True,
    ) -> str:
        """Generate text from a prompt or message list.

        Args:
            prompt: Either a plain text string or a list of chat messages
                    in OpenAI format [{role, content}, ...]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            use_adapter: Whether to use the LoRA adapter (if loaded)
        """
        if has_adapter:
            if use_adapter:
                model.enable_adapter_layers()
            else:
                model.disable_adapter_layers()

        # Format input
        if isinstance(prompt, list):
            text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        else:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=max(temperature, 0.01),
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        generated = outputs[0][input_len:]
        return tokenizer.decode(generated, skip_special_tokens=True)

    def _compute_loss(
        messages: list[dict],
        use_adapter: bool = True,
    ) -> float:
        """Compute cross-entropy loss on a full conversation.

        Used for perplexity measurement — lower loss = model better predicts the data.
        """
        if has_adapter:
            if use_adapter:
                model.enable_adapter_layers()
            else:
                model.disable_adapter_layers()

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_seq_length
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])

        return outputs.loss.item()

    def base_generate(prompt, **kwargs):
        return _generate(prompt, use_adapter=False, **kwargs)

    def ft_generate(prompt, **kwargs):
        return _generate(prompt, use_adapter=True, **kwargs)

    def base_loss(messages):
        return _compute_loss(messages, use_adapter=False)

    def ft_loss(messages):
        return _compute_loss(messages, use_adapter=True)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "base_generate": base_generate,
        "ft_generate": ft_generate,
        "base_loss": base_loss,
        "ft_loss": ft_loss,
        "has_adapter": has_adapter,
    }
