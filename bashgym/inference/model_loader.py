"""
Model loader for trained models.

Loads models from ~/.bashgym/models/{run_id}/ for inference.
Supports merged models, LoRA adapters, and GGUF exports.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for model inference."""
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.95
    do_sample: bool = True
    device: str = "auto"


class ModelLoader:
    """
    Loads trained models for inference.

    Supports:
    - Merged models (full weights in merged/ or final/)
    - LoRA adapters (adapters in lora_adapters/)
    - GGUF exports (quantized models in exported_gguf/)
    """

    def __init__(self, model_path: Union[str, Path], config: Optional[InferenceConfig] = None):
        """
        Initialize the model loader.

        Args:
            model_path: Path to the model directory (e.g., ~/.bashgym/models/{run_id}/)
            config: Inference configuration
        """
        self.model_path = Path(model_path)
        self.config = config or InferenceConfig()
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def _find_model_dir(self) -> Optional[Path]:
        """Find the model directory to load from."""
        # Priority order: merged > final > lora_adapters
        candidates = [
            self.model_path / "merged",
            self.model_path / "final",
            self.model_path,
        ]

        for candidate in candidates:
            if candidate.exists():
                # Check for model files
                config_file = candidate / "config.json"
                if config_file.exists():
                    return candidate

        return None

    def _find_lora_dir(self) -> Optional[Path]:
        """Find LoRA adapter directory if available."""
        lora_dir = self.model_path / "lora_adapters"
        if lora_dir.exists() and (lora_dir / "adapter_config.json").exists():
            return lora_dir
        return None

    def load(self, device: Optional[str] = None) -> "ModelLoader":
        """
        Load the model for inference.

        Args:
            device: Device to load on ("auto", "cuda", "cpu"). Uses config default if not specified.

        Returns:
            self for method chaining
        """
        if self._loaded:
            logger.warning("Model already loaded")
            return self

        device = device or self.config.device

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "transformers and torch required for inference. "
                "Install with: pip install transformers torch"
            ) from e

        # Find model directory
        model_dir = self._find_model_dir()
        if not model_dir:
            raise ValueError(f"No valid model found in {self.model_path}")

        logger.info(f"Loading model from {model_dir}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Check for LoRA adapters
        lora_dir = self._find_lora_dir()

        if lora_dir:
            # Load base model + LoRA adapters
            logger.info(f"Loading LoRA adapters from {lora_dir}")
            try:
                from peft import PeftModel
            except ImportError:
                raise ImportError(
                    "peft required for LoRA adapter loading. "
                    "Install with: pip install peft"
                )

            # Need base model path from adapter config
            import json
            adapter_config = json.loads((lora_dir / "adapter_config.json").read_text())
            base_model_name = adapter_config.get("base_model_name_or_path", "")

            if not base_model_name:
                raise ValueError("Could not determine base model from adapter config")

            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                device_map=device
            )
            self.model = PeftModel.from_pretrained(base_model, lora_dir)
        else:
            # Load merged/full model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                device_map=device,
                trust_remote_code=True
            )

        self.model.eval()
        self._loaded = True
        logger.info(f"Model loaded successfully on {device}")

        return self

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[list] = None
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: The input prompt
            max_new_tokens: Maximum tokens to generate (uses config default if not specified)
            temperature: Sampling temperature (uses config default if not specified)
            stop_sequences: Optional list of stop sequences

        Returns:
            Generated text (excluding the prompt)
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        import torch

        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature if temperature is not None else self.config.temperature

        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                top_p=self.config.top_p if temperature > 0 else 1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode and extract generated part
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Handle stop sequences
        if stop_sequences:
            for stop_seq in stop_sequences:
                if stop_seq in generated_text:
                    generated_text = generated_text.split(stop_seq)[0]
                    break

        return generated_text.strip()

    def batch_generate(
        self,
        prompts: list,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        batch_size: int = 4
    ) -> list:
        """
        Generate text for multiple prompts.

        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum tokens to generate per prompt
            temperature: Sampling temperature
            batch_size: Number of prompts to process at once

        Returns:
            List of generated texts
        """
        results = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            batch_results = [
                self.generate(p, max_new_tokens, temperature)
                for p in batch
            ]
            results.extend(batch_results)
        return results

    def unload(self):
        """Unload the model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self._loaded = False

        # Clear GPU cache if using CUDA
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        logger.info("Model unloaded")

    def __enter__(self):
        """Context manager entry."""
        return self.load()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unload()
        return False

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._loaded
