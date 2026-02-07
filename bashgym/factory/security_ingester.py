"""
Security Dataset Ingester

Converts public security datasets (EMBER, PhishTank, URLhaus, MalwareBazaar,
CIC-IDS) directly into NeMo-compatible training JSONL.  Bypasses the trace
pipeline entirely — produces the same TrainingExample objects that the rest
of the factory layer uses, so the output plugs straight into the existing
trainer.

Two conversion modes:
- direct  — Template-based, no API calls, fast
- enriched — LLM generates detailed reasoning per sample (async, batched)
"""

import bz2
import csv
import json
import hashlib
import random
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import httpx

from bashgym.factory.data_factory import TrainingExample
from bashgym.factory.security_prompts import (
    PHISHING_SYSTEM_PROMPT,
    MALWARE_SYSTEM_PROMPT,
    NETWORK_SYSTEM_PROMPT,
    phishing_user_prompt,
    malware_user_prompt,
    network_user_prompt,
    phishing_assistant_response_direct,
    malware_assistant_response_direct,
    network_assistant_response_direct,
    build_enrichment_prompt,
)


# =============================================================================
# Enums & Config
# =============================================================================

class SecurityDomain(str, Enum):
    PHISHING = "phishing"
    MALWARE = "malware"
    NETWORK = "network"


class DatasetType(str, Enum):
    EMBER = "ember"
    PHISHTANK = "phishtank"
    URLHAUS = "urlhaus"
    MALWAREBAZAAR = "malwarebazaar"
    CIC_IDS = "cic_ids"


class ConversionMode(str, Enum):
    DIRECT = "direct"
    ENRICHED = "enriched"


# Map dataset types to their security domain
DATASET_DOMAINS: Dict[DatasetType, SecurityDomain] = {
    DatasetType.EMBER: SecurityDomain.MALWARE,
    DatasetType.PHISHTANK: SecurityDomain.PHISHING,
    DatasetType.URLHAUS: SecurityDomain.PHISHING,
    DatasetType.MALWAREBAZAAR: SecurityDomain.MALWARE,
    DatasetType.CIC_IDS: SecurityDomain.NETWORK,
}


@dataclass
class IngestionConfig:
    """Configuration for a security dataset ingestion run."""
    dataset_type: DatasetType
    input_path: str
    mode: ConversionMode = ConversionMode.DIRECT
    max_samples: Optional[int] = None
    balance_classes: bool = True
    benign_ratio: float = 0.3  # Target ratio of benign samples
    output_dir: str = "data/security_training"
    train_split: float = 0.9

    # Enrichment settings (only used in enriched mode)
    enrichment_provider: str = "anthropic"  # anthropic | nim
    enrichment_model: str = "claude-sonnet-4-5-20250929"
    enrichment_api_key: Optional[str] = None
    enrichment_batch_size: int = 10
    enrichment_max_concurrent: int = 5


@dataclass
class IngestionResult:
    """Result of a completed ingestion run."""
    dataset_type: str
    mode: str
    total_samples_read: int
    examples_generated: int
    train_path: Optional[str] = None
    val_path: Optional[str] = None
    train_count: int = 0
    val_count: int = 0
    class_distribution: Dict[str, int] = field(default_factory=dict)


# =============================================================================
# Base Adapter
# =============================================================================

class SecurityDatasetAdapter(ABC):
    """Abstract base class for security dataset adapters."""

    domain: SecurityDomain
    system_prompt: str

    def __init__(self, input_path: str, config: IngestionConfig):
        self.input_path = Path(input_path)
        self.config = config
        if not self.input_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.input_path}")

    @abstractmethod
    def iterate_samples(self) -> Generator[Dict[str, Any], None, None]:
        """Yield raw samples from the dataset file."""

    def to_training_example(self, sample: Dict[str, Any], mode: ConversionMode) -> Optional[TrainingExample]:
        """Convert a raw sample to a TrainingExample."""
        user_prompt = self._build_user_prompt(sample)
        if not user_prompt or len(user_prompt.strip()) < 10:
            return None

        if mode == ConversionMode.DIRECT:
            assistant_response = self._build_direct_response(sample)
        else:
            # In enriched mode, direct response is used as placeholder;
            # caller replaces it with LLM output
            assistant_response = self._build_direct_response(sample)

        if not assistant_response:
            return None

        example_id = hashlib.sha256(
            f"{self.config.dataset_type}:{user_prompt[:200]}".encode()
        ).hexdigest()[:16]

        return TrainingExample(
            example_id=example_id,
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            assistant_response=assistant_response,
            metadata={
                "source": f"security:{self.config.dataset_type}",
                "domain": self.domain.value,
                "mode": mode.value,
            },
        )

    def build_enrichment_prompt(self, sample: Dict[str, Any]) -> str:
        """Build an enrichment prompt for LLM-based detailed analysis."""
        user_prompt = self._build_user_prompt(sample)
        direct_response = self._build_direct_response(sample)
        return build_enrichment_prompt(
            self.domain.value, sample, user_prompt, direct_response
        )

    @abstractmethod
    def _build_user_prompt(self, sample: Dict[str, Any]) -> str:
        """Build the user prompt from a sample."""

    @abstractmethod
    def _build_direct_response(self, sample: Dict[str, Any]) -> str:
        """Build the direct-mode assistant response."""

    @abstractmethod
    def _get_label(self, sample: Dict[str, Any]) -> str:
        """Return 'malicious' or 'benign' for class balancing."""


# =============================================================================
# Concrete Adapters
# =============================================================================

class EMBERAdapter(SecurityDatasetAdapter):
    """EMBER dataset adapter. Input: JSONL (one JSON object per line)."""

    domain = SecurityDomain.MALWARE
    system_prompt = MALWARE_SYSTEM_PROMPT

    def iterate_samples(self) -> Generator[Dict[str, Any], None, None]:
        with open(self.input_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def _build_user_prompt(self, sample: Dict[str, Any]) -> str:
        return malware_user_prompt(sample)

    def _build_direct_response(self, sample: Dict[str, Any]) -> str:
        return malware_assistant_response_direct(sample)

    def _get_label(self, sample: Dict[str, Any]) -> str:
        label = sample.get("label", 1)
        return "benign" if int(label) == 0 else "malicious"


class PhishTankAdapter(SecurityDatasetAdapter):
    """PhishTank adapter. Input: JSON or .json.bz2 (array of objects)."""

    domain = SecurityDomain.PHISHING
    system_prompt = PHISHING_SYSTEM_PROMPT

    def iterate_samples(self) -> Generator[Dict[str, Any], None, None]:
        path = self.input_path
        if path.suffix == ".bz2":
            with bz2.open(path, "rt", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)

        if isinstance(data, list):
            yield from data
        elif isinstance(data, dict):
            # Some exports wrap in a top-level key
            for key in ("data", "results", "entries"):
                if key in data and isinstance(data[key], list):
                    yield from data[key]
                    return
            yield data

    def _build_user_prompt(self, sample: Dict[str, Any]) -> str:
        return phishing_user_prompt(sample)

    def _build_direct_response(self, sample: Dict[str, Any]) -> str:
        return phishing_assistant_response_direct(sample)

    def _get_label(self, sample: Dict[str, Any]) -> str:
        # PhishTank entries are all phishing
        return "malicious"


class URLhausAdapter(SecurityDatasetAdapter):
    """URLhaus adapter. Input: CSV (with # comment header lines) or JSON."""

    domain = SecurityDomain.PHISHING
    system_prompt = PHISHING_SYSTEM_PROMPT

    def iterate_samples(self) -> Generator[Dict[str, Any], None, None]:
        path = self.input_path

        if path.suffix == ".json":
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
            if isinstance(data, list):
                yield from data
            elif isinstance(data, dict):
                for key in ("urls", "data", "results"):
                    if key in data and isinstance(data[key], list):
                        yield from data[key]
                        return
            return

        # CSV with comment lines starting with #
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            # Skip comment lines
            lines = []
            for raw_line in f:
                if raw_line.startswith("#"):
                    continue
                lines.append(raw_line)

        if not lines:
            return

        reader = csv.DictReader(lines)
        for row in reader:
            yield dict(row)

    def _build_user_prompt(self, sample: Dict[str, Any]) -> str:
        return phishing_user_prompt(sample)

    def _build_direct_response(self, sample: Dict[str, Any]) -> str:
        return phishing_assistant_response_direct(sample)

    def _get_label(self, sample: Dict[str, Any]) -> str:
        # URLhaus entries are all malicious
        return "malicious"


class MalwareBazaarAdapter(SecurityDatasetAdapter):
    """MalwareBazaar adapter. Input: JSON (array) or JSONL."""

    domain = SecurityDomain.MALWARE
    system_prompt = MALWARE_SYSTEM_PROMPT

    def iterate_samples(self) -> Generator[Dict[str, Any], None, None]:
        path = self.input_path

        # Try JSON array first
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
            if isinstance(data, list):
                yield from data
                return
            if isinstance(data, dict):
                for key in ("data", "results", "samples"):
                    if key in data and isinstance(data[key], list):
                        yield from data[key]
                        return
                yield data
                return
        except json.JSONDecodeError:
            pass

        # Fall back to JSONL
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def _build_user_prompt(self, sample: Dict[str, Any]) -> str:
        return malware_user_prompt(sample)

    def _build_direct_response(self, sample: Dict[str, Any]) -> str:
        return malware_assistant_response_direct(sample)

    def _get_label(self, sample: Dict[str, Any]) -> str:
        # MalwareBazaar entries are all malware
        return "malicious"


class CICIDSAdapter(SecurityDatasetAdapter):
    """CIC-IDS adapter. Input: CSV with 78-82 columns."""

    domain = SecurityDomain.NETWORK
    system_prompt = NETWORK_SYSTEM_PROMPT

    def iterate_samples(self) -> Generator[Dict[str, Any], None, None]:
        with open(
            self.input_path, "r", encoding="utf-8", errors="replace"
        ) as f:
            # Strip whitespace from header names (CIC-IDS quirk)
            reader = csv.DictReader(f)
            if reader.fieldnames:
                reader.fieldnames = [h.strip() for h in reader.fieldnames]
            for row in reader:
                # Strip whitespace from keys and values
                yield {k.strip(): v.strip() if isinstance(v, str) else v for k, v in row.items()}

    def _build_user_prompt(self, sample: Dict[str, Any]) -> str:
        return network_user_prompt(sample)

    def _build_direct_response(self, sample: Dict[str, Any]) -> str:
        return network_assistant_response_direct(sample)

    def _get_label(self, sample: Dict[str, Any]) -> str:
        label = sample.get("Label", sample.get("label", "BENIGN"))
        return "benign" if str(label).strip().upper() == "BENIGN" else "malicious"


# =============================================================================
# Adapter Registry
# =============================================================================

ADAPTER_REGISTRY: Dict[DatasetType, type] = {
    DatasetType.EMBER: EMBERAdapter,
    DatasetType.PHISHTANK: PhishTankAdapter,
    DatasetType.URLHAUS: URLhausAdapter,
    DatasetType.MALWAREBAZAAR: MalwareBazaarAdapter,
    DatasetType.CIC_IDS: CICIDSAdapter,
}


def get_adapter(
    dataset_type: DatasetType, input_path: str, config: IngestionConfig
) -> SecurityDatasetAdapter:
    """Factory function to get the right adapter for a dataset type."""
    adapter_cls = ADAPTER_REGISTRY.get(dataset_type)
    if not adapter_cls:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    return adapter_cls(input_path, config)


# =============================================================================
# Orchestrator
# =============================================================================

class SecurityIngester:
    """Orchestrates ingestion of security datasets into training examples."""

    def __init__(self, config: IngestionConfig):
        self.config = config
        self.adapter = get_adapter(config.dataset_type, config.input_path, config)

    def ingest_direct(self) -> IngestionResult:
        """Synchronous direct-mode ingestion. No API calls."""
        examples: List[TrainingExample] = []
        malicious_examples: List[TrainingExample] = []
        benign_examples: List[TrainingExample] = []
        total_read = 0

        for sample in self.adapter.iterate_samples():
            total_read += 1

            example = self.adapter.to_training_example(sample, ConversionMode.DIRECT)
            if example is None:
                continue

            label = self.adapter._get_label(sample)
            if label == "benign":
                benign_examples.append(example)
            else:
                malicious_examples.append(example)

            # Early exit if we have enough samples
            if self.config.max_samples and (
                len(malicious_examples) + len(benign_examples)
                >= self.config.max_samples * 2
            ):
                break

        # Class balancing
        if self.config.balance_classes and benign_examples and malicious_examples:
            examples = self._balance_classes(malicious_examples, benign_examples)
        else:
            examples = malicious_examples + benign_examples

        # Apply max_samples limit
        if self.config.max_samples and len(examples) > self.config.max_samples:
            random.shuffle(examples)
            examples = examples[: self.config.max_samples]

        # Shuffle before splitting
        random.shuffle(examples)

        # Train/val split and save
        return self._save_split(examples, total_read)

    async def ingest_enriched(self) -> IngestionResult:
        """Async enriched-mode ingestion with LLM reasoning chains."""
        # First collect samples and direct examples
        samples_and_examples: List[tuple] = []
        total_read = 0

        for sample in self.adapter.iterate_samples():
            total_read += 1

            example = self.adapter.to_training_example(sample, ConversionMode.DIRECT)
            if example is None:
                continue

            samples_and_examples.append((sample, example))

            if self.config.max_samples and len(samples_and_examples) >= self.config.max_samples:
                break

        # Enrich in batches
        enriched_examples = await self._enrich_batch(samples_and_examples)

        random.shuffle(enriched_examples)
        return self._save_split(enriched_examples, total_read)

    async def _enrich_batch(
        self, items: List[tuple]
    ) -> List[TrainingExample]:
        """Call LLM to enrich examples with detailed reasoning."""
        import os

        api_key = self.config.enrichment_api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            # Fall back to direct mode
            return [ex for _, ex in items]

        semaphore = asyncio.Semaphore(self.config.enrichment_max_concurrent)
        results: List[TrainingExample] = []

        async with httpx.AsyncClient(timeout=120.0) as client:
            tasks = []
            for sample, example in items:
                tasks.append(
                    self._enrich_single(client, semaphore, api_key, sample, example)
                )

            completed = await asyncio.gather(*tasks, return_exceptions=True)
            for result in completed:
                if isinstance(result, TrainingExample):
                    results.append(result)
                elif isinstance(result, Exception):
                    # Log but continue
                    continue

        return results

    async def _enrich_single(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        api_key: str,
        sample: Dict[str, Any],
        example: TrainingExample,
    ) -> TrainingExample:
        """Enrich a single example via LLM."""
        async with semaphore:
            prompt = self.adapter.build_enrichment_prompt(sample)

            try:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.config.enrichment_model,
                        "max_tokens": 2048,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )

                if response.status_code == 200:
                    result = response.json()
                    enriched_text = result["content"][0]["text"]
                    return TrainingExample(
                        example_id=example.example_id,
                        system_prompt=example.system_prompt,
                        user_prompt=example.user_prompt,
                        assistant_response=enriched_text,
                        metadata={
                            **example.metadata,
                            "mode": "enriched",
                            "enrichment_model": self.config.enrichment_model,
                        },
                    )
            except Exception:
                pass

            # Fall back to direct response on failure
            return example

    def _balance_classes(
        self,
        malicious: List[TrainingExample],
        benign: List[TrainingExample],
    ) -> List[TrainingExample]:
        """Balance classes according to benign_ratio."""
        total_target = len(malicious) + len(benign)
        if self.config.max_samples:
            total_target = min(total_target, self.config.max_samples)

        benign_target = int(total_target * self.config.benign_ratio)
        malicious_target = total_target - benign_target

        # Oversample or undersample as needed
        if len(benign) > benign_target:
            random.shuffle(benign)
            benign = benign[:benign_target]
        if len(malicious) > malicious_target:
            random.shuffle(malicious)
            malicious = malicious[:malicious_target]

        return malicious + benign

    def _save_split(
        self, examples: List[TrainingExample], total_read: int
    ) -> IngestionResult:
        """Split examples into train/val and write JSONL files."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        split_idx = int(len(examples) * self.config.train_split)
        train_examples = examples[:split_idx]
        val_examples = examples[split_idx:]

        train_path = output_dir / "train.jsonl"
        val_path = output_dir / "val.jsonl"

        self._write_jsonl(train_path, train_examples)
        self._write_jsonl(val_path, val_examples)

        # Compute class distribution
        class_dist: Dict[str, int] = {}
        for ex in examples:
            domain = ex.metadata.get("domain", "unknown")
            class_dist[domain] = class_dist.get(domain, 0) + 1

        return IngestionResult(
            dataset_type=self.config.dataset_type.value,
            mode=self.config.mode.value,
            total_samples_read=total_read,
            examples_generated=len(examples),
            train_path=str(train_path),
            val_path=str(val_path),
            train_count=len(train_examples),
            val_count=len(val_examples),
            class_distribution=class_dist,
        )

    @staticmethod
    def _write_jsonl(path: Path, examples: List[TrainingExample]) -> None:
        """Write examples to a JSONL file in NeMo format."""
        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                # NeMo format: {"messages": [...]}
                record = {
                    "messages": [
                        {"role": "system", "content": ex.system_prompt},
                        {"role": "user", "content": ex.user_prompt},
                        {"role": "assistant", "content": ex.assistant_response},
                    ]
                }
                f.write(json.dumps(record) + "\n")
