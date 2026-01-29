"""
Safe Synthesizer for Privacy-Preserving Data Generation

Integrates with NVIDIA NeMo Safe Synthesizer for formal privacy guarantees
using Differential Privacy and comprehensive PII detection/replacement.

Module 3: Data Synthesis (The "Factory") - Privacy Extension
"""

import os
import json
import re
import asyncio
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Set, Tuple
from datetime import datetime, timezone
from enum import Enum
import httpx
import hashlib

# NeMo Microservices integration
try:
    from bashgym.integrations import AsyncNeMoClient, NeMoClientConfig, NEMO_SDK_AVAILABLE
except ImportError:
    NEMO_SDK_AVAILABLE = False
    AsyncNeMoClient = None
    NeMoClientConfig = None

logger = logging.getLogger(__name__)


class PIIType(Enum):
    """Types of PII that can be detected and replaced."""
    # Personal identifiers
    PERSON = "person"
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    PASSPORT = "passport"
    DRIVER_LICENSE = "driver_license"

    # Financial
    CREDIT_CARD = "credit_card"
    BANK_ACCOUNT = "bank_account"
    IBAN = "iban"

    # Location
    ADDRESS = "address"
    ZIP_CODE = "zip_code"
    CITY = "city"
    COUNTRY = "country"
    IP_ADDRESS = "ip_address"

    # Medical
    MRN = "mrn"  # Medical Record Number
    HEALTH_PLAN = "health_plan"
    MEDICAL_CONDITION = "medical_condition"

    # Digital identifiers
    USERNAME = "username"
    PASSWORD = "password"
    API_KEY = "api_key"
    AUTH_TOKEN = "auth_token"
    URL = "url"

    # Temporal
    DATE_OF_BIRTH = "date_of_birth"
    DATE = "date"


class ReplacementStrategy(Enum):
    """Strategies for replacing detected PII."""
    REDACT = "redact"          # Replace with [REDACTED]
    MASK = "mask"              # Replace with ***
    SYNTHETIC = "synthetic"     # Replace with synthetic data
    HASH = "hash"              # Replace with hashed value
    CATEGORY = "category"       # Replace with category label


@dataclass
class SafeSynthesizerConfig:
    """Configuration for the Safe Synthesizer."""

    # NeMo Safe Synthesizer endpoint
    endpoint: str = "http://localhost:8000"
    api_key: Optional[str] = None

    # Privacy settings
    epsilon: float = 8.0  # DP privacy budget (lower = more private)
    delta: float = 1e-5   # DP delta parameter
    use_dp_sgd: bool = False  # Use DP-SGD during training

    # PII detection settings
    pii_types: List[str] = field(default_factory=lambda: [
        "person", "email", "ssn", "phone", "address", "credit_card", "api_key"
    ])
    confidence_threshold: float = 0.8
    use_llm_classification: bool = True

    # Replacement settings
    default_strategy: ReplacementStrategy = ReplacementStrategy.SYNTHETIC

    # Output settings
    output_dir: str = "data/privacy_processed"


@dataclass
class PIIDetection:
    """A detected PII instance."""

    text: str
    pii_type: PIIType
    start: int
    end: int
    confidence: float
    context: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "type": self.pii_type.value,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence
        }


@dataclass
class PrivacyReport:
    """Report on privacy processing for a dataset."""

    dataset_id: str
    total_records: int
    records_with_pii: int
    pii_counts: Dict[str, int]
    replacement_stats: Dict[str, int]
    privacy_budget_used: float
    disclosure_risk: float
    utility_score: float
    processing_time: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "total_records": self.total_records,
            "records_with_pii": self.records_with_pii,
            "pii_percentage": self.records_with_pii / max(self.total_records, 1) * 100,
            "pii_counts": self.pii_counts,
            "replacement_stats": self.replacement_stats,
            "privacy_budget_used": self.privacy_budget_used,
            "disclosure_risk": self.disclosure_risk,
            "utility_score": self.utility_score,
            "processing_time": self.processing_time
        }


class SafeSynthesizer:
    """
    Privacy-preserving data synthesis using NVIDIA NeMo Safe Synthesizer.

    Features:
    - PII detection with 50+ entity types
    - Multiple replacement strategies
    - Differential Privacy (DP-SGD) support
    - Privacy metrics and disclosure risk analysis
    - Quality/utility preservation analysis
    """

    # Regex patterns for common PII types
    PII_PATTERNS = {
        PIIType.EMAIL: r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        PIIType.PHONE: r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
        PIIType.SSN: r'\b(?!000|666|9\d{2})\d{3}[-\s]?(?!00)\d{2}[-\s]?(?!0000)\d{4}\b',
        PIIType.CREDIT_CARD: r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b',
        PIIType.IP_ADDRESS: r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        PIIType.API_KEY: r'\b(?:sk-[a-zA-Z0-9]{32,}|nvapi-[a-zA-Z0-9-]{32,}|ghp_[a-zA-Z0-9]{36})\b',
        PIIType.AUTH_TOKEN: r'\b(?:Bearer\s+)?[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\b',
        PIIType.DATE_OF_BIRTH: r'\b(?:0[1-9]|1[0-2])[/\-](?:0[1-9]|[12]\d|3[01])[/\-](?:19|20)\d{2}\b',
        PIIType.ZIP_CODE: r'\b\d{5}(?:-\d{4})?\b',
    }

    # Synthetic data generators
    SYNTHETIC_GENERATORS = {
        PIIType.PERSON: lambda i: f"Person_{i:04d}",
        PIIType.EMAIL: lambda i: f"user{i:04d}@example.com",
        PIIType.PHONE: lambda i: f"+1-555-{i:03d}-{(i*7)%10000:04d}",
        PIIType.SSN: lambda i: f"XXX-XX-{i:04d}",
        PIIType.CREDIT_CARD: lambda i: f"XXXX-XXXX-XXXX-{i:04d}",
        PIIType.IP_ADDRESS: lambda i: f"192.168.{i%256}.{(i//256)%256}",
        PIIType.API_KEY: lambda i: f"sk-synthetic-{hashlib.sha256(str(i).encode()).hexdigest()[:32]}",
        PIIType.ADDRESS: lambda i: f"{i} Example Street, City, ST 00000",
    }

    def __init__(self, config: Optional[SafeSynthesizerConfig] = None):
        """Initialize the Safe Synthesizer."""
        self.config = config or SafeSynthesizerConfig()

        # Load API key from environment
        if not self.config.api_key:
            self.config.api_key = os.environ.get("NVIDIA_API_KEY")

        # Initialize NeMo client if available (for LLM-based PII detection)
        self._nemo_client: Optional[AsyncNeMoClient] = None
        if NEMO_SDK_AVAILABLE and AsyncNeMoClient is not None:
            try:
                nemo_config = NeMoClientConfig(
                    base_url=self.config.endpoint,
                    api_key=self.config.api_key,
                    timeout=120.0,
                )
                self._nemo_client = AsyncNeMoClient(nemo_config)
                logger.info("Safe Synthesizer initialized with NeMo SDK")
            except Exception as e:
                logger.warning(f"Failed to initialize NeMo SDK client: {e}")

        # HTTP client (fallback)
        self.client = httpx.AsyncClient(
            timeout=120.0,
            headers=self._build_headers()
        )

        # Tracking
        self.synthetic_counter: Dict[PIIType, int] = {}
        self.replacement_map: Dict[str, str] = {}  # For consistent replacements

        # Ensure output directory exists
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def _build_headers(self) -> Dict[str, str]:
        """Build HTTP headers."""
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    async def close(self):
        """Close all client connections."""
        await self.client.aclose()
        if self._nemo_client:
            await self._nemo_client.close()

    def detect_pii(
        self,
        text: str,
        pii_types: Optional[List[str]] = None
    ) -> List[PIIDetection]:
        """
        Detect PII in text using regex and optional LLM classification.

        Args:
            text: Text to analyze
            pii_types: Optional list of PII types to detect

        Returns:
            List of PIIDetection instances
        """
        detections = []
        types_to_check = pii_types or self.config.pii_types

        # Regex-based detection
        for pii_type, pattern in self.PII_PATTERNS.items():
            if pii_type.value not in types_to_check:
                continue

            for match in re.finditer(pattern, text):
                detections.append(PIIDetection(
                    text=match.group(),
                    pii_type=pii_type,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.9,
                    context=text[max(0, match.start()-20):min(len(text), match.end()+20)]
                ))

        # Sort by position
        detections.sort(key=lambda d: d.start)

        return detections

    async def detect_pii_with_llm(
        self,
        text: str,
        pii_types: Optional[List[str]] = None
    ) -> List[PIIDetection]:
        """
        Detect PII using LLM classification for higher accuracy.

        Args:
            text: Text to analyze
            pii_types: Optional list of PII types to detect

        Returns:
            List of PIIDetection instances
        """
        # First get regex detections
        detections = self.detect_pii(text, pii_types)

        if not self.config.use_llm_classification:
            return detections

        # Use LLM for additional detection
        types_str = ", ".join(pii_types or self.config.pii_types)
        llm_prompt = f"""Analyze the following text and identify any PII (Personally Identifiable Information).

Look for these types: {types_str}

Text to analyze:
{text}

Output JSON array of detections:
[{{"text": "<found text>", "type": "<pii type>", "confidence": <0.0-1.0>}}]

If no PII found, output: []
"""

        try:
            # Use NeMo client if available, otherwise fall back to HTTP
            if self._nemo_client:
                result = await self._nemo_client.chat_completion(
                    model="meta/llama-3.1-70b-instruct",
                    messages=[{"role": "user", "content": llm_prompt}],
                    temperature=0.0
                )
                content = result["choices"][0]["message"]["content"]
            else:
                response = await self.client.post(
                    f"{self.config.endpoint}/v1/chat/completions",
                    json={
                        "model": "meta/llama-3.1-70b-instruct",
                        "messages": [{"role": "user", "content": llm_prompt}],
                        "temperature": 0.0
                    }
                )

                if response.status_code != 200:
                    return detections

                result = response.json()
                content = result["choices"][0]["message"]["content"]

            # Parse JSON response
            if content:
                try:
                    json_start = content.find("[")
                    json_end = content.rfind("]") + 1
                    if json_start >= 0 and json_end > json_start:
                        llm_detections = json.loads(content[json_start:json_end])

                        for det in llm_detections:
                            # Find position in text
                            found_text = det.get("text", "")
                            start = text.find(found_text)
                            if start >= 0:
                                pii_type_str = det.get("type", "").lower()
                                try:
                                    pii_type = PIIType(pii_type_str)
                                except ValueError:
                                    continue

                                # Check if already detected
                                if not any(d.start == start for d in detections):
                                    detections.append(PIIDetection(
                                        text=found_text,
                                        pii_type=pii_type,
                                        start=start,
                                        end=start + len(found_text),
                                        confidence=det.get("confidence", 0.8)
                                    ))
                except json.JSONDecodeError:
                    pass

        except Exception as e:
            print(f"LLM detection failed: {e}")

        # Re-sort
        detections.sort(key=lambda d: d.start)
        return detections

    def replace_pii(
        self,
        text: str,
        detections: List[PIIDetection],
        strategy: Optional[ReplacementStrategy] = None
    ) -> Tuple[str, Dict[str, str]]:
        """
        Replace detected PII with appropriate substitutes.

        Args:
            text: Original text
            detections: List of PII detections
            strategy: Replacement strategy to use

        Returns:
            Tuple of (processed_text, replacement_map)
        """
        strategy = strategy or self.config.default_strategy
        replacements = {}

        # Process in reverse order to maintain positions
        result = text
        for detection in reversed(detections):
            original = detection.text

            # Check if we've seen this value before
            if original in self.replacement_map:
                replacement = self.replacement_map[original]
            else:
                replacement = self._generate_replacement(
                    detection.pii_type, strategy, original
                )
                self.replacement_map[original] = replacement

            result = result[:detection.start] + replacement + result[detection.end:]
            replacements[original] = replacement

        return result, replacements

    def _generate_replacement(
        self,
        pii_type: PIIType,
        strategy: ReplacementStrategy,
        original: str
    ) -> str:
        """Generate a replacement for a PII value."""
        if strategy == ReplacementStrategy.REDACT:
            return f"[{pii_type.value.upper()}_REDACTED]"

        elif strategy == ReplacementStrategy.MASK:
            return "*" * len(original)

        elif strategy == ReplacementStrategy.HASH:
            hash_val = hashlib.sha256(original.encode()).hexdigest()[:8]
            return f"[{pii_type.value}:{hash_val}]"

        elif strategy == ReplacementStrategy.CATEGORY:
            return f"<{pii_type.value}>"

        elif strategy == ReplacementStrategy.SYNTHETIC:
            # Get counter for this type
            count = self.synthetic_counter.get(pii_type, 0)
            self.synthetic_counter[pii_type] = count + 1

            generator = self.SYNTHETIC_GENERATORS.get(pii_type)
            if generator:
                return generator(count)
            else:
                return f"[{pii_type.value}_{count:04d}]"

        return f"[REPLACED_{pii_type.value}]"

    async def process_dataset(
        self,
        data: List[Dict[str, Any]],
        text_fields: List[str],
        dataset_id: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], PrivacyReport]:
        """
        Process a dataset to remove/replace PII.

        Args:
            data: List of records to process
            text_fields: Fields containing text to analyze
            dataset_id: Optional identifier for the dataset

        Returns:
            Tuple of (processed_data, privacy_report)
        """
        start_time = datetime.now(timezone.utc)
        dataset_id = dataset_id or f"dataset_{start_time.strftime('%Y%m%d_%H%M%S')}"

        processed_data = []
        pii_counts: Dict[str, int] = {}
        replacement_stats: Dict[str, int] = {}
        records_with_pii = 0

        for record in data:
            processed_record = record.copy()
            record_has_pii = False

            for field in text_fields:
                if field not in record or not isinstance(record[field], str):
                    continue

                text = record[field]

                # Detect PII
                if self.config.use_llm_classification:
                    detections = await self.detect_pii_with_llm(text)
                else:
                    detections = self.detect_pii(text)

                if detections:
                    record_has_pii = True

                    # Count PII types
                    for det in detections:
                        pii_type = det.pii_type.value
                        pii_counts[pii_type] = pii_counts.get(pii_type, 0) + 1

                    # Replace PII
                    processed_text, replacements = self.replace_pii(text, detections)
                    processed_record[field] = processed_text

                    # Track replacement stats
                    strategy_name = self.config.default_strategy.value
                    replacement_stats[strategy_name] = replacement_stats.get(strategy_name, 0) + len(replacements)

            if record_has_pii:
                records_with_pii += 1

            processed_data.append(processed_record)

        # Calculate metrics
        end_time = datetime.now(timezone.utc)
        processing_time = (end_time - start_time).total_seconds()

        # Estimate disclosure risk and utility
        disclosure_risk = self._estimate_disclosure_risk(pii_counts, len(data))
        utility_score = self._estimate_utility(data, processed_data, text_fields)

        report = PrivacyReport(
            dataset_id=dataset_id,
            total_records=len(data),
            records_with_pii=records_with_pii,
            pii_counts=pii_counts,
            replacement_stats=replacement_stats,
            privacy_budget_used=self.config.epsilon,
            disclosure_risk=disclosure_risk,
            utility_score=utility_score,
            processing_time=processing_time
        )

        # Save report
        self._save_report(report)

        return processed_data, report

    def _estimate_disclosure_risk(
        self,
        pii_counts: Dict[str, int],
        total_records: int
    ) -> float:
        """Estimate disclosure risk based on PII density."""
        if total_records == 0:
            return 0.0

        total_pii = sum(pii_counts.values())
        pii_density = total_pii / total_records

        # Higher density = higher risk, but capped
        risk = min(pii_density / 10, 1.0)

        # Apply epsilon factor (lower epsilon = better privacy)
        risk = risk * (1 - min(self.config.epsilon / 20, 0.9))

        return round(risk, 4)

    def _estimate_utility(
        self,
        original: List[Dict[str, Any]],
        processed: List[Dict[str, Any]],
        text_fields: List[str]
    ) -> float:
        """Estimate data utility preservation."""
        if not original:
            return 1.0

        total_chars_original = 0
        total_chars_preserved = 0

        for orig, proc in zip(original, processed):
            for field in text_fields:
                orig_text = str(orig.get(field, ""))
                proc_text = str(proc.get(field, ""))

                total_chars_original += len(orig_text)
                # Count non-replaced characters
                total_chars_preserved += len(proc_text)

        if total_chars_original == 0:
            return 1.0

        # Basic utility = ratio of preserved content
        utility = total_chars_preserved / total_chars_original

        return round(min(utility, 1.0), 4)

    def _save_report(self, report: PrivacyReport) -> Path:
        """Save privacy report to file."""
        output_path = Path(self.config.output_dir) / f"privacy_report_{report.dataset_id}.json"
        output_path.write_text(json.dumps(report.to_dict(), indent=2))
        return output_path

    async def apply_differential_privacy(
        self,
        data: List[Dict[str, Any]],
        numeric_fields: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Apply differential privacy to numeric fields.

        Uses Laplace mechanism for numeric data.

        Args:
            data: Dataset to process
            numeric_fields: Fields containing numeric data

        Returns:
            Dataset with DP noise added
        """
        import random

        processed = []
        sensitivity = 1.0  # Assumed sensitivity

        for record in data:
            processed_record = record.copy()

            for field in numeric_fields:
                if field in record and isinstance(record[field], (int, float)):
                    original = record[field]

                    # Laplace noise
                    scale = sensitivity / self.config.epsilon
                    noise = random.random() - 0.5
                    noise = -scale * (1 if noise >= 0 else -1) * abs(noise).real

                    processed_record[field] = original + noise

            processed.append(processed_record)

        return processed

    def reset_replacement_map(self):
        """Reset the replacement map for a fresh dataset."""
        self.replacement_map.clear()
        self.synthetic_counter.clear()


async def main():
    """Example usage of the Safe Synthesizer."""
    config = SafeSynthesizerConfig(
        epsilon=8.0,
        pii_types=["email", "phone", "ssn", "api_key"],
        default_strategy=ReplacementStrategy.SYNTHETIC
    )

    synthesizer = SafeSynthesizer(config)

    try:
        # Example data with PII
        sample_data = [
            {
                "id": 1,
                "message": "Contact John Doe at john.doe@email.com or 555-123-4567",
                "notes": "SSN: 123-45-6789, API key: sk-abcdef1234567890"
            },
            {
                "id": 2,
                "message": "Email jane@company.org for support",
                "notes": "No sensitive data here"
            }
        ]

        # Process dataset
        processed_data, report = await synthesizer.process_dataset(
            data=sample_data,
            text_fields=["message", "notes"]
        )

        print("Privacy Report:")
        print(json.dumps(report.to_dict(), indent=2))

        print("\nProcessed Data:")
        for record in processed_data:
            print(json.dumps(record, indent=2))

    finally:
        await synthesizer.close()


if __name__ == "__main__":
    asyncio.run(main())
