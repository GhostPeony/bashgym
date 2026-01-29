"""
Benchmark data loader using HuggingFace datasets.

Downloads and caches benchmark datasets on first use.
Datasets are cached in ~/.cache/huggingface/datasets/
"""

from typing import List, Dict, Any, Optional
import logging
import json

logger = logging.getLogger(__name__)


class BenchmarkLoader:
    """Loads benchmark datasets from HuggingFace."""

    # Dataset mappings: benchmark_id -> (dataset_name, config/subset, split)
    DATASETS = {
        # Code generation
        "humaneval": ("openai_humaneval", None, "test"),
        "mbpp": ("google-research-datasets/mbpp", None, "test"),
        "bigcodebench": ("bigcode/bigcodebench", "default", "v0.1.2"),
        "ds1000": ("xlangai/DS-1000", None, "test"),
        # Function calling - BFCL uses custom loading
        "bfcl": None,  # Loaded via _load_bfcl()
        # Reasoning
        "gsm8k": ("openai/gsm8k", "main", "test"),
        "arc": ("allenai/ai2_arc", "ARC-Challenge", "test"),
        "hellaswag": ("Rowan/hellaswag", None, "validation"),
        # Safety
        "toxigen": ("toxigen/toxigen-data", "annotated", "test"),
        "bbq": ("walledai/BBQ", None, "test"),
        # Agentic
        "swe_bench": ("princeton-nlp/SWE-bench_Lite", None, "test"),
    }

    _cache: Dict[str, Any] = {}

    @classmethod
    def load(cls, benchmark_id: str, num_samples: Optional[int] = None) -> List[Dict]:
        """
        Load benchmark dataset, using cache if available.

        Args:
            benchmark_id: The benchmark identifier (e.g., 'humaneval', 'gsm8k')
            num_samples: Optional limit on number of samples to return

        Returns:
            List of sample dictionaries
        """
        if benchmark_id not in cls.DATASETS:
            logger.warning(f"Unknown benchmark: {benchmark_id}")
            return []

        # Check cache
        cache_key = benchmark_id
        if cache_key in cls._cache:
            data = cls._cache[cache_key]
            if num_samples:
                return data[:num_samples]
            return data

        # Special handling for BFCL (has inconsistent schemas)
        if benchmark_id == "bfcl":
            return cls._load_bfcl(num_samples)

        # Special handling for BBQ (uses category splits)
        if benchmark_id == "bbq":
            return cls._load_bbq(num_samples)

        # Load from HuggingFace
        dataset_info = cls.DATASETS[benchmark_id]
        if dataset_info is None:
            logger.warning(f"No dataset mapping for {benchmark_id}")
            return []

        dataset_name, config, preferred_split = dataset_info

        try:
            from datasets import load_dataset

            logger.info(f"Loading {benchmark_id} from HuggingFace ({dataset_name})...")

            if config:
                dataset = load_dataset(dataset_name, config)
            else:
                dataset = load_dataset(dataset_name)

            # Get the right split
            if preferred_split and preferred_split in dataset:
                data = list(dataset[preferred_split])
            elif "test" in dataset:
                data = list(dataset["test"])
            elif "validation" in dataset:
                data = list(dataset["validation"])
            elif "train" in dataset:
                data = list(dataset["train"])
            else:
                # Use first available split
                first_split = list(dataset.keys())[0]
                data = list(dataset[first_split])

            # Cache the full dataset
            cls._cache[cache_key] = data
            logger.info(f"Loaded {len(data)} samples for {benchmark_id}")

            if num_samples:
                return data[:num_samples]
            return data

        except ImportError:
            logger.error("HuggingFace datasets library not installed. Run: pip install datasets")
            return []
        except Exception as e:
            logger.error(f"Failed to load {benchmark_id}: {e}")
            return []

    @classmethod
    def _load_bfcl(cls, num_samples: Optional[int] = None) -> List[Dict]:
        """
        Load BFCL using direct file download to avoid schema conflicts.

        BFCL has inconsistent schemas across its JSON files, which causes
        load_dataset() to fail. We download specific consistent files directly.
        """
        cache_key = "bfcl"
        if cache_key in cls._cache:
            data = cls._cache[cache_key]
            return data[:num_samples] if num_samples else data

        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            logger.error("huggingface_hub not installed. Run: pip install huggingface_hub")
            return []

        # Download specific BFCL files with consistent schemas
        files_to_load = [
            "BFCL_v3_simple.json",
            "BFCL_v3_multiple.json",
            "BFCL_v3_parallel.json",
        ]

        all_data = []
        for filename in files_to_load:
            try:
                file_path = hf_hub_download(
                    repo_id="gorilla-llm/Berkeley-Function-Calling-Leaderboard",
                    filename=filename,
                    repo_type="dataset"
                )
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                all_data.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
                logger.info(f"Loaded {filename} for BFCL")
            except Exception as e:
                logger.warning(f"Could not load BFCL file {filename}: {e}")

        if not all_data:
            logger.error("Failed to load any BFCL data")
            return []

        cls._cache[cache_key] = all_data
        logger.info(f"Loaded {len(all_data)} samples for BFCL")

        return all_data[:num_samples] if num_samples else all_data

    @classmethod
    def _load_bbq(cls, num_samples: Optional[int] = None) -> List[Dict]:
        """
        Load BBQ dataset which uses category-based splits.

        BBQ tests for bias across different demographic categories.
        """
        cache_key = "bbq"
        if cache_key in cls._cache:
            data = cls._cache[cache_key]
            return data[:num_samples] if num_samples else data

        try:
            from datasets import load_dataset

            logger.info("Loading BBQ from HuggingFace...")

            # Load the dataset - BBQ uses category splits like "age", "gender", etc.
            # We'll load the main test split if available
            try:
                dataset = load_dataset("walledai/BBQ", split="test")
                data = list(dataset)
            except Exception:
                # Try loading with age category as fallback
                dataset = load_dataset("walledai/BBQ", "age")
                if "test" in dataset:
                    data = list(dataset["test"])
                elif "validation" in dataset:
                    data = list(dataset["validation"])
                else:
                    first_split = list(dataset.keys())[0]
                    data = list(dataset[first_split])

            cls._cache[cache_key] = data
            logger.info(f"Loaded {len(data)} samples for BBQ")

            return data[:num_samples] if num_samples else data

        except ImportError:
            logger.error("HuggingFace datasets library not installed.")
            return []
        except Exception as e:
            logger.error(f"Failed to load BBQ: {e}")
            return []

    @classmethod
    def get_dataset_info(cls, benchmark_id: str) -> Dict[str, Any]:
        """Get information about a benchmark dataset."""
        if benchmark_id not in cls.DATASETS:
            return {"error": f"Unknown benchmark: {benchmark_id}"}

        dataset_name, config, _ = cls.DATASETS[benchmark_id]

        info = {
            "benchmark_id": benchmark_id,
            "dataset_name": dataset_name,
            "config": config,
            "loaded": benchmark_id in cls._cache,
        }

        if benchmark_id in cls._cache:
            info["num_samples"] = len(cls._cache[benchmark_id])

        return info

    @classmethod
    def clear_cache(cls, benchmark_id: Optional[str] = None):
        """Clear cached datasets."""
        if benchmark_id:
            cls._cache.pop(benchmark_id, None)
        else:
            cls._cache.clear()

    @classmethod
    def is_available(cls, benchmark_id: str) -> bool:
        """Check if a benchmark is available."""
        # BFCL is available via custom loading even though DATASETS[bfcl] is None
        return benchmark_id in cls.DATASETS
