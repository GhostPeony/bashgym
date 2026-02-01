"""
Schema Builder for Enhanced Data Designer Integration

Provides a fluent API for building rich synthetic data schemas using
NVIDIA NeMo Data Designer column types (Sampler, LLM, Expression, Validator).

Module 3: Data Synthesis (The "Factory") - Enhanced Data Generation
"""

import os
import json
import asyncio
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union, Callable
from datetime import datetime, timezone
from enum import Enum
import httpx
import random


class ColumnType(Enum):
    """Data Designer column types."""
    # Sampler columns (deterministic/random)
    CATEGORY = "category"
    PERSON = "person"
    UUID = "uuid"
    DATETIME = "datetime"
    GAUSSIAN = "gaussian"
    POISSON = "poisson"
    BERNOULLI = "bernoulli"
    UNIFORM = "uniform"

    # LLM columns (generated)
    LLM = "llm"
    CODE = "code"
    STRUCTURED = "structured"

    # Expression columns (computed)
    EXPRESSION = "expression"
    JINJA = "jinja"

    # Validator columns
    PYTHON_VALIDATOR = "python_validator"
    SQL_VALIDATOR = "sql_validator"
    REMOTE_VALIDATOR = "remote_validator"


@dataclass
class ColumnDefinition:
    """Definition of a schema column."""

    name: str
    column_type: ColumnType
    params: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.column_type.value,
            "params": self.params,
            "depends_on": self.depends_on,
            "description": self.description
        }


@dataclass
class DataSchema:
    """A complete data generation schema."""

    name: str
    columns: List[ColumnDefinition] = field(default_factory=list)
    num_rows: int = 100
    seed: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "columns": [c.to_dict() for c in self.columns],
            "num_rows": self.num_rows,
            "seed": self.seed,
            "metadata": self.metadata
        }


class SchemaBuilder:
    """
    Fluent builder for Data Designer schemas.

    Supports:
    - Sampler columns: Category, Person, UUID, DateTime, Gaussian, etc.
    - LLM columns: Structured JSON output, code synthesis
    - Expression columns: Jinja2 templates, computed values
    - Validator columns: Python/SQL/Remote validation
    """

    def __init__(self, name: str):
        """Initialize schema builder."""
        self.schema = DataSchema(name=name)

    # ==================== Sampler Columns ====================

    def category(
        self,
        name: str,
        categories: List[str],
        weights: Optional[List[float]] = None,
        description: str = ""
    ) -> "SchemaBuilder":
        """Add a category column (randomly samples from list)."""
        self.schema.columns.append(ColumnDefinition(
            name=name,
            column_type=ColumnType.CATEGORY,
            params={
                "categories": categories,
                "weights": weights or [1.0 / len(categories)] * len(categories)
            },
            description=description
        ))
        return self

    def person(
        self,
        name: str,
        locale: str = "en_US",
        gender: Optional[str] = None,
        description: str = ""
    ) -> "SchemaBuilder":
        """Add a person name column."""
        self.schema.columns.append(ColumnDefinition(
            name=name,
            column_type=ColumnType.PERSON,
            params={"locale": locale, "gender": gender},
            description=description
        ))
        return self

    def uuid(self, name: str, description: str = "") -> "SchemaBuilder":
        """Add a UUID column."""
        self.schema.columns.append(ColumnDefinition(
            name=name,
            column_type=ColumnType.UUID,
            params={},
            description=description
        ))
        return self

    def datetime(
        self,
        name: str,
        start: str = "2020-01-01",
        end: str = "2024-12-31",
        format: str = "%Y-%m-%d %H:%M:%S",
        description: str = ""
    ) -> "SchemaBuilder":
        """Add a datetime column."""
        self.schema.columns.append(ColumnDefinition(
            name=name,
            column_type=ColumnType.DATETIME,
            params={"start": start, "end": end, "format": format},
            description=description
        ))
        return self

    def gaussian(
        self,
        name: str,
        mean: float = 0.0,
        std: float = 1.0,
        description: str = ""
    ) -> "SchemaBuilder":
        """Add a Gaussian (normal distribution) column."""
        self.schema.columns.append(ColumnDefinition(
            name=name,
            column_type=ColumnType.GAUSSIAN,
            params={"mean": mean, "std": std},
            description=description
        ))
        return self

    def poisson(
        self,
        name: str,
        lam: float = 5.0,
        description: str = ""
    ) -> "SchemaBuilder":
        """Add a Poisson distribution column."""
        self.schema.columns.append(ColumnDefinition(
            name=name,
            column_type=ColumnType.POISSON,
            params={"lambda": lam},
            description=description
        ))
        return self

    def bernoulli(
        self,
        name: str,
        p: float = 0.5,
        description: str = ""
    ) -> "SchemaBuilder":
        """Add a Bernoulli (boolean) column."""
        self.schema.columns.append(ColumnDefinition(
            name=name,
            column_type=ColumnType.BERNOULLI,
            params={"p": p},
            description=description
        ))
        return self

    def uniform(
        self,
        name: str,
        low: float = 0.0,
        high: float = 1.0,
        description: str = ""
    ) -> "SchemaBuilder":
        """Add a uniform distribution column."""
        self.schema.columns.append(ColumnDefinition(
            name=name,
            column_type=ColumnType.UNIFORM,
            params={"low": low, "high": high},
            description=description
        ))
        return self

    # ==================== LLM Columns ====================

    def llm(
        self,
        name: str,
        prompt: str,
        model: str = "meta/llama-3.1-70b-instruct",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        depends_on: Optional[List[str]] = None,
        description: str = ""
    ) -> "SchemaBuilder":
        """Add an LLM-generated column."""
        self.schema.columns.append(ColumnDefinition(
            name=name,
            column_type=ColumnType.LLM,
            params={
                "prompt": prompt,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens
            },
            depends_on=depends_on or [],
            description=description
        ))
        return self

    def code(
        self,
        name: str,
        task_description: str,
        language: str = "python",
        model: str = "qwen/qwen2.5-coder-72b-instruct",
        depends_on: Optional[List[str]] = None,
        description: str = ""
    ) -> "SchemaBuilder":
        """Add a code generation column."""
        prompt = f"Write {language} code for the following task:\n\n{{{{ {task_description} }}}}\n\nOutput only the code, no explanations."
        self.schema.columns.append(ColumnDefinition(
            name=name,
            column_type=ColumnType.CODE,
            params={
                "prompt": prompt,
                "model": model,
                "language": language,
                "temperature": 0.3,
                "max_tokens": 2048
            },
            depends_on=depends_on or [],
            description=description
        ))
        return self

    def structured(
        self,
        name: str,
        prompt: str,
        json_schema: Dict[str, Any],
        model: str = "meta/llama-3.1-70b-instruct",
        depends_on: Optional[List[str]] = None,
        description: str = ""
    ) -> "SchemaBuilder":
        """Add a structured JSON output column."""
        self.schema.columns.append(ColumnDefinition(
            name=name,
            column_type=ColumnType.STRUCTURED,
            params={
                "prompt": prompt,
                "json_schema": json_schema,
                "model": model,
                "temperature": 0.3
            },
            depends_on=depends_on or [],
            description=description
        ))
        return self

    # ==================== Expression Columns ====================

    def expression(
        self,
        name: str,
        expression: str,
        depends_on: List[str],
        description: str = ""
    ) -> "SchemaBuilder":
        """Add a computed expression column (Jinja2)."""
        self.schema.columns.append(ColumnDefinition(
            name=name,
            column_type=ColumnType.EXPRESSION,
            params={"expression": expression},
            depends_on=depends_on,
            description=description
        ))
        return self

    def jinja(
        self,
        name: str,
        template: str,
        depends_on: List[str],
        description: str = ""
    ) -> "SchemaBuilder":
        """Add a Jinja2 template column."""
        self.schema.columns.append(ColumnDefinition(
            name=name,
            column_type=ColumnType.JINJA,
            params={"template": template},
            depends_on=depends_on,
            description=description
        ))
        return self

    # ==================== Validator Columns ====================

    def python_validator(
        self,
        name: str,
        validation_code: str,
        depends_on: List[str],
        description: str = ""
    ) -> "SchemaBuilder":
        """Add a Python validation column."""
        self.schema.columns.append(ColumnDefinition(
            name=name,
            column_type=ColumnType.PYTHON_VALIDATOR,
            params={"code": validation_code},
            depends_on=depends_on,
            description=description
        ))
        return self

    # ==================== Build Methods ====================

    def with_rows(self, num_rows: int) -> "SchemaBuilder":
        """Set the number of rows to generate."""
        self.schema.num_rows = num_rows
        return self

    def with_seed(self, seed: int) -> "SchemaBuilder":
        """Set random seed for reproducibility."""
        self.schema.seed = seed
        return self

    def with_metadata(self, **kwargs) -> "SchemaBuilder":
        """Add metadata to the schema."""
        self.schema.metadata.update(kwargs)
        return self

    def build(self) -> DataSchema:
        """Build and return the schema."""
        return self.schema


class DataDesignerClient:
    """
    Client for NVIDIA NeMo Data Designer service.

    Generates synthetic data based on schemas.
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:8000",
        api_key: Optional[str] = None
    ):
        """Initialize the Data Designer client."""
        self.endpoint = endpoint
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY")

        self.client = httpx.AsyncClient(
            timeout=300.0,
            headers=self._build_headers()
        )

    def _build_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()

    async def generate(
        self,
        schema: DataSchema,
        output_path: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate synthetic data from a schema.

        Args:
            schema: The data schema to use
            output_path: Optional path to save results

        Returns:
            List of generated records
        """
        # Set seed for reproducibility
        if schema.seed:
            random.seed(schema.seed)

        records = []

        for i in range(schema.num_rows):
            record = {}

            for column in schema.columns:
                # Check dependencies
                for dep in column.depends_on:
                    if dep not in record:
                        raise ValueError(f"Column {column.name} depends on {dep} which hasn't been generated")

                # Generate value based on column type
                value = await self._generate_column_value(column, record, i)
                record[column.name] = value

            records.append(record)

        # Save if path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")

        return records

    async def _generate_column_value(
        self,
        column: ColumnDefinition,
        context: Dict[str, Any],
        row_index: int
    ) -> Any:
        """Generate a value for a column."""
        params = column.params

        # Sampler columns
        if column.column_type == ColumnType.CATEGORY:
            categories = params["categories"]
            weights = params.get("weights")
            return random.choices(categories, weights=weights)[0]

        elif column.column_type == ColumnType.PERSON:
            # Simplified - in production use faker library
            first_names = ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Jamie", "Riley", "Quinn"]
            last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis"]
            return f"{random.choice(first_names)} {random.choice(last_names)}"

        elif column.column_type == ColumnType.UUID:
            import uuid
            return str(uuid.uuid4())

        elif column.column_type == ColumnType.DATETIME:
            from datetime import datetime as dt
            start = dt.strptime(params["start"], "%Y-%m-%d")
            end = dt.strptime(params["end"], "%Y-%m-%d")
            delta = (end - start).days
            random_days = random.randint(0, delta)
            result_date = start + __import__("datetime").timedelta(days=random_days)
            return result_date.strftime(params.get("format", "%Y-%m-%d"))

        elif column.column_type == ColumnType.GAUSSIAN:
            return random.gauss(params["mean"], params["std"])

        elif column.column_type == ColumnType.POISSON:
            # Simplified Poisson
            lam = params["lambda"]
            return sum(random.random() < (lam / 100) for _ in range(100))

        elif column.column_type == ColumnType.BERNOULLI:
            return random.random() < params["p"]

        elif column.column_type == ColumnType.UNIFORM:
            return random.uniform(params["low"], params["high"])

        # LLM columns
        elif column.column_type in (ColumnType.LLM, ColumnType.CODE, ColumnType.STRUCTURED):
            return await self._generate_llm_value(column, context)

        # Expression columns
        elif column.column_type in (ColumnType.EXPRESSION, ColumnType.JINJA):
            return self._evaluate_expression(column, context)

        # Validator columns
        elif column.column_type == ColumnType.PYTHON_VALIDATOR:
            return self._run_validator(column, context)

        return None

    async def _generate_llm_value(
        self,
        column: ColumnDefinition,
        context: Dict[str, Any]
    ) -> str:
        """Generate a value using LLM."""
        params = column.params
        prompt = params["prompt"]

        # Substitute context variables using Jinja2
        try:
            from jinja2 import Environment
            prompt = Environment().from_string(prompt).render(**context)
        except ImportError:
            for key, value in context.items():
                prompt = prompt.replace(f"{{{{ {key} }}}}", str(value))

        try:
            response = await self.client.post(
                f"{self.endpoint}/v1/chat/completions",
                json={
                    "model": params.get("model", "meta/llama-3.1-70b-instruct"),
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": params.get("temperature", 0.7),
                    "max_tokens": params.get("max_tokens", 1024)
                }
            )

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]

        except Exception as e:
            return f"[Generation failed: {e}]"

        return "[Generation failed]"

    def _evaluate_expression(
        self,
        column: ColumnDefinition,
        context: Dict[str, Any]
    ) -> str:
        """Evaluate a Jinja2 expression."""
        template_str = column.params.get("expression") or column.params.get("template", "")

        try:
            from jinja2 import Environment
            env = Environment()
            template = env.from_string(template_str)
            return template.render(**context)
        except ImportError:
            # Fallback: simple variable substitution (no control flow)
            result = template_str
            for key, value in context.items():
                result = result.replace(f"{{{{ {key} }}}}", str(value))
            return result

    def _run_validator(
        self,
        column: ColumnDefinition,
        context: Dict[str, Any]
    ) -> bool:
        """Run Python validator code."""
        code = column.params.get("code", "True")

        try:
            # Create safe evaluation context
            safe_context = {"row": context, **context}
            return eval(code, {"__builtins__": {}}, safe_context)
        except Exception:
            return False


# ==================== Predefined Schema Templates ====================

def coding_task_schema(
    languages: List[str] = None,
    difficulties: List[str] = None,
    num_rows: int = 100
) -> DataSchema:
    """Create a schema for generating coding task datasets."""
    languages = languages or ["python", "javascript", "rust", "go"]
    difficulties = difficulties or ["easy", "medium", "hard"]

    return (
        SchemaBuilder("coding_tasks")
        .uuid("task_id", description="Unique task identifier")
        .category("language", languages, description="Programming language")
        .category("difficulty", difficulties, weights=[0.4, 0.4, 0.2])
        .category("task_type", [
            "function_implementation",
            "bug_fix",
            "code_optimization",
            "test_writing",
            "refactoring"
        ])
        .person("author", description="Task author name")
        .datetime("created_at", start="2024-01-01", end="2024-12-31")
        .llm(
            "task_description",
            prompt="""Generate a {{ difficulty }} {{ language }} coding task of type {{ task_type }}.

The task should be:
- Clear and specific
- Appropriately challenging for the {{ difficulty }} level
- Focused on a single concept

Output only the task description, starting with "Write a..."
""",
            depends_on=["language", "difficulty", "task_type"],
            description="The task description"
        )
        .code(
            "reference_solution",
            task_description="task_description",
            language="{{ language }}",
            depends_on=["task_description", "language"],
            description="Reference solution code"
        )
        .with_rows(num_rows)
        .build()
    )


def conversation_schema(
    personas: List[str] = None,
    topics: List[str] = None,
    num_rows: int = 100
) -> DataSchema:
    """Create a schema for generating conversation datasets."""
    personas = personas or [
        "helpful assistant",
        "expert programmer",
        "patient teacher",
        "curious learner"
    ]
    topics = topics or [
        "software development",
        "debugging",
        "code review",
        "architecture design"
    ]

    return (
        SchemaBuilder("conversations")
        .uuid("conversation_id")
        .category("persona", personas)
        .category("topic", topics)
        .gaussian("complexity", mean=5.0, std=2.0)
        .llm(
            "user_message",
            prompt="Generate a realistic user question about {{ topic }} that a {{ persona }} would answer. Complexity level: {{ complexity | int }}/10.",
            depends_on=["persona", "topic", "complexity"]
        )
        .llm(
            "assistant_response",
            prompt="As a {{ persona }}, provide a helpful response to: {{ user_message }}",
            depends_on=["persona", "user_message"]
        )
        .with_rows(num_rows)
        .build()
    )


async def main():
    """Example usage of the Schema Builder and Data Designer."""
    # Build a custom schema
    schema = (
        SchemaBuilder("agentic_tasks")
        .uuid("id")
        .category("task_type", ["file_operation", "git_command", "code_edit", "debugging"])
        .category("complexity", ["simple", "moderate", "complex"], weights=[0.3, 0.5, 0.2])
        .person("requester")
        .datetime("timestamp")
        .llm(
            "task_prompt",
            prompt="Generate a {{ complexity }} agentic coding task involving {{ task_type }}. Be specific and actionable.",
            depends_on=["task_type", "complexity"]
        )
        .expression(
            "full_prompt",
            expression="[{{ requester }}] {{ task_prompt }}",
            depends_on=["requester", "task_prompt"]
        )
        .with_rows(10)
        .with_seed(42)
        .build()
    )

    print("Schema Definition:")
    print(json.dumps(schema.to_dict(), indent=2))

    # Generate data
    client = DataDesignerClient()

    try:
        records = await client.generate(
            schema,
            output_path=Path("data/synthetic/agentic_tasks.jsonl")
        )

        print(f"\nGenerated {len(records)} records:")
        for record in records[:3]:
            print(json.dumps(record, indent=2))

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
