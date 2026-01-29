"""
Model Router for The Gym Layer

Routes requests between Teacher (Claude) and Student (fine-tuned SLM) models.
Implements progressive handoff as student models improve.

Instrumentation:
  - Guardrails check student outputs before returning
  - Profiler tracks student vs teacher performance
  - Blocked student responses trigger teacher fallback

Module 4: Training (The "Gym")
"""

import os
import json
import asyncio
import httpx
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, Union, TYPE_CHECKING
from datetime import datetime, timezone
from enum import Enum
import random

# Import instrumentation (optional)
try:
    from bashgym.core import get_instrumentation, Instrumentation
    HAS_INSTRUMENTATION = True
except ImportError:
    HAS_INSTRUMENTATION = False
    Instrumentation = None

if TYPE_CHECKING:
    from bashgym.core import Instrumentation


class ModelType(Enum):
    """Types of models in the system."""
    TEACHER = "teacher"      # Claude (high capability)
    STUDENT = "student"      # Fine-tuned SLM
    HYBRID = "hybrid"        # Mix of both


class RoutingStrategy(Enum):
    """Strategies for routing requests."""
    TEACHER_ONLY = "teacher_only"
    STUDENT_ONLY = "student_only"
    CONFIDENCE_BASED = "confidence_based"
    TASK_COMPLEXITY = "task_complexity"
    PROGRESSIVE = "progressive"
    RANDOM_SAMPLE = "random_sample"


@dataclass
class ModelConfig:
    """Configuration for a model."""
    
    name: str
    model_type: ModelType
    endpoint: str
    api_key: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    
    # Performance metrics
    avg_latency_ms: float = 0.0
    success_rate: float = 1.0
    total_requests: int = 0


@dataclass
class RouterConfig:
    """Configuration for the model router."""

    # Routing settings
    strategy: RoutingStrategy = RoutingStrategy.CONFIDENCE_BASED
    confidence_threshold: float = 0.7
    complexity_threshold: float = 0.5

    # Progressive handoff settings
    student_sample_rate: float = 0.1  # Start with 10% to student
    max_student_rate: float = 0.9     # Cap at 90% to student
    improvement_increment: float = 0.05

    # Fallback settings
    fallback_to_teacher: bool = True
    max_retries: int = 2

    # Logging
    log_routing_decisions: bool = True
    metrics_dir: str = "data/routing_metrics"

    # Instrumentation
    enable_guardrails: bool = True
    enable_profiling: bool = True
    fallback_on_guardrail_block: bool = True  # Fallback to teacher if student blocked


@dataclass
class RoutingDecision:
    """A routing decision with metadata."""
    
    request_id: str
    selected_model: str
    model_type: ModelType
    strategy_used: RoutingStrategy
    confidence: float
    task_complexity: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "selected_model": self.selected_model,
            "model_type": self.model_type.value,
            "strategy": self.strategy_used.value,
            "confidence": self.confidence,
            "task_complexity": self.task_complexity,
            "timestamp": self.timestamp
        }


@dataclass
class ModelResponse:
    """Response from a model."""

    content: str
    model_name: str
    model_type: ModelType
    latency_ms: float
    tokens_used: int
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Instrumentation fields
    blocked_by_guardrails: bool = False
    fell_back_to_teacher: bool = False
    source: str = ""  # "student" or "teacher" for final source


class ModelRouter:
    """
    Routes requests between Teacher and Student models.
    
    Features:
    - Multiple routing strategies
    - Progressive handoff as student improves
    - Automatic fallback to teacher
    - Performance tracking
    """
    
    def __init__(
        self,
        config: Optional[RouterConfig] = None,
        instrumentation: Optional["Instrumentation"] = None
    ):
        """Initialize the model router."""
        self.config = config or RouterConfig()
        self.models: Dict[str, ModelConfig] = {}
        self.routing_history: List[RoutingDecision] = []
        self.current_student_rate = self.config.student_sample_rate

        # Instrumentation (guardrails + profiling)
        self._instrumentation = instrumentation
        if self._instrumentation is None and HAS_INSTRUMENTATION:
            if self.config.enable_guardrails or self.config.enable_profiling:
                self._instrumentation = get_instrumentation()

        # HTTP client for API calls
        self.client = httpx.AsyncClient(timeout=120.0)

        # Ensure metrics directory exists
        Path(self.config.metrics_dir).mkdir(parents=True, exist_ok=True)

        # Load default models from environment
        self._load_default_models()

    @property
    def instrumentation(self) -> Optional["Instrumentation"]:
        """Get the instrumentation instance."""
        return self._instrumentation

    def get_guardrail_events(self) -> List[Dict[str, Any]]:
        """Get guardrail events from routing."""
        if self._instrumentation:
            return [e.to_dict() for e in self._instrumentation.get_guardrail_events(model_source="student")]
        return []
    
    def _load_default_models(self) -> None:
        """Load default model configurations from environment."""
        # Teacher model (Claude)
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        if anthropic_key:
            self.register_model(ModelConfig(
                name="claude-sonnet",
                model_type=ModelType.TEACHER,
                endpoint="https://api.anthropic.com/v1/messages",
                api_key=anthropic_key,
                max_tokens=8192,
                temperature=0.7
            ))
        
        # Student model (NVIDIA NIM or local)
        nvidia_key = os.environ.get("NVIDIA_API_KEY")
        if nvidia_key:
            self.register_model(ModelConfig(
                name="llama-student",
                model_type=ModelType.STUDENT,
                endpoint="https://integrate.api.nvidia.com/v1/chat/completions",
                api_key=nvidia_key,
                max_tokens=4096,
                temperature=0.7
            ))
    
    def register_model(self, model_config: ModelConfig) -> None:
        """Register a model with the router."""
        self.models[model_config.name] = model_config
    
    def get_teacher_model(self) -> Optional[ModelConfig]:
        """Get the primary teacher model."""
        for model in self.models.values():
            if model.model_type == ModelType.TEACHER:
                return model
        return None
    
    def get_student_model(self) -> Optional[ModelConfig]:
        """Get the primary student model."""
        for model in self.models.values():
            if model.model_type == ModelType.STUDENT:
                return model
        return None
    
    def route(
        self,
        prompt: str,
        request_id: Optional[str] = None,
        task_complexity: Optional[float] = None,
        force_model: Optional[str] = None
    ) -> RoutingDecision:
        """
        Decide which model to route a request to.
        
        Args:
            prompt: The input prompt
            request_id: Optional request identifier
            task_complexity: Optional pre-computed complexity (0-1)
            force_model: Force routing to a specific model
            
        Returns:
            RoutingDecision with selected model
        """
        request_id = request_id or f"req_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        
        # Force specific model if requested
        if force_model and force_model in self.models:
            model = self.models[force_model]
            decision = RoutingDecision(
                request_id=request_id,
                selected_model=force_model,
                model_type=model.model_type,
                strategy_used=RoutingStrategy.TEACHER_ONLY,
                confidence=1.0,
                task_complexity=task_complexity or 0.5
            )
            self._log_decision(decision)
            return decision
        
        # Calculate task complexity if not provided
        if task_complexity is None:
            task_complexity = self._estimate_complexity(prompt)
        
        # Route based on strategy
        if self.config.strategy == RoutingStrategy.TEACHER_ONLY:
            selected = self.get_teacher_model()
            confidence = 1.0
        elif self.config.strategy == RoutingStrategy.STUDENT_ONLY:
            selected = self.get_student_model()
            confidence = 1.0
        elif self.config.strategy == RoutingStrategy.CONFIDENCE_BASED:
            selected, confidence = self._route_by_confidence(prompt)
        elif self.config.strategy == RoutingStrategy.TASK_COMPLEXITY:
            selected, confidence = self._route_by_complexity(task_complexity)
        elif self.config.strategy == RoutingStrategy.PROGRESSIVE:
            selected, confidence = self._route_progressive()
        elif self.config.strategy == RoutingStrategy.RANDOM_SAMPLE:
            selected, confidence = self._route_random()
        else:
            selected = self.get_teacher_model()
            confidence = 1.0
        
        # Fallback to teacher if no model selected
        if selected is None:
            selected = self.get_teacher_model()
            if selected is None:
                raise ValueError("No models available for routing")
        
        decision = RoutingDecision(
            request_id=request_id,
            selected_model=selected.name,
            model_type=selected.model_type,
            strategy_used=self.config.strategy,
            confidence=confidence,
            task_complexity=task_complexity
        )
        
        self._log_decision(decision)
        return decision
    
    def _estimate_complexity(self, prompt: str) -> float:
        """
        Estimate task complexity from the prompt.
        
        Simple heuristic based on:
        - Prompt length
        - Presence of technical terms
        - Multi-step indicators
        """
        score = 0.0
        
        # Length factor
        word_count = len(prompt.split())
        if word_count > 200:
            score += 0.3
        elif word_count > 100:
            score += 0.2
        elif word_count > 50:
            score += 0.1
        
        # Technical terms
        technical_terms = [
            "refactor", "optimize", "debug", "architecture",
            "async", "concurrent", "distributed", "scale",
            "security", "authentication", "encryption",
            "database", "migration", "deploy", "kubernetes"
        ]
        prompt_lower = prompt.lower()
        tech_count = sum(1 for term in technical_terms if term in prompt_lower)
        score += min(tech_count * 0.1, 0.4)
        
        # Multi-step indicators
        multi_step_terms = ["first", "then", "after", "finally", "step", "multiple"]
        if any(term in prompt_lower for term in multi_step_terms):
            score += 0.2
        
        return min(score, 1.0)
    
    def _route_by_confidence(
        self,
        prompt: str
    ) -> tuple[Optional[ModelConfig], float]:
        """Route based on student model confidence."""
        student = self.get_student_model()
        teacher = self.get_teacher_model()
        
        if not student:
            return teacher, 1.0
        
        # Estimate student confidence based on historical performance
        confidence = student.success_rate
        
        if confidence >= self.config.confidence_threshold:
            return student, confidence
        else:
            return teacher, 1.0
    
    def _route_by_complexity(
        self,
        complexity: float
    ) -> tuple[Optional[ModelConfig], float]:
        """Route based on task complexity."""
        student = self.get_student_model()
        teacher = self.get_teacher_model()
        
        if complexity <= self.config.complexity_threshold:
            # Simple task - use student
            return student or teacher, 1.0 - complexity
        else:
            # Complex task - use teacher
            return teacher, complexity
    
    def _route_progressive(self) -> tuple[Optional[ModelConfig], float]:
        """Progressive routing with increasing student usage."""
        student = self.get_student_model()
        teacher = self.get_teacher_model()
        
        if not student:
            return teacher, 1.0
        
        # Use current student rate for sampling
        if random.random() < self.current_student_rate:
            return student, self.current_student_rate
        else:
            return teacher, 1.0 - self.current_student_rate
    
    def _route_random(self) -> tuple[Optional[ModelConfig], float]:
        """Random sampling between models."""
        student = self.get_student_model()
        teacher = self.get_teacher_model()
        
        if not student:
            return teacher, 1.0
        
        if random.random() < 0.5:
            return student, 0.5
        else:
            return teacher, 0.5
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        request_id: Optional[str] = None,
        **kwargs
    ) -> ModelResponse:
        """
        Generate a response using the routed model with instrumentation.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            request_id: Optional request identifier
            **kwargs: Additional generation parameters

        Returns:
            ModelResponse with generated content
        """
        # Start profiler trace for this request
        trace_id = ""
        if self._instrumentation and self.config.enable_profiling:
            trace_id = self._instrumentation.start_trace(
                f"router:{request_id or 'unnamed'}",
                metadata={"prompt_preview": prompt[:100]}
            )

        try:
            # Check input with guardrails
            if self._instrumentation and self.config.enable_guardrails:
                async with self._instrumentation.instrument_input(
                    prompt,
                    location="router.input"
                ) as ctx:
                    if not ctx.allowed:
                        return ModelResponse(
                            content="Request blocked: input failed safety check",
                            model_name="none",
                            model_type=ModelType.TEACHER,
                            latency_ms=0,
                            tokens_used=0,
                            success=False,
                            error_message="Input blocked by guardrails",
                            blocked_by_guardrails=True,
                            source="blocked"
                        )
                    prompt = ctx.content  # Use potentially filtered prompt

            # Get routing decision
            decision = self.route(prompt, request_id)
            model = self.models[decision.selected_model]

            start_time = datetime.now()
            fell_back = False
            blocked = False

            try:
                if model.model_type == ModelType.TEACHER:
                    response = await self._call_anthropic(model, prompt, system_prompt, **kwargs)
                    response.source = "teacher"
                else:
                    response = await self._call_openai_compatible(model, prompt, system_prompt, **kwargs)
                    response.source = "student"

                    # Check student output with guardrails
                    if self._instrumentation and self.config.enable_guardrails and response.success:
                        async with self._instrumentation.instrument_output(
                            response.content,
                            location="router.student_output",
                            model_source="student"
                        ) as ctx:
                            if not ctx.allowed:
                                # Student output blocked - fallback to teacher
                                blocked = True
                                if self.config.fallback_on_guardrail_block:
                                    teacher = self.get_teacher_model()
                                    if teacher:
                                        teacher_response = await self._call_anthropic(
                                            teacher, prompt, system_prompt, **kwargs
                                        )
                                        teacher_response.source = "teacher"
                                        teacher_response.fell_back_to_teacher = True
                                        teacher_response.blocked_by_guardrails = True
                                        fell_back = True

                                        # Record profiler metrics
                                        if self._instrumentation:
                                            self._instrumentation.record_llm_call(
                                                model=teacher.name,
                                                prompt=prompt[:200],
                                                response=teacher_response.content[:200],
                                                input_tokens=0,
                                                output_tokens=teacher_response.tokens_used,
                                                latency_ms=teacher_response.latency_ms,
                                                model_source="teacher",
                                                fallback=True
                                            )

                                        return teacher_response
                                else:
                                    # Return blocked response
                                    return ModelResponse(
                                        content="Response blocked: output failed safety check",
                                        model_name=model.name,
                                        model_type=model.model_type,
                                        latency_ms=response.latency_ms,
                                        tokens_used=response.tokens_used,
                                        success=False,
                                        error_message="Output blocked by guardrails",
                                        blocked_by_guardrails=True,
                                        source="student"
                                    )
                            else:
                                # Apply any PII filtering
                                response.content = ctx.content

                # Update model metrics
                latency = (datetime.now() - start_time).total_seconds() * 1000
                self._update_model_metrics(model.name, latency, True)

                # Record profiler metrics
                if self._instrumentation and self.config.enable_profiling:
                    self._instrumentation.record_llm_call(
                        model=model.name,
                        prompt=prompt[:200],
                        response=response.content[:200],
                        input_tokens=0,
                        output_tokens=response.tokens_used,
                        latency_ms=response.latency_ms,
                        model_source=response.source
                    )

                return response

            except Exception as e:
                # Update metrics for failure
                latency = (datetime.now() - start_time).total_seconds() * 1000
                self._update_model_metrics(model.name, latency, False)

                # Fallback to teacher if enabled
                if self.config.fallback_to_teacher and model.model_type == ModelType.STUDENT:
                    teacher = self.get_teacher_model()
                    if teacher:
                        teacher_response = await self._call_anthropic(teacher, prompt, system_prompt, **kwargs)
                        teacher_response.source = "teacher"
                        teacher_response.fell_back_to_teacher = True
                        return teacher_response

                return ModelResponse(
                    content="",
                    model_name=model.name,
                    model_type=model.model_type,
                    latency_ms=latency,
                    tokens_used=0,
                    success=False,
                    error_message=str(e),
                    source=model.model_type.value
                )

        finally:
            # End profiler trace
            if trace_id and self._instrumentation:
                self._instrumentation.end_trace(trace_id)
    
    async def _call_anthropic(
        self,
        model: ModelConfig,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> ModelResponse:
        """Call Anthropic API."""
        start_time = datetime.now()
        
        messages = [{"role": "user", "content": prompt}]
        
        payload = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": kwargs.get("max_tokens", model.max_tokens),
            "messages": messages
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        response = await self.client.post(
            model.endpoint,
            headers={
                "x-api-key": model.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            },
            json=payload
        )
        
        latency = (datetime.now() - start_time).total_seconds() * 1000
        
        if response.status_code == 200:
            data = response.json()
            content = data["content"][0]["text"]
            tokens = data.get("usage", {}).get("output_tokens", 0)
            
            return ModelResponse(
                content=content,
                model_name=model.name,
                model_type=model.model_type,
                latency_ms=latency,
                tokens_used=tokens,
                success=True
            )
        else:
            raise Exception(f"API error: {response.status_code} - {response.text}")
    
    async def _call_openai_compatible(
        self,
        model: ModelConfig,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> ModelResponse:
        """Call OpenAI-compatible API (NVIDIA NIM, local models)."""
        start_time = datetime.now()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": kwargs.get("model_name", "meta/llama-3.1-8b-instruct"),
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", model.max_tokens),
            "temperature": kwargs.get("temperature", model.temperature)
        }
        
        headers = {"Content-Type": "application/json"}
        if model.api_key:
            headers["Authorization"] = f"Bearer {model.api_key}"
        
        response = await self.client.post(
            model.endpoint,
            headers=headers,
            json=payload
        )
        
        latency = (datetime.now() - start_time).total_seconds() * 1000
        
        if response.status_code == 200:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            tokens = data.get("usage", {}).get("completion_tokens", 0)
            
            return ModelResponse(
                content=content,
                model_name=model.name,
                model_type=model.model_type,
                latency_ms=latency,
                tokens_used=tokens,
                success=True
            )
        else:
            raise Exception(f"API error: {response.status_code} - {response.text}")
    
    def _update_model_metrics(
        self,
        model_name: str,
        latency_ms: float,
        success: bool
    ) -> None:
        """Update model performance metrics."""
        if model_name not in self.models:
            return
        
        model = self.models[model_name]
        model.total_requests += 1
        
        # Update rolling average latency
        model.avg_latency_ms = (
            (model.avg_latency_ms * (model.total_requests - 1) + latency_ms)
            / model.total_requests
        )
        
        # Update success rate
        if success:
            model.success_rate = (
                (model.success_rate * (model.total_requests - 1) + 1.0)
                / model.total_requests
            )
        else:
            model.success_rate = (
                (model.success_rate * (model.total_requests - 1))
                / model.total_requests
            )
    
    def update_student_performance(self, success_rate: float) -> None:
        """
        Update student model performance and adjust routing.
        
        Called after verification to adjust progressive handoff.
        """
        if success_rate > 0.8:
            # Student doing well - increase usage
            self.current_student_rate = min(
                self.current_student_rate + self.config.improvement_increment,
                self.config.max_student_rate
            )
        elif success_rate < 0.5:
            # Student struggling - decrease usage
            self.current_student_rate = max(
                self.current_student_rate - self.config.improvement_increment,
                self.config.student_sample_rate
            )
    
    def _log_decision(self, decision: RoutingDecision) -> None:
        """Log a routing decision."""
        self.routing_history.append(decision)
        
        if self.config.log_routing_decisions:
            # Append to daily log file
            date_str = datetime.now().strftime("%Y%m%d")
            log_path = Path(self.config.metrics_dir) / f"routing_{date_str}.jsonl"
            
            with open(log_path, 'a') as f:
                f.write(json.dumps(decision.to_dict()) + "\n")
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        if not self.routing_history:
            return {"error": "No routing history"}
        
        teacher_count = sum(
            1 for d in self.routing_history
            if d.model_type == ModelType.TEACHER
        )
        student_count = len(self.routing_history) - teacher_count
        
        return {
            "total_requests": len(self.routing_history),
            "teacher_requests": teacher_count,
            "student_requests": student_count,
            "student_rate": student_count / len(self.routing_history),
            "current_student_rate": self.current_student_rate,
            "avg_complexity": sum(d.task_complexity for d in self.routing_history) / len(self.routing_history),
            "models": {
                name: {
                    "type": model.model_type.value,
                    "requests": model.total_requests,
                    "success_rate": model.success_rate,
                    "avg_latency_ms": model.avg_latency_ms
                }
                for name, model in self.models.items()
            }
        }
    
    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()


async def main():
    """Example usage of the Model Router."""
    config = RouterConfig(
        strategy=RoutingStrategy.PROGRESSIVE,
        student_sample_rate=0.2
    )
    
    router = ModelRouter(config)
    
    # Example routing decisions
    prompts = [
        "Fix the typo in README.md",
        "Refactor the authentication system to use OAuth2 with PKCE flow",
        "Add a print statement to debug.py",
        "Implement a distributed cache with Redis cluster support"
    ]
    
    for prompt in prompts:
        decision = router.route(prompt)
        print(f"Prompt: {prompt[:50]}...")
        print(f"  -> Model: {decision.selected_model} ({decision.model_type.value})")
        print(f"  -> Complexity: {decision.task_complexity:.2f}")
        print()
    
    # Print stats
    stats = router.get_routing_stats()
    print("Routing Stats:")
    print(json.dumps(stats, indent=2))
    
    await router.close()


if __name__ == "__main__":
    asyncio.run(main())
