# Guardrails & Profiler Integration Design

> Integrating NemoGuard and AgentProfiler across all execution paths

**Date:** 2026-01-25
**Status:** Draft

---

## Overview

This design integrates the existing `NemoGuard` (guardrails) and `AgentProfiler` (profiler) components into the actual execution paths of Bash Gym. Currently these modules exist but are not wired into the system.

**Goals:**
- Safety: Block dangerous commands, detect injection attempts, filter PII from training data
- Observability: Profile execution across all components, expose metrics to dashboard
- Training signals: Capture guardrail violations as negative examples for DPO

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         API Layer                                    │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐     │
│   │  /api/tasks │    │ /api/training│    │ /api/observability │     │
│   └──────┬──────┘    └──────┬──────┘    └──────────┬──────────┘     │
│          │                  │                      │                 │
│          ▼                  ▼                      ▼                 │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │              Profiler Middleware (all requests)              │   │
│   └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
   │  AgentRunner │     │  GymEnv     │     │ ModelRouter │
   │  (Arena)     │     │  (Gym)      │     │  (Gym)      │
   └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
          │                   │                   │
          ▼                   ▼                   ▼
   ┌─────────────────────────────────────────────────────────────┐
   │         Instrumented Execution Layer                         │
   │  ┌───────────────┐              ┌───────────────┐           │
   │  │   Profiler    │◄────────────►│   Guardrails  │           │
   │  │  (spans,      │   records    │  (check before│           │
   │  │   metrics)    │   events     │   execute)    │           │
   │  └───────────────┘              └───────────────┘           │
   └─────────────────────────────────────────────────────────────┘
```

**Key principle:** Guardrails check *before* execution, Profiler records *around* execution. When a guardrail blocks something, the profiler records it as a failed span with the block reason.

---

## Integration Points

### Summary

| # | Component | Guardrails | Profiler |
|---|-----------|------------|----------|
| 1 | Trace Import | PII filter, injection detection | Import stats, redaction count |
| 2 | Gym Environment | Code safety on commands | Episode traces, step spans, rewards |
| 3 | Agent Runner | Input/output checks | Task execution traces |
| 4 | Model Router | Student output validation | Student vs teacher comparison |
| 5 | API Layer | Request validation | Expose metrics to dashboard |

### 1. Trace Import (`bashgym/trace_capture/importers/claude_history.py`)

| Method | Guardrails | Profiler |
|--------|------------|----------|
| `parse_session_file()` | `check_input()` + PII filter on `user_initial_prompt` | Start span |
| Tool result extraction | `check_output()` + PII filter on `tool_output` | Record redaction count |
| `import_session()` | — | End span, record stats |

### 2. Gym Environment (`bashgym/gym/environment.py`)

| Method | Guardrails | Profiler |
|--------|------------|----------|
| `reset()` | — | Start trace for episode |
| `step()` | — | Wrap as span, record reward |
| `_execute_bash()` | `check_command()` before exec | Record tool call span |
| `_execute_write()` | `check_output()` on content | Record tool call span |
| `_execute_edit()` | `check_output()` on edit spec | Record tool call span |
| `_compute_verification_reward()` | — | Record verification span |

### 3. Agent Runner (`bashgym/arena/runner.py`)

| Method | Guardrails | Profiler |
|--------|------------|----------|
| `run_task()` | `check_input()` on task prompt | Start trace for task |
| `build_claude_command()` | — | — |
| Output streaming | `check_output()` + PII filter | Record LLM call spans |
| Task completion | — | End trace, export |

### 4. Model Router (`bashgym/gym/router.py`)

| Method | Guardrails | Profiler |
|--------|------------|----------|
| `route_request()` | `check_input()` on prompt | Start span for routing decision |
| `_call_teacher()` | `check_output()` on response | Record LLM span (teacher) |
| `_call_student()` | `check_output()` on response | Record LLM span (student) |
| `_compare_responses()` | — | Record comparison metrics |

### 5. API Layer (`bashgym/api/routes.py`)

| Location | Guardrails | Profiler |
|----------|------------|----------|
| Middleware | `check_input()` on request body | Start request span |
| Task endpoints | Inherit from runner | Inherit from runner |
| New endpoints | — | `/api/observability/traces`, `/api/observability/guardrails` |

---

## Guardrails Details

### Check Types by Location

| Check Type | Where Applied |
|------------|---------------|
| Injection Detection | Trace Import (user prompts), Agent Runner (task prompts), API Layer (request bodies) |
| Code Safety | Gym Environment (`_execute_bash`), Agent Runner (captured commands), Model Router (student outputs with code) |
| PII Filtering | Trace Import (prompts + outputs), Agent Runner (before trace capture), Training data export |
| Content Moderation | Model Router (student responses), API Layer (if exposing to users) |

### Actions When Triggered

| Trigger | Action | Training Signal |
|---------|--------|-----------------|
| Injection detected | Block, log, alert | Negative example (if from student) |
| Dangerous command | Block execution, return error | Negative example |
| PII found | Redact, continue | N/A (just sanitize) |
| Content violation | Block response | Negative example |

### Guardrail Event Schema

```python
@dataclass
class GuardrailEvent:
    timestamp: datetime
    check_type: GuardrailType      # INJECTION, CODE_SAFETY, PII, CONTENT
    location: str                   # "gym.execute_bash", "import.user_prompt"
    action_taken: GuardrailAction   # ALLOW, BLOCK, WARN, MODIFY
    original_content: str           # What was checked (truncated)
    modified_content: Optional[str] # If PII was redacted
    confidence: float               # How confident the detection was
    model_source: Optional[str]     # "teacher" or "student" if applicable
    trace_id: Optional[str]         # Link to profiler trace
```

---

## Profiler Details

### Trace Hierarchy

```
Trace (top-level, e.g., "task_12345" or "training_batch_7")
│
├── Span: gym.episode
│   ├── Span: gym.step[0]
│   │   ├── Span: guardrails.check_command
│   │   ├── Span: gym.execute_bash
│   │   └── Span: gym.compute_reward
│   ├── Span: gym.step[1]
│   │   └── ...
│   └── Span: gym.verification
│
├── Span: router.request
│   ├── Span: guardrails.check_input
│   ├── Span: router.call_student
│   │   └── Span: llm.inference (tokens, latency)
│   ├── Span: guardrails.check_output
│   └── Span: router.compare (if teacher fallback)
│
└── Span: import.session
    ├── Span: guardrails.pii_filter (count redacted)
    └── Span: import.save_trace
```

### Metrics Collected

| Location | Metrics |
|----------|---------|
| Gym Environment | steps_per_episode, reward_per_step, success_rate, tokens_per_step |
| Agent Runner | task_duration_ms, commands_executed, commands_blocked |
| Model Router | student_latency_ms, teacher_latency_ms, student_token_count, fallback_rate, confidence_scores |
| Trace Import | sessions_imported, steps_per_session, pii_redactions_count, import_duration_ms |
| Guardrails | checks_per_type, block_rate, avg_confidence |

### Profiler-Guardrails Integration

```python
# Example: How they work together
with profiler.span("gym.execute_bash", kind=SpanKind.TOOL_CALL) as span:
    # Guardrail check (creates child span automatically)
    check_result = await guardrails.check_command(command)
    span.set_attribute("guardrail.action", check_result.action.value)

    if check_result.action == GuardrailAction.BLOCK:
        span.finish("blocked")
        return blocked_observation

    # Execute if allowed
    result = sandbox.execute(command)
    span.set_attribute("exit_code", result.exit_code)
```

---

## API Endpoints

### New Observability Endpoints

```
GET  /api/observability/traces
     → List recent traces with summary stats

GET  /api/observability/traces/{trace_id}
     → Full trace detail with all spans

GET  /api/observability/metrics
     → Aggregated metrics (avg latency, token usage, block rates)

GET  /api/observability/guardrails/events
     → Recent guardrail events (blocks, warnings, PII redactions)

GET  /api/observability/guardrails/stats
     → Guardrail statistics (checks by type, block rate over time)
```

### WebSocket Events

```
WS   /ws
     → Add new message types:
        - "guardrail:blocked" (real-time alert)
        - "profiler:span_complete" (live trace updates)
        - "import:pii_redacted" (PII filtering notifications)
```

### Settings Endpoints

```
POST /api/settings/guardrails
     {"pii_filtering": false}  # Toggle features

POST /api/settings/profiler
     {"enabled": false}  # Toggle profiler
```

---

## Frontend Dashboard

### Components

| Component | Data Source | Shows |
|-----------|-------------|-------|
| Trace Timeline | `/api/observability/traces` | Visual trace/span hierarchy |
| Metrics Cards | `/api/observability/metrics` | Avg latency, tokens/step, success rate |
| Guardrail Feed | WebSocket `guardrail:*` | Real-time blocked commands, PII redactions |
| Student vs Teacher | `/api/observability/metrics` | Side-by-side comparison chart |
| Import Stats | `/api/observability/traces` | Sessions imported, PII redacted count |

### Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  Bash Gym - Observability                                       │
├─────────────────┬───────────────────────────────────────────────┤
│                 │                                                │
│  METRICS        │  TRACE TIMELINE                                │
│  ┌───────────┐  │  ┌─────────────────────────────────────────┐  │
│  │ Latency   │  │  │ ▼ task_abc123 (2.3s)                    │  │
│  │ 1.2s avg  │  │  │   ├─ gym.step[0] (0.4s)                 │  │
│  ├───────────┤  │  │   │  ├─ guardrails.check ✓              │  │
│  │ Tokens    │  │  │   │  └─ execute_bash (0.3s)             │  │
│  │ 847/step  │  │  │   ├─ gym.step[1] (0.5s)                 │  │
│  ├───────────┤  │  │   │  └─ guardrails.check ✗ BLOCKED      │  │
│  │ Blocked   │  │  │   └─ ...                                │  │
│  │ 3 today   │  │  └─────────────────────────────────────────┘  │
│  └───────────┘  │                                                │
├─────────────────┴───────────────────────────────────────────────┤
│  GUARDRAIL EVENTS (live)                                         │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ 10:32:15  BLOCK  code_safety  "rm -rf /" from student       ││
│  │ 10:31:02  MODIFY pii_filter   Redacted 2 emails in import   ││
│  │ 10:28:44  WARN   injection    Suspicious pattern detected   ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

---

## Configuration

### Settings Classes

Add to `bashgym/settings.py`:

```python
@dataclass
class GuardrailsSettings:
    enabled: bool = True

    # Check toggles
    injection_detection: bool = True
    code_safety: bool = True
    pii_filtering: bool = True
    content_moderation: bool = False  # Requires NeMo endpoint

    # Thresholds
    injection_threshold: float = 0.8

    # Code safety
    blocked_commands: List[str] = field(default_factory=lambda: [
        "rm -rf /", "rm -rf /*", ":(){:|:&};:",
        "dd if=/dev/zero", "mkfs.", "> /dev/sda",
    ])

    # NeMo Guardrails endpoint (optional)
    nemo_endpoint: Optional[str] = None


@dataclass
class ProfilerSettings:
    enabled: bool = True

    # What to profile
    profile_tokens: bool = True
    profile_latency: bool = True
    profile_guardrails: bool = True

    # Storage
    output_dir: str = "data/profiler_traces"
    max_traces_in_memory: int = 1000
```

### Environment Variables

```bash
# Guardrails
GUARDRAILS_ENABLED=true
GUARDRAILS_PII_FILTERING=true
GUARDRAILS_INJECTION_DETECTION=true
GUARDRAILS_CODE_SAFETY=true

# Profiler
PROFILER_ENABLED=true
PROFILER_OUTPUT_DIR=data/profiler_traces
```

---

## Data Flow

### 1. Trace Import Flow

```
~/.claude/projects/*.jsonl
        │
        ▼
┌───────────────────┐
│ ClaudeSessionImporter │
└───────────┬───────────┘
            │
            ▼
┌───────────────────┐     ┌─────────────────┐
│ NemoGuard         │────▶│ Profiler        │
│  • PII filter     │     │  • import stats │
│  • injection check│     │  • redact count │
└───────────┬───────┘     └─────────────────┘
            │
            ▼
data/traces/*.json (sanitized)
```

### 2. Gym Execution Flow

```
Task Prompt
     │
     ▼
┌─────────────┐
│ BashGymEnv │──────▶ Profiler: start_trace("episode")
│   reset()   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   step()    │──────▶ Profiler: start_span("step[n]")
└──────┬──────┘
       │
       ▼
┌─────────────────┐     ┌─────────────────┐
│ NemoGuard       │────▶│ Profiler        │
│  check_command()│     │  span: guardrail│
└───────┬─────────┘     └─────────────────┘
        │
  ┌─────┴─────┐
  │           │
ALLOW       BLOCK
  │           │
  ▼           ▼
Execute    Return error
           (negative reward)
```

### 3. Model Router Flow

```
Inference Request
       │
       ▼
┌──────────────────┐
│ ModelRouter      │──────▶ Profiler: start_span("router.request")
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Call Student     │──────▶ Profiler: span("llm.student")
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ NemoGuard        │
│  check_output()  │
└────────┬─────────┘
         │
   ┌─────┴─────┐
   │           │
 PASS        BLOCK
   │           │
   ▼           ▼
Return      Fallback to Teacher
```

---

## Files to Create

```
bashgym/
├── core/
│   └── instrumentation.py       # Shared guardrails + profiler wrapper
├── api/
│   └── observability_routes.py  # New endpoints

tests/
└── e2e/
    ├── conftest.py
    ├── test_guardrails_e2e.py
    ├── test_profiler_e2e.py
    └── test_integration_e2e.py
```

## Files to Modify

| File | Changes |
|------|---------|
| `bashgym/trace_capture/importers/claude_history.py` | Add guardrails + profiler integration |
| `bashgym/gym/environment.py` | Wrap execution with guardrails + profiler |
| `bashgym/arena/runner.py` | Add guardrails + profiler to task execution |
| `bashgym/gym/router.py` | Add guardrails + profiler to routing |
| `bashgym/api/routes.py` | Add observability endpoints, WebSocket events |
| `bashgym/settings.py` | Add GuardrailsSettings, ProfilerSettings |
| `frontend/src/` | Add observability dashboard components |

---

## E2E Tests

### Test Structure

```
tests/
└── e2e/
    ├── __init__.py
    ├── conftest.py              # Shared fixtures
    ├── test_guardrails_e2e.py   # Guardrails across all points
    ├── test_profiler_e2e.py     # Profiler across all points
    └── test_integration_e2e.py  # Combined guardrails + profiler
```

### Test Cases

#### Trace Import E2E

| Test | Verifies |
|------|----------|
| `test_import_redacts_pii_from_prompt` | Email replaced with `[EMAIL_REDACTED]` |
| `test_import_redacts_pii_from_output` | Phone replaced with `[PHONE_REDACTED]` |
| `test_import_blocks_injection_attempt` | Trace flagged/quarantined |
| `test_import_records_profiler_stats` | Profiler shows import count, redaction count |
| `test_import_with_guardrails_disabled` | PII passes through when disabled |

#### Gym Environment E2E

| Test | Verifies |
|------|----------|
| `test_gym_blocks_dangerous_command` | Action blocked, negative reward, span recorded |
| `test_gym_allows_safe_command` | Command executes, span shows success |
| `test_gym_profiles_full_episode` | Trace has step spans + guardrail spans |
| `test_gym_records_token_usage` | Spans include token counts |
| `test_gym_blocked_command_creates_negative_example` | Event available for DPO |

#### Model Router E2E

| Test | Verifies |
|------|----------|
| `test_router_blocks_unsafe_student_output` | Response blocked, fallback to teacher |
| `test_router_allows_safe_student_output` | Response passes through |
| `test_router_profiles_student_vs_teacher` | Both spans recorded, latency compared |
| `test_router_records_confidence_scores` | Span attributes include confidence |
| `test_router_pii_filters_student_response` | Email redacted before returning |

#### API Layer E2E

| Test | Verifies |
|------|----------|
| `test_api_returns_traces` | Returns recent traces with spans |
| `test_api_returns_guardrail_events` | Returns recent blocks/warnings |
| `test_api_returns_metrics` | Returns aggregated stats |
| `test_websocket_guardrail_alert` | WebSocket receives `guardrail:blocked` |
| `test_api_toggle_guardrails` | Guardrails disabled/enabled at runtime |

#### Full Pipeline E2E

| Test | Verifies |
|------|----------|
| `test_import_to_training_pii_safe` | No PII in final training JSONL |
| `test_gym_episode_to_dashboard` | Trace visible in API |
| `test_student_blocked_to_dpo_negative` | Event available as DPO negative example |
| `test_guardrail_stats_aggregate` | Stats endpoint shows totals across components |

---

## Implementation Order

1. **Phase 1: Core Infrastructure**
   - Add settings classes to `settings.py`
   - Create `instrumentation.py` wrapper

2. **Phase 2: Trace Import**
   - Integrate guardrails + profiler into `claude_history.py`
   - Add E2E tests for import

3. **Phase 3: Gym Environment**
   - Integrate into `environment.py`
   - Add E2E tests for gym

4. **Phase 4: Model Router**
   - Integrate into `router.py`
   - Add E2E tests for router

5. **Phase 5: Agent Runner**
   - Integrate into `runner.py`
   - Add E2E tests for runner

6. **Phase 6: API Layer**
   - Create `observability_routes.py`
   - Add WebSocket events
   - Add E2E tests for API

7. **Phase 7: Frontend**
   - Add dashboard components
   - Connect to API endpoints

---

## Success Criteria

- [ ] All 5 integration points have guardrails + profiler wired in
- [ ] PII is filtered from all imported traces before storage
- [ ] Dangerous commands are blocked in Gym with negative rewards
- [ ] Student model outputs are validated before returning
- [ ] Dashboard shows real-time guardrail events and profiler metrics
- [ ] All E2E tests pass
- [ ] No PII appears in training data exports
