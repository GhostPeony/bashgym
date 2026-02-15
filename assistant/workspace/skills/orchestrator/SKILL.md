---
name: orchestrator
description: "Submit development specs, approve execution plans, monitor workers, retry failed tasks, check job status, and manage orchestration jobs. Use when asked to start a development job, check worker progress, approve a plan, retry something, or list orchestrator jobs."
---

# Orchestrator

Submit development specs for multi-agent decomposition and parallel execution.

## Submit a Spec

Decompose a development task into a parallel execution plan:
```
scripts/api.sh POST /orchestrate/submit '{
  "title": "Add user authentication",
  "description": "Implement JWT-based auth with login/register endpoints",
  "constraints": ["Use existing User model", "Add pytest tests"],
  "acceptance_criteria": ["All tests pass", "Endpoints return proper status codes"],
  "max_workers": 3,
  "budget_usd": 5.0,
  "provider": "anthropic"
}'
```

Returns: job_id and decomposed task plan. **Always summarize the plan for the user before approving.**

## Approve a Plan

After reviewing the plan, approve to begin execution:
```
scripts/api.sh POST /orchestrate/{job_id}/approve '{}'
```

## Check Job Status

Monitor worker progress and task states:
```
scripts/api.sh GET /orchestrate/{job_id}/status
```

Returns: task states (pending/running/completed/failed), worker assignments, budget usage.

## Retry a Failed Task

Retry a specific failed task within a job:
```
scripts/api.sh POST /orchestrate/{job_id}/task/{task_id}/retry '{}'
```

## List All Jobs

```
scripts/api.sh GET /orchestrate/jobs
```

## Delete a Job

**Destructive — confirm with user first.**
```
scripts/api.sh DELETE /orchestrate/{job_id}
```

## List Available Providers

Check which LLM providers are configured for planning:
```
scripts/api.sh GET /orchestrate/providers
```

## Example Interactions

User: "Build me an auth system with JWT"
→ Ask for any constraints. Compose a spec. Call POST /orchestrate/submit.
   Summarize the decomposed plan. Ask user to approve before calling /approve.

User: "What's the status of my last job?"
→ Call GET /orchestrate/jobs to find the latest. Call GET /orchestrate/{job_id}/status.
   Report task states and budget usage.

User: "Task 3 failed, retry it"
→ Call POST /orchestrate/{job_id}/task/{task_id}/retry. Report result.
