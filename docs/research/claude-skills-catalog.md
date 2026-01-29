# Claude Skills Catalog: Training & Dataset Tools

> Research compiled 2026-01-24 for Bash Gym project

## Executive Summary

This catalog compiles open-source CLAUDE.md files, Claude Code skills, and related resources for building training datasets, LoRA adapters, agent training, and trace parsing. The ecosystem is mature with strong official support from Anthropic and active community adoption.

---

## Top-Tier Resources (Must-Use)

### 1. Training Data Extraction

| Resource | URL | Why It Matters |
|----------|-----|----------------|
| **hanzoai/claude-collector** | [GitHub](https://github.com/hanzoai/claude-collector) | One-command extraction (`uvx claude-collector`) with PII sanitization. Outputs JSONL ready for training. |
| **badlogic/lemmy (claude-trace)** | [GitHub](https://github.com/badlogic/lemmy/tree/main/apps/claude-trace) | Captures full API traces including system prompts, tool outputs, thinking blocks. JSONL format. |
| **jimmc414/cctrace** | [GitHub](https://github.com/jimmc414/cctrace) | Multi-format export (JSONL, Markdown, XML). Captures todos, file snapshots, portable sessions. |
| **langchain-ai/tracing-claude-code** | [GitHub](https://github.com/langchain-ai/tracing-claude-code) | Enterprise-grade tracing with sub-agent support to LangSmith. |

### 2. Training Pipelines

| Resource | URL | Why It Matters |
|----------|-----|----------------|
| **huggingface/skills** | [GitHub](https://github.com/huggingface/skills) | Claude can train models directly via HF Skills. Supports SFT, DPO, GRPO. GPU selection guidance. |
| **zechenzhangAGI/AI-research-SKILLs** | [GitHub](https://github.com/zechenzhangAGI/AI-research-SKILLs) | 77 skills covering Axolotl, Unsloth, PEFT, TRL, GRPO, DeepSpeed. Full research workflow. |
| **Danau5tin/terminal-bench-rl** | [GitHub](https://github.com/Danau5tin/terminal-bench-rl) | GRPO training with synthetic data pipeline. 331 tasks with difficulty levels. Top Qwen3 agent. |

### 3. Datasets

| Resource | URL | Size | Why It Matters |
|----------|-----|------|----------------|
| **nlile/misc-merged-claude-code-traces-v1** | [HuggingFace](https://huggingface.co/datasets/nlile/misc-merged-claude-code-traces-v1) | 32K traces | Unified Claude Code traces for software engineering. Deduplicated, includes git diffs. |
| **agentlans/claude** | [HuggingFace](https://huggingface.co/datasets/agentlans/claude) | 500K+ | Combined Claude dataset from 7 sources including Opus and Sonnet. |
| **arcee-ai/agent-data** | [HuggingFace](https://huggingface.co/datasets/arcee-ai/agent-data) | 486K | Agent training data for function calling and multi-turn conversations. |
| **jupyter-agent/jupyter-agent-dataset** | [HuggingFace](https://huggingface.co/datasets/jupyter-agent/jupyter-agent-dataset) | 95K | Synthetic notebooks with thinking traces. Verified answers. |
| **Lichang-Chen/claude2-alpaca** | [GitHub](https://github.com/Lichang-Chen/claude2-alpaca) | 52K | Claude-2 distillation dataset. Fine-tuned LLaMA-2 models included. |

---

## CLAUDE.md Best Practices

### Official Examples

| Source | URL | Key Patterns |
|--------|-----|--------------|
| **claude-code-action** | [GitHub](https://github.com/anthropics/claude-code-action/blob/main/CLAUDE.md) | Architecture overview, phase documentation, component breakdown |
| **LangChain** | [GitHub](https://github.com/langchain-ai/langchain/blob/master/CLAUDE.md) | API stability ("Would this change break code?"), type hints, docstrings |
| **LangGraph** | [GitHub](https://github.com/langchain-ai/langgraph/blob/main/CLAUDE.md) | Monorepo commands, dependency awareness |

### Community Guidelines

- **Keep under 60 lines** - Less is more; Claude can follow ~150 instructions reliably
- **Use pointers, not copies** - Reference `file:line` instead of embedding code
- **Let linters handle style** - Don't use Claude for deterministic formatting
- **Press # key** - Auto-incorporates instructions during coding

---

## Skills Libraries

### Official Anthropic

| Skill Category | Description |
|----------------|-------------|
| **document-skills** | docx, pdf, pptx, xlsx manipulation |
| **creative** | Art, music, design generation |
| **development** | Web testing, MCP server generation |
| **enterprise** | Communications, branding |

Installation: `/plugin marketplace add anthropics/skills`

### Community Collections

| Repository | Focus | Stars/Quality |
|------------|-------|---------------|
| **travisvn/awesome-claude-skills** | General curation | Best organized |
| **karanb192/awesome-claude-skills** | 50+ verified, TDD, debugging | Most comprehensive |
| **hesreallyhim/awesome-claude-code** | Hooks, slash-commands, orchestrators | Most technical |

---

## Prompt Optimization

| Resource | URL | Techniques |
|----------|-----|------------|
| **2025-Claude-OpusPrompt** | [GitHub](https://github.com/samihalawa/2025-Claude-OpusPrompt) | 35 techniques: extended thinking, effort parameter, tool search, agentic patterns |
| **DSPy-skills** | [GitHub](https://github.com/OmidZamani/dspy-skills) | 5 optimizers: Bootstrap, MIProv2, GEPA, SIMBA, Fine-tune bootstrap |
| **claude-code-prompt-optimizer** | [GitHub](https://github.com/johnpsasser/claude-code-prompt-optimizer) | Hook using Opus 4.1 extended thinking |

---

## Self-Improvement Patterns

| Resource | URL | Pattern |
|----------|-----|---------|
| **bokan/claude-skill-self-improvement** | [GitHub](https://github.com/bokan/claude-skill-self-improvement) | Parallel agents analyze sessions, generate CLAUDE.md improvements |
| **aviadr1/claude-meta** | [GitHub](https://github.com/aviadr1/claude-meta) | Reflect -> Abstract -> Generalize. "Magic prompt" for learning from mistakes. |

---

## System Prompts & Internals

| Resource | URL | Contents |
|----------|-----|----------|
| **claude-code-system-prompts** | [GitHub](https://github.com/Piebald-AI/claude-code-system-prompts) | 110+ prompts from Claude Code v2.1.19. Updated within minutes of releases. |

---

## RL Environments

| Resource | URL | Pattern |
|----------|-----|---------|
| **KhoomeiK/LlamaGym** | [GitHub](https://github.com/KhoomeiK/LlamaGym) | Fine-tune LLM agents with PPO in Gym environments |
| **PrimeIntellect-ai/verifiers** | [GitHub](https://github.com/PrimeIntellect-ai/verifiers) | Dataset + Harness + Rubric pattern for RL |

---

## Data Format Standards

### ChatML/Messages (Most Common)
```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

### Bedrock Fine-Tuning Format
```json
{
  "system": "<optional_system_message>",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```
- First message: user
- Last message: assistant
- Alternating turns required

---

## Quick Installation Commands

```bash
# Official Anthropic skills
/plugin marketplace add anthropics/skills

# HuggingFace training skills
claude mcp add --transport http hf-skills https://huggingface.co/mcp?bouquet=skills --header "Authorization: Bearer $HF_TOKEN"

# AI Research skills (77 skills)
/plugin install fine-tuning@ai-research-skills

# Claude Collector for trace extraction
uvx claude-collector
```

---

## Sources by Platform

### GitHub
- anthropics/skills, anthropics/evals, anthropics/claude-code-action
- hanzoai/claude-collector, badlogic/lemmy, jimmc414/cctrace
- zechenzhangAGI/AI-research-SKILLs, huggingface/skills
- langchain-ai/langchain, langchain-ai/langgraph
- Lichang-Chen/claude2-alpaca, Danau5tin/terminal-bench-rl

### HuggingFace
- Datasets: nlile/misc-merged-claude-code-traces-v1, agentlans/claude, arcee-ai/agent-data
- Blog: hf-skills-training, sionic-ai skills pattern

### Community
- Skills Marketplace: skillsmp.com (71,000+ skills)
- Awesome lists: travisvn, VoltAgent, karanb192, hesreallyhim
