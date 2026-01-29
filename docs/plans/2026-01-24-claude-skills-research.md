# Claude Skills Research Plan: Training & Dataset Tools

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Find and catalog open-source CLAUDE.md files and Claude Code skills for building training datasets, LoRA adapters, agent training, and trace parsing.

**Architecture:** Systematic search across GitHub, Anthropic resources, HuggingFace, and community platforms. Collect findings into a curated catalog with quality assessments.

**Tech Stack:** Web search, GitHub API, community platforms

---

## Task 1: GitHub Search - CLAUDE.md Files

**Goal:** Find repos containing CLAUDE.md files related to ML training workflows

**Step 1: Search for CLAUDE.md files with training keywords**

Search queries to run:
```
"CLAUDE.md" training dataset site:github.com
"CLAUDE.md" LoRA fine-tuning site:github.com
"CLAUDE.md" agent training site:github.com
"CLAUDE.md" trace parsing site:github.com
filename:CLAUDE.md training
filename:CLAUDE.md dataset
filename:CLAUDE.md fine-tune
```

**Step 2: Search for Claude Code skill repositories**

```
"claude code" skills training site:github.com
claude-code plugin training site:github.com
anthropic claude skills agent site:github.com
```

**Step 3: Document findings**

For each relevant repo found, record:
- Repo URL
- What the CLAUDE.md covers
- Relevance score (1-5) for our use case
- Key techniques/patterns worth adopting

---

## Task 2: GitHub Search - Training Pipeline Repos

**Goal:** Find repos that may have CLAUDE.md for ML/training workflows even if not explicitly labeled

**Step 1: Search LoRA/fine-tuning repos**

```
LoRA training pipeline CLAUDE site:github.com
unsloth training CLAUDE.md site:github.com
axolotl fine-tuning claude site:github.com
peft training claude instructions site:github.com
```

**Step 2: Search agent training repos**

```
agent training synthetic data CLAUDE site:github.com
LLM agent fine-tuning claude site:github.com
self-improving agent CLAUDE.md site:github.com
```

**Step 3: Search trace/data processing repos**

```
LLM trace parsing CLAUDE site:github.com
conversation trace dataset site:github.com
claude code session parser site:github.com
training data synthesis claude site:github.com
```

---

## Task 3: Anthropic Official Resources

**Goal:** Find official Anthropic examples and recommended patterns

**Step 1: Check Anthropic GitHub org**

```
site:github.com/anthropics CLAUDE.md
site:github.com/anthropics training
site:github.com/anthropics fine-tuning
```

**Step 2: Check Claude Code documentation**

- https://docs.anthropic.com - Claude Code section
- https://github.com/anthropics/claude-code - official repo
- Any official skill/plugin examples

**Step 3: Check Anthropic cookbook**

```
site:github.com/anthropics/anthropic-cookbook training
site:github.com/anthropics/anthropic-cookbook fine-tuning
site:github.com/anthropics/anthropic-cookbook dataset
```

---

## Task 4: HuggingFace Search

**Goal:** Find HuggingFace repos/spaces with Claude integration for training

**Step 1: Search HuggingFace repos**

```
CLAUDE.md site:huggingface.co
claude code training site:huggingface.co
claude agent fine-tuning site:huggingface.co
```

**Step 2: Search HuggingFace datasets**

```
claude conversation dataset site:huggingface.co/datasets
claude code traces site:huggingface.co/datasets
agent training data claude site:huggingface.co/datasets
```

**Step 3: Search HuggingFace Spaces**

```
claude training pipeline site:huggingface.co/spaces
LoRA training claude site:huggingface.co/spaces
```

---

## Task 5: Community Resources

**Goal:** Find community-shared skills and patterns

**Step 1: Reddit search**

```
site:reddit.com/r/ClaudeAI CLAUDE.md training
site:reddit.com/r/LocalLLaMA claude code fine-tuning
site:reddit.com/r/MachineLearning claude agent training
```

**Step 2: Discord communities**

Check these communities for shared resources:
- Anthropic Discord (if public channels exist)
- Unsloth Discord
- LocalLLaMA Discord
- AI/ML training communities

**Step 3: Twitter/X search**

```
CLAUDE.md training site:twitter.com
claude code skills training site:x.com
```

---

## Task 6: Specialized Tool Ecosystems

**Goal:** Check ecosystems around popular training tools

**Step 1: Unsloth ecosystem**

```
unsloth CLAUDE.md site:github.com
unslothai training instructions claude
```

**Step 2: Axolotl ecosystem**

```
axolotl CLAUDE.md site:github.com
OpenAccess-AI-Collective claude
```

**Step 3: LlamaIndex/LangChain agent training**

```
llamaindex agent training CLAUDE.md site:github.com
langchain fine-tuning CLAUDE.md site:github.com
```

**Step 4: DSPy / prompt optimization**

```
dspy training CLAUDE.md site:github.com
prompt optimization claude instructions site:github.com
```

---

## Task 7: Compile Research Catalog

**Goal:** Organize all findings into actionable catalog

**Step 1: Create catalog structure**

Create `docs/research/claude-skills-catalog.md` with sections:
- Training Dataset Skills (scored by relevance)
- LoRA/Fine-tuning Skills
- Trace Parsing Skills
- Agent Training Skills
- Data Synthesis Skills

**Step 2: Rate and prioritize**

For each finding:
- Quality score (1-5): How well-written/comprehensive
- Relevance score (1-5): How applicable to bashgym
- Adoption potential: Can we use directly vs. learn from

**Step 3: Extract patterns**

Document common patterns across high-quality CLAUDE.md files:
- How they structure instructions
- What metadata they include
- How they handle edge cases
- Testing/verification approaches

---

## Task 8: Gap Analysis

**Goal:** Identify what's missing that bashgym needs

**Step 1: Compare findings to bashgym needs**

Map catalog to bashgym components:
- Trace capture (post_tool_use.py, session_end.py)
- Data factory (data_factory.py, trace_processor.py)
- Training (trainer.py, gym_env.py)
- Verification (verifier.py)

**Step 2: Identify gaps**

What capabilities exist in found skills that bashgym lacks?
What does bashgym have that others don't?

**Step 3: Prioritize build vs. adopt**

For each gap:
- Can we adopt existing skill directly?
- Should we adapt/customize?
- Need to build from scratch?

---

## Deliverables

After completing all tasks:

1. `docs/research/claude-skills-catalog.md` - Curated catalog of all findings
2. `docs/research/patterns-analysis.md` - Common patterns extracted
3. `docs/research/gap-analysis.md` - What to build vs. adopt
4. Collection of bookmarked/downloaded CLAUDE.md files worth studying

---

## Execution Notes

- Use WebSearch tool for broad searches
- Use WebFetch to retrieve promising CLAUDE.md files
- Some Discord/community content may require manual checking
- Prioritize GitHub and HuggingFace as primary sources
- Document "dead ends" too - knowing what doesn't exist is valuable
