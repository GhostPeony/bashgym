# Common Patterns in High-Quality CLAUDE.md Files

> Extracted from research across 50+ repositories

---

## 1. Document Structure

### Standard Sections
```markdown
# Project Name

## Quick Reference (or Commands)
- Build: `command`
- Test: `command`
- Lint: `command`

## Architecture Overview
[Brief description of system design]

## Key Files
- `path/to/file.py` - Purpose

## Development Guidelines
[Rules and conventions]

## Common Tasks
[How-to guides for frequent operations]
```

### YAML Frontmatter (for Skills)
```yaml
---
name: skill-name
description: What it does and when to activate
---
```

---

## 2. Instruction Patterns

### DO: Concise, Actionable Rules
```markdown
- Always run `pytest` before committing
- Use type hints on all public functions
- First message in conversations must be from user role
```

### DON'T: Vague or Redundant
```markdown
- Try to write good code (too vague)
- Make sure to test things (too vague)
- Follow best practices (not actionable)
```

---

## 3. Reference Patterns

### Pointer Style (Preferred)
```markdown
See authentication logic in `src/auth/handler.py:45-67`
```

### Embedded Style (Avoid for Large Blocks)
```markdown
The function should look like:
\`\`\`python
def validate(input):
    # 50 lines of code...
\`\`\`
```

---

## 4. Command Documentation

### Complete Examples
```markdown
## Testing

Run all tests:
\`\`\`bash
pytest tests/ -v
\`\`\`

Run single test:
\`\`\`bash
TEST=tests/test_auth.py make test
\`\`\`
```

### With Expected Output
```markdown
Run: `npm run build`
Expected: No errors, bundle in `dist/`
```

---

## 5. Architecture Descriptions

### Component Diagram (ASCII)
```
┌─────────┐    ┌─────────┐    ┌─────────┐
│  Input  │───▶│ Process │───▶│ Output  │
└─────────┘    └─────────┘    └─────────┘
```

### Layer Table
| Layer | Files | Purpose |
|-------|-------|---------|
| API | `routes.py` | HTTP endpoints |
| Service | `service.py` | Business logic |
| Data | `models.py` | Database models |

---

## 6. Constraint Patterns

### API Stability (LangChain Pattern)
```markdown
Before any change, ask: "Would this break someone's existing code?"
```

### Safety Constraints
```markdown
NEVER:
- Execute commands with `sudo`
- Modify files outside project directory
- Commit credentials or API keys
```

### Quality Gates
```markdown
All PRs must:
- Pass CI checks
- Include tests for new functionality
- Update documentation if applicable
```

---

## 7. Tool-Specific Patterns

### For Fine-Tuning Skills
```markdown
## Data Format
- JSONL with messages array
- First message: user, last message: assistant
- Alternating roles required

## Hyperparameters
- Learning rate: 2e-5
- Batch size: 4 (adjust for VRAM)
- LoRA rank: 16-64
```

### For Trace Capture
```markdown
## Output Location
Traces written to: `~/.claude-trace/log-YYYY-MM-DD.jsonl`

## Fields Captured
- request: Full API request
- response: Complete response with thinking blocks
- tokens: Usage counts
```

---

## 8. Size Guidelines

### Optimal Length
- **CLAUDE.md**: 50-150 lines (under 300 max)
- **SKILL.md**: 100-500 tokens metadata, <5k tokens full

### Instruction Limits
- Claude reliably follows: 150-200 instructions
- System prompt uses: ~50 slots
- Available for CLAUDE.md: ~100-150 instructions

---

## 9. Anti-Patterns to Avoid

| Anti-Pattern | Why It's Bad | Better Approach |
|--------------|--------------|-----------------|
| Embedding entire files | Bloats context | Use file:line pointers |
| Style rules for linting | Redundant with tools | Let linters handle |
| Vague "be careful" warnings | Not actionable | Specific constraints |
| Duplicating docs | Maintenance burden | Link to source |
| Over-detailed history | Wastes context | Keep current state only |

---

## 10. Skill-Specific Patterns

### Progressive Disclosure
1. Metadata loads first (~100 tokens)
2. Full instructions load on activation (<5k tokens)
3. Resources load as needed

### Activation Triggers
```yaml
description: Use when creating PDF documents or extracting text from PDFs
```

### Example-Driven
```markdown
## Examples
- "Create a PDF report" → activates pdf skill
- "Extract tables from invoice.pdf" → activates pdf skill

## Non-Examples
- "What is a PDF?" → does NOT activate (question, not task)
```

---

## 11. Self-Improvement Patterns

### Reflection Mechanism (claude-meta)
```markdown
After any mistake:
1. Reflect on what went wrong
2. Abstract the lesson
3. Generalize to a rule
4. Write to CLAUDE.md
```

### Magic Prompt
```
"Reflect on this mistake. Abstract and generalize the learning. Write it to CLAUDE.md."
```

---

## 12. Training Data Patterns

### Sanitization (claude-collector)
- API keys: All major providers (OpenAI, Anthropic, AWS, etc.)
- PII: Emails, phones, SSNs, credit cards
- Paths: Replace usernames with placeholders
- Secrets: Crypto seed phrases

### Segmentation Heuristics
- Time gaps > 5 minutes
- Git commits (task completion)
- Directory changes
- File scope changes

---

## Summary: The Ideal CLAUDE.md

1. **Starts with commands** - Quick reference for common tasks
2. **Architecture in 5 sentences** - What, why, how
3. **Key files with line numbers** - Navigation aids
4. **Constraints as NEVER/ALWAYS** - Clear boundaries
5. **Examples over explanations** - Show, don't tell
6. **Under 100 lines** - Respect context limits
