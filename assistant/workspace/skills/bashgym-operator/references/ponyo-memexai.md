# Ponyo MemexAI operator map

Use this reference only for the current MemexAI embedding-retrieval experiment. The shared BashGym operator remains task-general.

## Start here

Run from the installed skill directory:

```bash
cd /home/ponyo/.hermes/skills/bashgym-operator
python3 scripts/operator_context.py doctor
python3 scripts/operator_context.py context --project memexai
```

The context command reads metadata and metrics, not protected query rows, raw logs, checkpoints, or dataset contents. It reports active jobs, recent run manifests, dataset lineage, development-system comparisons, and exact help/preflight commands.

## Exact live sources

| Ability | Source on Ponyo |
|---|---|
| Active/historical job normalization | `/home/ponyo/.hermes/training_runs.py` |
| Current MemexAI experiment evidence | `/home/ponyo/bashgym-training/memexai-positive-aware-v1-20260712` |
| Dataset lineage | `inputs/positive-aware-manifest.json` |
| Run lineage, hyperparameters, result metrics | `runs/*/training_manifest.json` and `training_metrics.jsonl` |
| Development retrieval comparisons | `system-ablations/protected-dev-v1/*/retrieval_system_ablation_manifest.json` |
| Dataset construction | `scripts/build_positive_aware_dataset.py` |
| Guarded embedding training | `scripts/train_embedding_retriever.py` |
| Corpus embedding and development evaluation | `scripts/embed_corpus_with_model.py`, `evaluate_retrieval_system_ablation.py`, and `evaluate_product_retrieval_fixture.py` |
| Curated operational knowledge | GBrain source `bashgym-activity` |
| Broader MemexAI/BashGym decisions and preferences | GBrain source `default` |

Use the experiment virtual environment for every experiment script:

```bash
PY=/home/ponyo/bashgym-training/memexai-positive-aware-v1-20260712/.venv/bin/python
ROOT=/home/ponyo/bashgym-training/memexai-positive-aware-v1-20260712
$PY $ROOT/scripts/build_positive_aware_dataset.py --help
$PY $ROOT/scripts/train_embedding_retriever.py --help
$PY $ROOT/scripts/evaluate_retrieval_system_ablation.py --help
```

## Recover context and decide the next session

1. Search broader project context and curated activity separately:

   ```bash
   /home/ponyo/gbrain/bin/gbrain search "MemexAI retrieval goals decisions" --source default --limit 10
   /home/ponyo/gbrain/bin/gbrain search "MemexAI embedding candidate evaluation" --source bashgym-activity --limit 10
   ```

2. Read the local context helper output. Compare the base, Candidate A, and Candidate B `dense`, `bm25`, and `rrf` lanes on the physical development split. Never infer model quality from training loss alone.
3. Choose one bounded next-session hypothesis:
   - improve the grouped-positive/hard-negative data, including the pending dense top-50 mining and annotation work;
   - improve retrieval composition using the already-separated dense, BM25, and RRF evidence;
   - train another embedding candidate only when its data or optimization hypothesis is materially different;
   - add a reranker only with a pinned model revision, input template, candidate depth, truncation policy, and latency protocol, and implement product retrieval changes in MemexAI rather than merging repositories.
4. Confirm the goal, development KPIs/gates, immutable input digests, base model, method, compute budget, run ID/output directory, stop policy, and protected-test boundary.
5. Run the trainer with the verified arguments plus `--dry-run`. Inspect its proposed manifest before removing `--dry-run`.
6. Launch only a new output directory. Never overwrite Candidate A, Candidate B, an existing manifest, or the frozen input artifacts.
7. Evaluate against the physical development split, compare to the pinned base and system lanes, then curate the result. Keep `heldout-dev-test.jsonl` unopened until its explicit gate is met.

## Current access boundary

The canvas injects the desktop workspace/campaign projection into Hermes prompts. The current Ponyo Discord environment can inspect and run the local guarded MemexAI scripts, but it cannot automatically mutate the desktop campaign ledger unless `operator_context.py doctor` reports a reachable desktop API or campaign CLI. State this boundary plainly; local run manifests remain valid evidence, but they are not a fabricated campaign transition.
