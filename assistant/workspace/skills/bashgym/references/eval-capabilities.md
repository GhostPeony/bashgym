# BashGym Evaluation And Promotion Evidence

Every reusable run needs exact config/lineage, metrics, method-specific evaluation, baseline comparison, and a RunCard or equivalent immutable evidence record. Training completion and lower loss alone are insufficient.

| Method                      | Minimum evaluation evidence                                                                                                  |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| SFT                         | Heldout behavior; environment pass@k and holdout gate for terminal-facing models; format/tool validity                       |
| DPO                         | Strict same-prompt pair validation, preference metrics, heldout preference behavior, and no regression from SFT/base         |
| GRPO/RLVR                   | Rollout pass@k, reward variance and zero-std diagnostics, environment holdout, reward-hacking canaries, and release evidence |
| Teacher distillation        | Student loss/divergence plus heldout comparison against both base student and teacher                                        |
| Session Distillation        | Record validation, masked-token loss/KL/CE, heldout recovery behavior, and terminal pass@k when applicable                   |
| DPPO                        | Replay/logprob contract, smoke bundle/backend evidence, pass@k before/after, divergence thresholds, and safety comparison    |
| ECHO/RWML                   | Diagnostic quality correlated with heldout pass@k/failure reduction; never standalone promotion evidence                     |
| Reward model                | Strict reward examples, heldout pair accuracy/calibration, bias/leakage slices, and downstream selection/control comparison  |
| Cascade/MOPD                | Per-stage RunCards and gates plus final broad heldout and routing/safety comparison                                          |
| Embedding retrieval profile | Frozen query/corpus split, nDCG/MRR/Recall, hard-slice and latency/footprint comparisons, and contamination controls         |

## Evidence rules

- Preserve protected test sets until the declared final gate.
- Compare against a pinned baseline on the same data, environment, and scoring contract.
- Label smoke/runtime evidence separately from model-quality evidence.
- Keep raw metrics and reports in BashGym; write only curated summaries and artifact references into GBrain.
- A missing metrics file, empty evaluation history, or absent baseline comparison blocks promotion.
- Public release also requires model/data license, provenance, privacy, safety, and model-card review.
