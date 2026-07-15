# Gemma 4 12B Unified: training and NVFP4 deployment

## Model roles

| Role | Model | Runtime | Use in BashGym |
|---|---|---|---|
| Trainable base | `unsloth/gemma-4-12b-it` | Unsloth + BF16 LoRA | SFT, DPO, and future training runs |
| Deployment candidate | `unsloth/gemma-4-12b-it-NVFP4` | vLLM on Blackwell | Low-latency inference and evaluation |

The NVFP4 repository is a post-training inference quant. It must not be supplied
to `scripts/train_model.py` as the base model. Train an adapter against the
matching 12B base, evaluate it, and then validate that adapter against the NVFP4
runtime before promotion.

The pinned deployment revision evaluated for BashGym is
`4c77b7a6786b83271b3cd3285c03d7191a7ca442`.

## Safe local workflow

On the Blackwell compute host:

```bash
scripts/serve_gemma4_12b_nvfp4.sh prepare
scripts/serve_gemma4_12b_nvfp4.sh preflight
scripts/serve_gemma4_12b_nvfp4.sh start
scripts/serve_gemma4_12b_nvfp4.sh status
```

To expose the existing BashGym 12B LoRA alongside the base model:

```bash
LORA_PATH=/path/to/the/validated/final-adapter \
  LORA_NAME=bashgym-gemma4-12b-sft \
  scripts/serve_gemma4_12b_nvfp4.sh start
```

vLLM then lists both the stable base model ID `gemma4-12b-nvfp4` and the
adapter ID supplied in `LORA_NAME`. Keep the default loopback binding for local
Hermes use; use an authenticated SSH tunnel for a desktop canvas client rather
than exposing the unauthenticated OpenAI-compatible endpoint to the LAN.

The start action intentionally uses a small text-only canary: 4,096-token
context, one sequence, fixed 1 GiB KV cache, eager execution, two compile jobs,
a 24 GiB cgroup ceiling, and an independent host-memory/endpoint watchdog.
CUDA unified-memory allocations are not fully represented by a user cgroup on
DGX Spark, so the watchdog is a required control rather than an optional extra.

## Promotion gates

Do not change the default model until all of these are preserved in a benchmark
artifact:

1. The candidate starts without restarting or degrading protected services.
2. The same fixed prompt suite records latency, generated tokens/second, tool-call
   correctness, task quality, host memory, and service memory.
3. The existing 12B LoRA adapter either loads successfully on the NVFP4 base or a
   replacement adapter is trained and passes the same held-out evaluation.
4. Quality does not regress beyond the campaign threshold and the measured speed
   or footprint improvement is material on the actual target device.
5. The prior checkpoint, service definition, and endpoint remain available for
   immediate rollback.

The initial device canary is intentionally a smoke benchmark rather than a
promotion-quality eval: all 24 paired base/adapter requests completed, exact
text, arithmetic, and shell-tool routing passed in all repetitions, while both
variants incorrectly selected `run_command("ls -a")` for the dedicated
`read_file` case. That result proves runtime and LoRA compatibility, not that the
adapter is ready to replace a held-out task-quality gate.

## Rollback

```bash
scripts/serve_gemma4_12b_nvfp4.sh stop
```

Stopping the service does not delete the downloaded model, training checkpoint,
adapter, benchmark evidence, or previous service configuration.
