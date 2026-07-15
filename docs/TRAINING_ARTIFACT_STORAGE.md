# Training artifact storage

BashGym uses LoRA/QLoRA by default, but a training run can still consume a full model's worth of disk when it saves merged weights or deployment exports. The base-model Hugging Face cache is shared across runs; each run's adapters, checkpoints, merged model, and GGUF files are separate artifacts.

## Retention policies

| Policy | Final adapter | Resumable checkpoints | Merged model | GGUF option | Recommended use |
| --- | --- | --- | --- | --- | --- |
| Adapter only | Yes | Removed after success | No | No | Routine experiments and evaluation candidates |
| Adapter + checkpoint | Yes | Latest `checkpoint_limit` retained | No | No | Work that may need to resume or branch |
| Deployable | Yes | Removed after success | Yes | Optional | A candidate being prepared for local serving |
| Full run | Yes | Latest `checkpoint_limit` retained | Yes | Optional | Audited or promoted runs that need all artifacts |

`adapter_only` is the default. Checkpoints are still written during training so an interrupted job can be recovered; policies that do not retain checkpoints remove them only after the final adapter is saved successfully.

## Hugging Face upload

When Hugging Face is connected, a run can automatically upload to a private or public model repository. The artifact selector supports:

- `auto`: upload merged weights when present, otherwise the adapter;
- `adapter`: require and upload only the lightweight adapter;
- `merged`: require a standalone merged model.

The same selector is available from a completed model's **Push to Hugging Face Hub** dialog. Public repositories should be used only after reviewing the base-model license, training-data provenance, private information, and generated model card.

Hugging Face Storage Buckets remain available under the Hugging Face storage view for mutable checkpoints, logs, and intermediate artifacts. Model repositories are the preferred destination for versioned, promoted model releases.

## Cleanup

The Checkpoints view reports total local artifact storage and includes **Keep adapter only** for each run. This permanently removes that run's merged model, GGUF exports, and intermediate checkpoints while preserving its final adapter and metadata. Individual artifacts can also be deleted separately.

For remote training, the selected retention policy is embedded in the generated script, so the compute device does not create or retain unnecessary merged/checkpoint copies before BashGym downloads the final artifacts.
