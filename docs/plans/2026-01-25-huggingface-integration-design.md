# HuggingFace Pro Integration Design

> Full integration of HuggingFace Pro features into Bash Gym

**Goal:** Enable Bash Gym users with HuggingFace Pro subscriptions to leverage cloud training, inference providers, ZeroGPU Spaces, and 1TB dataset storage.

**Architecture:** New `bashgym/integrations/` module wrapping `huggingface_hub` library with graceful degradation for users without Pro.

---

## Architecture Overview

### New Module Structure

```
bashgym/
├── integrations/
│   ├── __init__.py
│   ├── huggingface.py      # Main HF client wrapper, Pro detection
│   ├── hf_jobs.py          # Cloud training via Jobs
│   ├── hf_inference.py     # Inference Providers API
│   ├── hf_spaces.py        # ZeroGPU Space deployment
│   └── hf_datasets.py      # Dataset upload/Data Studio
```

### Configuration Extension

```python
@dataclass
class HuggingFaceSettings:
    token: str                    # HF_TOKEN
    username: str                 # HF_USERNAME
    default_org: str | None       # HF_ORG (for billing)
    pro_enabled: bool = False     # Auto-detected from token
    storage_repo: str | None      # Where to store datasets/models
```

### Core Principle

All HF features are optional. Bash Gym works without HF Pro, but if `HF_TOKEN` is set and Pro is detected, additional capabilities unlock.

---

## Component 1: Cloud Training via Jobs

### Purpose

Submit training runs to HuggingFace GPUs instead of requiring local hardware.

### Interface

```python
@dataclass
class HFJobConfig:
    hardware: str = "a10g-small"  # t4-small, a10g-small, a10g-large, a100-large
    timeout_minutes: int = 30
    docker_image: str = "huggingface/transformers-pytorch-gpu"
    environment: dict = None      # HF_TOKEN, WANDB_KEY, etc.

class HFJobRunner:
    def submit_training_job(
        self,
        training_script: Path,
        dataset_repo: str,
        output_repo: str,
        config: HFJobConfig
    ) -> str:  # Returns job_id

    def get_job_status(self, job_id: str) -> JobStatus
    def get_job_logs(self, job_id: str) -> str
    def cancel_job(self, job_id: str) -> bool
```

### Integration with trainer.py

```python
class Trainer:
    def train(self, config: TrainerConfig):
        if config.use_hf_jobs and self.hf_enabled:
            return self._train_on_hf_jobs(config)
        else:
            return self._train_local(config)
```

### Hardware Options

| Flavor | GPU | VRAM | Best For |
|--------|-----|------|----------|
| `t4-small` | T4 | 16GB | Small models (<3B) |
| `a10g-small` | A10G | 24GB | Medium models (3-7B) |
| `a10g-large` | A10G | 24GB | Longer training |
| `a100-large` | A100 | 80GB | Large models (7B+) |

---

## Component 2: Inference Providers

### Purpose

Use HF's multi-provider inference API as alternative/supplement to Claude and NVIDIA NIM.

### Interface

```python
@dataclass
class HFInferenceConfig:
    provider: str = "auto"        # auto, together, replicate, sambanova, fal
    routing: str = "fastest"      # fastest, cheapest
    bill_to: str | None = None
    timeout: int = 30

class HFInferenceClient:
    def generate(self, model: str, prompt: str, **kwargs) -> str
    def embed(self, model: str, texts: list[str]) -> list[list[float]]
    def classify(self, model: str, text: str) -> dict
```

### Integration with model_router.py

```python
class RoutingStrategy(Enum):
    TEACHER_ONLY = "teacher_only"
    STUDENT_ONLY = "student_only"
    HF_INFERENCE = "hf_inference"      # NEW
    CONFIDENCE_BASED = "confidence"
    COST_OPTIMIZED = "cost_optimized"  # NEW
```

### Use Cases

| Use Case | Model | Provider |
|----------|-------|----------|
| Data augmentation | `meta-llama/Llama-3.3-70B-Instruct` | Together |
| Trace quality scoring | `mistralai/Mistral-7B-Instruct` | Sambanova |
| Embedding for dedup | `BAAI/bge-large-en-v1.5` | HF Serverless |
| Cheap bulk generation | `Qwen/Qwen2.5-72B-Instruct:cheapest` | Auto-routed |

---

## Component 3: ZeroGPU Spaces

### Purpose

Deploy trained student models to HuggingFace Spaces for inference testing and demos.

### Interface

```python
@dataclass
class SpaceConfig:
    name: str
    hardware: str = "zero-gpu"
    private: bool = True
    dev_mode: bool = False

class HFSpaceManager:
    def create_inference_space(
        self,
        model_repo: str,
        space_name: str,
        config: SpaceConfig
    ) -> str:  # Returns Space URL

    def update_space_model(self, space_name: str, new_model_repo: str)
    def get_space_status(self, space_name: str) -> SpaceStatus
    def delete_space(self, space_name: str)
    def enable_dev_mode(self, space_name: str) -> SSHCredentials
```

### Auto-Generated Gradio App

```python
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import spaces

model_id = "{{MODEL_REPO}}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

@spaces.GPU(duration=60)
def generate(prompt: str, max_tokens: int = 512):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

demo = gr.Interface(fn=generate, inputs=["text", gr.Slider(64, 1024)], outputs="text")
demo.launch()
```

---

## Component 4: Dataset Storage & Data Studio

### Purpose

Upload training datasets to HuggingFace Hub, leverage 1TB storage and Data Studio viewer.

### Interface

```python
@dataclass
class DatasetConfig:
    repo_name: str
    private: bool = True
    enable_viewer: bool = True

class HFDatasetManager:
    def upload_training_data(
        self,
        local_path: Path,
        config: DatasetConfig
    ) -> str:

    def upload_traces(self, traces: list[dict], config: DatasetConfig) -> str
    def download_dataset(self, repo_id: str, local_path: Path)
    def list_datasets(self, prefix: str = "bashgym") -> list[str]
    def delete_dataset(self, repo_id: str)
```

### Dataset Structure

```
username/bashgym-training-v1/
├── train.jsonl
├── val.jsonl
├── metadata.json
└── README.md            # Auto-generated dataset card
```

---

## API Endpoints

```python
# HuggingFace Status
GET  /api/hf/status

# Cloud Training
POST /api/hf/jobs
GET  /api/hf/jobs
GET  /api/hf/jobs/{id}
DELETE /api/hf/jobs/{id}

# Inference
POST /api/hf/inference/generate
POST /api/hf/inference/embed
GET  /api/hf/inference/usage

# Spaces
POST /api/hf/spaces
GET  /api/hf/spaces
DELETE /api/hf/spaces/{name}

# Datasets
POST /api/hf/datasets
GET  /api/hf/datasets
```

### WebSocket Events

```python
"hf:job:started"
"hf:job:log"
"hf:job:completed"
"hf:space:ready"
```

---

## Frontend Components

```
frontend/src/components/
├── HFStatus.tsx           # Pro badge, credits remaining
├── CloudTraining.tsx      # Job submission form, hardware picker
├── JobMonitor.tsx         # Live job logs, progress
├── SpaceManager.tsx       # List/create/delete Spaces
└── DatasetBrowser.tsx     # Link to Data Studio, upload button
```

---

## Configuration

### Environment Variables

```bash
# HuggingFace Integration (Optional)
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
HF_USERNAME=your-username
HF_ORG=                           # Optional: bill to org
HF_STORAGE_REPO=                  # Optional: default repo
```

### Feature Availability Matrix

| Feature | No Token | Free Token | Pro Token |
|---------|----------|------------|-----------|
| Upload public datasets | ❌ | ✓ | ✓ |
| Upload private datasets | ❌ | Limited | ✓ (1TB) |
| Data Studio on private | ❌ | ❌ | ✓ |
| Inference API | ❌ | Limited | ✓ ($2/mo) |
| ZeroGPU Spaces | ❌ | ❌ | ✓ (10 max) |
| Jobs (cloud training) | ❌ | ❌ | ✓ |

---

## Error Handling

```python
class HFError(Exception): pass
class HFAuthError(HFError): pass
class HFProRequiredError(HFError): pass
class HFQuotaExceededError(HFError): pass
class HFJobFailedError(HFError): pass
```

---

## Summary

| Component | File | Purpose |
|-----------|------|---------|
| Core wrapper | `huggingface.py` | Token validation, Pro detection |
| Jobs | `hf_jobs.py` | Cloud training on HF GPUs |
| Inference | `hf_inference.py` | Multi-provider inference API |
| Spaces | `hf_spaces.py` | Deploy models to ZeroGPU |
| Datasets | `hf_datasets.py` | Upload data, Data Studio |
| API | `routes.py` additions | REST endpoints |
| Frontend | New components | Dashboard integration |
