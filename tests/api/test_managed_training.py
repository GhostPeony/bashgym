"""Tests for the managed fine-tune training routes."""

from fastapi.testclient import TestClient

from bashgym.api.routes import app
from bashgym.gym.training_backends import TrainingJob, TrainingStatus

client = TestClient(app)


def _dataset(tmp_path):
    ds = tmp_path / "train.jsonl"
    ds.write_text('{"messages": []}\n')
    return str(ds)


def test_submit_invalid_platform(tmp_path):
    r = client.post(
        "/api/training/managed/submit",
        json={
            "platform": "nope",
            "base_model": "m",
            "dataset_path": _dataset(tmp_path),
            "api_key": "k",
        },
    )
    assert r.status_code == 400


def test_submit_requires_model_and_dataset():
    r = client.post("/api/training/managed/submit", json={"platform": "together", "api_key": "k"})
    assert r.status_code == 400


def test_submit_missing_dataset_file(tmp_path):
    r = client.post(
        "/api/training/managed/submit",
        json={
            "platform": "together",
            "base_model": "m",
            "dataset_path": str(tmp_path / "nope.jsonl"),
            "api_key": "k",
        },
    )
    assert r.status_code == 400


def test_submit_requires_api_key(tmp_path):
    # no body key + no connected provider for the platform -> 400
    r = client.post(
        "/api/training/managed/submit",
        json={"platform": "together", "base_model": "m", "dataset_path": _dataset(tmp_path)},
    )
    assert r.status_code == 400


def test_submit_success(tmp_path, monkeypatch):
    async def fake_submit(self, spec):
        assert spec.base_model == "meta/Llama-3"
        assert spec.n_epochs == 2
        return TrainingJob(job_id="ft-1", backend=self.backend_type, status=TrainingStatus.PENDING)

    monkeypatch.setattr("bashgym.gym.training_backends.ManagedFineTuneBackend.submit", fake_submit)
    r = client.post(
        "/api/training/managed/submit",
        json={
            "platform": "together",
            "base_model": "meta/Llama-3",
            "dataset_path": _dataset(tmp_path),
            "n_epochs": 2,
            "api_key": "k",
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["job_id"] == "ft-1"
    assert data["backend"] == "together"
    assert data["status"] == "pending"


def test_poll(monkeypatch):
    async def fake_poll(self, job):
        return TrainingJob(
            job_id=job.job_id,
            backend=self.backend_type,
            status=TrainingStatus.SUCCEEDED,
            output_model="user/model-ft",
        )

    monkeypatch.setattr("bashgym.gym.training_backends.ManagedFineTuneBackend.poll", fake_poll)
    r = client.get("/api/training/managed/together/ft-1")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "succeeded"
    assert data["output_model"] == "user/model-ft"
