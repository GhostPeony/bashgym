"""Tests for the managed fine-tuning training backends (Together / OpenAI dialects)."""

import json

import httpx
import pytest

from bashgym.gym.training_backends import (
    ManagedFineTuneBackend,
    TrainingJob,
    TrainingSpec,
    TrainingStatus,
)


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _spec(tmp_path):
    ds = tmp_path / "train.jsonl"
    ds.write_text('{"messages": []}\n')
    return TrainingSpec(
        base_model="meta-llama/Llama-3-8B", dataset_path=ds, n_epochs=2, suffix="bashgym"
    )


class TestStatus:
    def test_terminal_flags(self):
        assert TrainingStatus.SUCCEEDED.terminal
        assert TrainingStatus.FAILED.terminal
        assert TrainingStatus.CANCELLED.terminal
        assert not TrainingStatus.RUNNING.terminal
        assert not TrainingStatus.PENDING.terminal


class TestForPlatform:
    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="unknown platform"):
            ManagedFineTuneBackend.for_platform("nope", api_key="k")

    def test_known(self):
        assert (
            ManagedFineTuneBackend.for_platform("together", api_key="k").backend_type == "together"
        )


class TestOpenAIDialect:
    async def test_submit_uploads_then_creates_nested_payload(self, tmp_path):
        captured = {}

        def handler(req):
            path = req.url.path
            if path.endswith("/files"):
                return httpx.Response(200, json={"id": "file-1"})
            if path.endswith("/fine_tuning/jobs") and req.method == "POST":
                captured["payload"] = json.loads(req.content)
                return httpx.Response(200, json={"id": "ftjob-1", "status": "queued"})
            return httpx.Response(404)

        backend = ManagedFineTuneBackend.for_platform(
            "openai", api_key="k", client=_client(handler)
        )
        job = await backend.submit(_spec(tmp_path))
        assert job.job_id == "ftjob-1"
        assert job.status == TrainingStatus.PENDING
        assert job.backend == "openai"
        assert captured["payload"]["training_file"] == "file-1"
        assert (
            captured["payload"]["hyperparameters"]["n_epochs"] == 2
        )  # OpenAI nests hyperparameters

    async def test_poll_succeeded(self, tmp_path):
        def handler(req):
            return httpx.Response(
                200, json={"status": "succeeded", "fine_tuned_model": "ft:gpt:abc"}
            )

        backend = ManagedFineTuneBackend.for_platform(
            "openai", api_key="k", client=_client(handler)
        )
        job = await backend.poll(TrainingJob(job_id="ftjob-1", backend="openai"))
        assert job.status == TrainingStatus.SUCCEEDED
        assert job.output_model == "ft:gpt:abc"


class TestTogetherDialect:
    async def test_submit_uses_together_path_and_flat_payload(self, tmp_path):
        captured = {}

        def handler(req):
            path = req.url.path
            if path.endswith("/files"):
                return httpx.Response(200, json={"id": "file-2"})
            if path.endswith("/fine-tunes") and req.method == "POST":
                captured["payload"] = json.loads(req.content)
                captured["path"] = path
                return httpx.Response(200, json={"id": "ft-2", "status": "pending"})
            return httpx.Response(404)

        backend = ManagedFineTuneBackend.for_platform(
            "together", api_key="k", client=_client(handler)
        )
        job = await backend.submit(_spec(tmp_path))
        assert job.job_id == "ft-2"
        assert captured["path"].endswith("/fine-tunes")  # not OpenAI's /fine_tuning/jobs
        assert captured["payload"]["n_epochs"] == 2  # flat, not nested
        assert "learning_rate" in captured["payload"]

    async def test_poll_completed_maps_to_succeeded(self, tmp_path):
        def handler(req):
            return httpx.Response(200, json={"status": "completed", "output_name": "user/model-ft"})

        backend = ManagedFineTuneBackend.for_platform(
            "together", api_key="k", client=_client(handler)
        )
        job = await backend.poll(TrainingJob(job_id="ft-2", backend="together"))
        assert job.status == TrainingStatus.SUCCEEDED
        assert job.output_model == "user/model-ft"  # Together's result field

    async def test_error_status_maps_to_failed(self, tmp_path):
        def handler(req):
            return httpx.Response(200, json={"status": "error", "error": "OOM on node"})

        backend = ManagedFineTuneBackend.for_platform(
            "together", api_key="k", client=_client(handler)
        )
        job = await backend.poll(TrainingJob(job_id="ft-2", backend="together"))
        assert job.status == TrainingStatus.FAILED
        assert "OOM" in job.error


class TestFireworksDialect:
    def test_requires_account_id(self):
        with pytest.raises(ValueError, match="requires an account_id"):
            ManagedFineTuneBackend.for_platform("fireworks", api_key="k")

    async def test_submit_two_step_dataset_then_camelcase_job(self, tmp_path):
        captured: dict = {"hits": set()}

        def handler(req):
            path = req.url.path
            captured["hits"].add(path.split("/v1")[-1])
            if path.endswith("/datasets") and req.method == "POST":
                return httpx.Response(200, json={"datasetId": "bashgym-x"})
            if ":upload" in path:
                return httpx.Response(200, json={})
            if path.endswith("/supervisedFineTuningJobs") and req.method == "POST":
                captured["payload"] = json.loads(req.content)
                captured["job_path"] = path
                return httpx.Response(
                    200,
                    json={
                        "name": "accounts/acct-1/supervisedFineTuningJobs/ftj-9",
                        "state": "JOB_STATE_RUNNING",
                        "outputModel": "accounts/acct-1/models/m",
                    },
                )
            return httpx.Response(404)

        backend = ManagedFineTuneBackend.for_platform(
            "fireworks", api_key="k", account_id="acct-1", client=_client(handler)
        )
        job = await backend.submit(_spec(tmp_path))

        assert job.job_id == "ftj-9"  # resource name reduced to the short id
        assert job.status == TrainingStatus.RUNNING
        assert job.output_model == "accounts/acct-1/models/m"
        # account-scoped path + two-step dataset upload happened
        assert "/accounts/acct-1/" in captured["job_path"]
        assert any(h.endswith("/datasets") for h in captured["hits"])
        assert any(":upload" in h for h in captured["hits"])
        # camelCase job payload referencing the dataset resource
        p = captured["payload"]
        assert p["baseModel"] == "meta-llama/Llama-3-8B"
        assert p["dataset"].startswith("accounts/acct-1/datasets/")
        assert p["epochs"] == 2 and "learningRate" in p

    async def test_poll_completed_and_failed(self, tmp_path):
        def ok(req):
            return httpx.Response(
                200, json={"state": "JOB_STATE_COMPLETED", "outputModel": "accounts/a/models/m"}
            )

        backend = ManagedFineTuneBackend.for_platform(
            "fireworks", api_key="k", account_id="a", client=_client(ok)
        )
        job = await backend.poll(TrainingJob(job_id="ftj-9", backend="fireworks"))
        assert job.status == TrainingStatus.SUCCEEDED
        assert job.output_model == "accounts/a/models/m"

        def failed(req):
            return httpx.Response(
                200, json={"state": "JOB_STATE_FAILED", "status": {"code": 13, "message": "boom"}}
            )

        backend2 = ManagedFineTuneBackend.for_platform(
            "fireworks", api_key="k", account_id="a", client=_client(failed)
        )
        job2 = await backend2.poll(TrainingJob(job_id="ftj-9", backend="fireworks"))
        assert job2.status == TrainingStatus.FAILED
        assert "boom" in job2.error


class TestUnknownStatus:
    async def test_unrecognized_status_is_unknown(self, tmp_path):
        def handler(req):
            return httpx.Response(200, json={"status": "weird_new_status"})

        backend = ManagedFineTuneBackend.for_platform(
            "openai", api_key="k", client=_client(handler)
        )
        job = await backend.poll(TrainingJob(job_id="x", backend="openai"))
        assert job.status == TrainingStatus.UNKNOWN
