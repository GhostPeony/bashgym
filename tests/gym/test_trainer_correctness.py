"""Execute generated trainer functions without loading any model."""

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from bashgym.gym.trainer import GRPOTrainer, Trainer, TrainerConfig, TrainingRun, TrainingStrategy


@pytest.mark.parametrize("settings", [{"use_lora": False}, {"use_liger": True}])
def test_unsloth_does_not_silently_ignore_adapter_or_kernel_settings(settings):
    with pytest.raises(ValueError):
        script(TrainingStrategy.SFT, sft_backend="unsloth", **settings)


def test_unsupported_backend_does_not_silently_select_unsloth():
    with pytest.raises(ValueError, match="unsupported"):
        script(TrainingStrategy.SFT, sft_backend="trl_vllm")


@pytest.mark.parametrize("method", ["embedding_similarity_trace_pair", "trace_pair"])
def test_direct_dpo_rejects_unverified_trace_pairs_before_run_registration(
    tmp_path, method, monkeypatch
):
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        json.dumps(
            {
                "prompt": "question",
                "chosen": "good answer",
                "rejected": "bad answer",
                "metadata": {"pair_generation_method": method},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    trainer = Trainer(
        TrainerConfig(
            base_model="test-model",
            load_in_4bit=False,
            dpo_backend="plain",
            output_dir=str(tmp_path / "output"),
        )
    )

    def no_training(*args, **kwargs):
        raise AssertionError("Invalid preference data reached the training boundary")

    monkeypatch.setattr(trainer, "_train_with_unsloth_dpo", no_training)
    with pytest.raises(ValueError, match="conditioning"):
        trainer.train_dpo(path)
    assert trainer.active_runs == {}


def test_direct_dpo_accepts_plain_user_preference_records(tmp_path):
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        json.dumps({"prompt": "question", "chosen": "good answer", "rejected": "bad answer"})
        + "\n",
        encoding="utf-8",
    )
    trainer = Trainer(TrainerConfig(base_model="test-model"))
    assert hasattr(trainer, "_validate_dpo_dataset")
    trainer._validate_dpo_dataset(path)


def script(strategy, **settings):
    config = TrainerConfig(base_model="test-model", strategy=strategy, **settings)
    run = TrainingRun(
        run_id="correctness",
        strategy=strategy,
        base_model=config.base_model,
        dataset_path=Path("data.jsonl"),
        output_path=Path("out"),
    )
    owner = GRPOTrainer(config) if strategy == TrainingStrategy.GRPO else Trainer(config)
    return getattr(owner, f"_generate_{strategy.value}_script")(run)


def definitions(source, names, **namespace):
    tree = ast.parse(source)
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    exec(compile(ast.Module(body=selected, type_ignores=[]), "<generated>", "exec"), namespace)
    return namespace


@pytest.mark.parametrize("backend", ["plain", "unsloth"])
@pytest.mark.parametrize(
    "tests, expected",
    [
        ("def test_pass():\n    assert True\n", (1, 1)),
        ("def test_fail():\n    assert False\n", (0, 1)),
        ("def test_pass():\n    assert True\ndef test_fail():\n    assert False\n", (1, 2)),
        (
            "import pytest\n@pytest.fixture\ndef broken():\n    raise RuntimeError()\ndef test_error(broken):\n    pass\n",
            (0, 1),
        ),
        ("raise RuntimeError('collection error')\n", (0, 1)),
        ("", (0, 1)),
        ("import pytest\n@pytest.mark.skip\ndef test_skip():\n    pass\n", (0, 1)),
    ],
)
def test_generated_verification_scores_junit_cases(backend, tests, expected, monkeypatch):
    source = script(TrainingStrategy.GRPO, grpo_backend=backend, load_in_4bit=False)
    real_run = subprocess.run

    def hidden_run(*args, **kwargs):
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        return real_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", hidden_run)
    fn = definitions(
        source,
        {"run_verification"},
        os=os,
        re=re,
        subprocess=subprocess,
        sys=sys,
        tempfile=tempfile,
    )["run_verification"]
    assert fn("answer = 42", tests) == expected


@pytest.mark.parametrize("backend", ["plain", "unsloth"])
def test_generated_verification_timeout_is_not_success(backend, monkeypatch):
    source = script(TrainingStrategy.GRPO, grpo_backend=backend, load_in_4bit=False)

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("pytest", 30)

    monkeypatch.setattr(subprocess, "run", timeout)
    fn = definitions(
        source,
        {"run_verification"},
        os=os,
        re=re,
        subprocess=subprocess,
        sys=sys,
        tempfile=tempfile,
    )["run_verification"]
    assert fn("", "def test_ok(): pass") == (0, 1)


@pytest.mark.parametrize("backend", ["plain", "unsloth"])
@pytest.mark.parametrize("tests", [None, [""]])
def test_verification_reward_requires_tests(backend, tests):
    source = script(TrainingStrategy.GRPO, grpo_backend=backend, load_in_4bit=False)
    reward = definitions(
        source, {"extract_code", "run_verification", "verification_reward"}, ast=ast, re=re
    )["verification_reward"]
    assert reward(["answer = 42"], ["question"], tests=tests) == [0.0]


@pytest.mark.parametrize("backend", ["plain", "unsloth"])
@pytest.mark.parametrize(
    "tests, status, counts",
    [
        ("def test_pass(): pass", "completed", (1, 0, 0, 0)),
        ("def test_fail(): assert False", "completed", (0, 1, 0, 0)),
        ("raise RuntimeError('collection')", "collection_error", (0, 0, 1, 0)),
        ("", "empty_suite", (0, 0, 0, 0)),
        ("import pytest\n@pytest.mark.skip\ndef test_skip(): pass", "completed", (0, 0, 0, 1)),
    ],
)
def test_verification_details_distinguish_outcomes(backend, tests, status, counts):
    source = script(TrainingStrategy.GRPO, grpo_backend=backend, load_in_4bit=False)
    fn = definitions(source, {"run_verification"})["run_verification"]
    result = fn("answer = 42", tests, detailed=True)
    assert result["status"] == status
    assert tuple(result[key] for key in ("passed", "failed", "errors", "skipped")) == counts


def test_verification_details_distinguish_timeout(monkeypatch):
    source = script(TrainingStrategy.GRPO, grpo_backend="plain", load_in_4bit=False)
    fn = definitions(source, {"run_verification"})["run_verification"]

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("pytest", 30)

    monkeypatch.setattr(subprocess, "run", timeout)
    assert fn("", "def test_ok(): pass", detailed=True)["status"] == "timeout"


@pytest.mark.parametrize(
    "strategy", [TrainingStrategy.SFT, TrainingStrategy.DPO, TrainingStrategy.GRPO]
)
@pytest.mark.parametrize(
    "options, field",
    [
        ({"load_in_4bit": True}, "load_in_4bit"),
        ({"load_in_4bit": False, "use_lora": False}, "use_lora"),
    ],
)
def test_plain_backend_rejects_settings_it_cannot_honor(strategy, options, field):
    with pytest.raises(ValueError, match=field):
        script(strategy, **{f"{strategy.value}_backend": "plain"}, **options)


class Tokenizer:
    chat_template = None

    def __call__(self, text, **kwargs):
        return {"input_ids": [int(token) for token in text.split()]}


def preprocess():
    source = script(TrainingStrategy.SESSION_DISTILLATION, max_seq_length=8)
    return definitions(source, {"encode_context_target", "preprocess"}, tokenizer=Tokenizer())[
        "preprocess"
    ]


def test_distillation_preserves_full_target_and_trims_long_contexts_from_left():
    row = preprocess()(
        {"original_context": "1 2", "hinted_context": "1 2 3 4 5 6", "target_text": "7 8 9"}
    )
    original = [x for x in row["original_labels"] if x != -100]
    hinted = [x for x in row["hinted_labels"] if x != -100]
    assert original == hinted == [7, 8, 9]
    assert row["hinted_input_ids"] == [2, 3, 4, 5, 6, 7, 8, 9]
    assert row["original_input_ids"] == [1, 2, 7, 8, 9]
    assert row["target_token_count"] == 3


@pytest.mark.parametrize("context, target", [("1", "1 2 3 4 5 6 7 8"), ("1", ""), ("", "9")])
def test_distillation_rejects_rows_without_a_predictable_common_target(context, target):
    with pytest.raises(ValueError, match="target|context"):
        preprocess()({"original_context": "1", "hinted_context": context, "target_text": target})


def loss_namespace():
    torch = pytest.importorskip("torch")
    source = script(TrainingStrategy.SESSION_DISTILLATION)

    class Base:
        def log(self, metrics):
            self.metrics = metrics

    ns = definitions(
        source,
        {"SessionDistillationTrainer"},
        Trainer=Base,
        torch=torch,
        F=torch.nn.functional,
        MASK_POLICY="target_span_only",
        TEMPERATURE=1.0,
        ALPHA=1.0,
    )
    return torch, ns["SessionDistillationTrainer"]()


@pytest.mark.parametrize("target", [None, "", "  ", 12])
def test_distillation_rejects_missing_or_nontext_target(target):
    with pytest.raises(ValueError, match="target_text"):
        preprocess()({"original_context": "1", "hinted_context": "2", "target_text": target})


@pytest.mark.parametrize("kind", ["unequal_per_row", "different_labels", "empty"])
def test_distillation_loss_rejects_incompatible_row_targets(kind):
    torch, trainer = loss_namespace()
    original = torch.tensor([[1, 2, 3], [1, 2, 3]])
    hinted = original.clone()
    original_mask = torch.tensor([[0, 1, 1], [0, 0, 1]])
    hinted_mask = original_mask.clone()
    if kind == "unequal_per_row":
        hinted_mask = torch.tensor([[0, 0, 1], [0, 1, 1]])
    elif kind == "different_labels":
        hinted[0, 1] = 4
    else:
        original_mask.zero_()
        hinted_mask.zero_()
    inputs = {}
    for prefix, ids, mask in [
        ("original", original, original_mask),
        ("hinted", hinted, hinted_mask),
    ]:
        inputs.update(
            {
                f"{prefix}_input_ids": ids,
                f"{prefix}_labels": ids.masked_fill(~mask.bool(), -100),
                f"{prefix}_attention_mask": torch.ones_like(ids),
                f"{prefix}_target_mask": mask,
            }
        )

    def model(**kwargs):
        return SimpleNamespace(logits=torch.zeros((2, 3, 5), requires_grad=True))

    with pytest.raises(ValueError, match="target|align"):
        trainer.compute_loss(model, inputs)


def test_distillation_kl_uses_matching_tokens_with_different_context_lengths():
    torch, trainer = loss_namespace()
    inputs = {}
    for prefix, ids, masks in [
        ("original", [[0, 1, 2, 0], [0, 1, 0, 0]], [[0, 1, 1, 0], [0, 1, 0, 0]]),
        ("hinted", [[0, 0, 1, 2], [0, 0, 0, 1]], [[0, 0, 1, 1], [0, 0, 0, 1]]),
    ]:
        ids, masks = torch.tensor(ids), torch.tensor(masks)
        inputs.update(
            {
                f"{prefix}_input_ids": ids,
                f"{prefix}_labels": ids.masked_fill(~masks.bool(), -100),
                f"{prefix}_attention_mask": torch.ones_like(ids),
                f"{prefix}_target_mask": masks,
            }
        )
    student = torch.zeros((2, 4, 3), requires_grad=True)
    teacher = torch.zeros((2, 4, 3))
    teacher[0, 1, :] = torch.tensor([0.0, 0.0, 1.0])
    outputs = iter([SimpleNamespace(logits=student), SimpleNamespace(logits=teacher)])
    loss = trainer.compute_loss(lambda **kwargs: next(outputs), inputs)
    # One of three target distributions differs: softmax(0,0,1) vs uniform.
    expected = 0.0410948198
    assert loss.item() == pytest.approx(expected, abs=1e-7)
    assert trainer.metrics["session_distillation_masked_tokens"] == 3
    loss.backward()
    assert student.grad[0, 0].abs().sum() > 0
    assert student.grad[0, 1:].abs().sum() == 0
