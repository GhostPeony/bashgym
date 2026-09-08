"""Public CLI connects task preparation to the existing NeMo bundle transport."""

import json

from bashgym import cli


def run(argv, capsys):
    args = cli.build_parser().parse_args([*argv, "--json"])
    code = args.func(args)
    return code, json.loads(capsys.readouterr().out)


def test_build_inspect_export_coding_bundle_without_compute(tmp_path, capsys):
    dataset, bundle, archive = tmp_path / "dataset", tmp_path / "bundle", tmp_path / "bundle.zip"
    code, built = run(["environments", "build-coding", "--output", str(dataset)], capsys)
    assert code == 0 and built["compute_started"] is False
    code, inspected = run(
        ["environments", "inspect-coding", "--dataset", str(dataset), "--split", "dev"], capsys
    )
    assert code == 0 and len(inspected["tasks"]) == 2
    assert {item["split"] for item in inspected["tasks"]} == {"dev"}
    code, exported = run(
        [
            "environments",
            "export-nemo",
            "--dataset",
            str(dataset),
            "--output",
            str(bundle),
            "--archive",
            str(archive),
            "--nemo-gym-revision",
            "a" * 40,
            "--bashgym-revision",
            "b" * 40,
            "--sandbox-image",
            "sha256:" + "c" * 64,
        ],
        capsys,
    )
    assert code == 0 and exported["verified"] is False
    assert exported["compute_started"] is False
    from bashgym.environments.nemo_gym import inspect_nemo_gym_bundle_archive

    assert inspect_nemo_gym_bundle_archive(archive)["environment_id"] == "personal-coding-v1"
