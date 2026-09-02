"""Value-content scanning for proposals."""

from bashgym.campaigns.secret_scan import scan_values


def test_scan_reports_credential_shaped_values_by_path_without_echoing_them() -> None:
    findings = scan_values({"training_recipe": {"note": "ghp_" + "a" * 36}, "ok": "plain"})

    assert [(item.path, item.kind) for item in findings] == [("training_recipe.note", "credential")]
    assert "ghp_" not in repr(findings)


def test_scan_reports_unresolved_placeholders() -> None:
    findings = scan_values(
        {
            "evaluation_recipe": {"api_base": "<ASK_USER: base url>"},
            "hypothesis": "REPLACE_ME_hypothesis",
        }
    )

    assert sorted((item.path, item.kind) for item in findings) == [
        ("evaluation_recipe.api_base", "placeholder"),
        ("hypothesis", "placeholder"),
    ]


def test_scan_ignores_ordinary_prose_numbers_and_short_tokens() -> None:
    assert (
        scan_values(
            {
                "hypothesis": "Bearer tokens are unrelated to this hypothesis.",
                "seed": 42,
                "script_args": ["--seed", "17", "--token-budget", "4096"],
            }
        )
        == ()
    )


def test_scan_reports_one_unscannable_finding_when_depth_is_exceeded() -> None:
    nested: dict = {"leaf": "plain"}
    for _ in range(40):
        nested = {"child": nested}

    findings = scan_values(nested)

    assert [item.kind for item in findings] == ["unscannable"]
