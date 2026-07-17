from pathlib import Path
import re


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "generate_dpo_dataset.py"


def test_dpo_dataset_loader_has_no_private_checkout_env_fallback() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert not re.search(
        r'Path\.home\(\)\s*/\s*"desktop-home"\s*/\s*"Projects"\s*/\s*"ghostwork"\s*/\s*"\.env"',
        source,
    )
    assert "desktop-home" not in source
    assert 'Path.home() / "bashgym"' not in source
    assert 'Path.home() / ".bashgym" / ".env"' in source
    assert 'default=Path.home() / ".bashgym" / "gold_traces"' in source
    assert 'default=Path.home() / ".bashgym" / "data" / "dpo_synthetic"' in source
    assert "api_key[:" not in source
    assert "anth_key[:" not in source
