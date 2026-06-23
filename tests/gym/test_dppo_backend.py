import pytest

from bashgym.gym.dppo_backend import (
    DPPOBackendCapability,
    probe_dppo_backends,
    select_dppo_backend,
)


def _capabilities(**available: bool) -> dict[str, DPPOBackendCapability]:
    return {
        name: DPPOBackendCapability(
            name=name,
            available=available.get(name, False),
            reason=f"{name} {'available' if available.get(name, False) else 'missing'}",
        )
        for name in ("verl", "skyrl", "tmax_open_instruct")
    }


def test_auto_selects_first_available_backend_by_priority():
    selection = select_dppo_backend(
        "auto",
        capabilities=_capabilities(skyrl=True, tmax_open_instruct=True),
    )

    assert selection.selected == "skyrl"
    assert selection.available is True
    assert selection.fallback_to_grpo is False


def test_auto_falls_back_to_grpo_when_no_dppo_backend_is_available():
    selection = select_dppo_backend("auto", capabilities=_capabilities())

    assert selection.selected == "grpo_fallback"
    assert selection.available is False
    assert selection.fallback_to_grpo is True
    assert "No DPPO-capable backend detected" in selection.reason


def test_explicit_backend_falls_back_with_reason_when_unavailable():
    selection = select_dppo_backend("verl", capabilities=_capabilities())

    assert selection.requested == "verl"
    assert selection.selected == "grpo_fallback"
    assert selection.fallback_to_grpo is True
    assert "Requested verl" in selection.reason


def test_explicit_grpo_fallback_is_supported():
    selection = select_dppo_backend(
        "grpo_fallback",
        capabilities=_capabilities(verl=True),
    )

    assert selection.selected == "grpo_fallback"
    assert selection.available is False
    assert selection.fallback_to_grpo is True
    assert selection.reason == "GRPO fallback explicitly requested"


def test_invalid_backend_raises():
    with pytest.raises(ValueError, match="dppo_backend"):
        select_dppo_backend("mystery", capabilities=_capabilities())


def test_probe_detects_modules_commands_and_checkout_paths():
    def find_spec(name: str):
        return object() if name == "skyrl" else None

    def which(command: str) -> str | None:
        return "C:/tools/verl.exe" if command == "verl" else None

    caps = probe_dppo_backends(
        env={"TMAX_OPEN_INSTRUCT_DIR": "C:/src/open-instruct"},
        find_spec=find_spec,
        which=which,
        path_exists=lambda path: path == "C:/src/open-instruct",
    )

    assert caps["verl"].available is True
    assert caps["verl"].command == "C:/tools/verl.exe"
    assert caps["skyrl"].available is True
    assert caps["tmax_open_instruct"].available is True
    assert caps["tmax_open_instruct"].path == "C:/src/open-instruct"
