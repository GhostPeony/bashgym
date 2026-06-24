"""SystemInfoService.get_model_recommendations estimator wiring.

Verifies the recommendation layer can be driven by an explicit VRAM budget (e.g.
a remote SSH device's discovered ``effective_vram_gb``) and exposes per-regime
capacity from the hardware estimator, while preserving the existing fields.
"""

from bashgym.api.system_info import get_system_info_service


def test_recommendations_accept_a_budget_override_and_report_regime_capacity():
    service = get_system_info_service()

    rec = service.get_model_recommendations(vram_gb=128.0, unified_memory=True)

    # existing contract preserved
    assert rec["max_vram_gb"] == 128.0
    assert "recommended_models" in rec
    # new estimator-driven fields
    assert rec["regime_capacities"]["full_finetune"] == 8.0  # 128 / 16
    assert rec["regime_capacities"]["qlora"] == 128.0
    assert rec["unified_memory"] is True


def test_recommendations_default_to_detected_hardware():
    service = get_system_info_service()

    rec = service.get_model_recommendations()

    # no override -> still returns the full contract, capacities derived from
    # detected VRAM (which may be 0 on a CPU-only CI host)
    assert "regime_capacities" in rec
    assert rec["unified_memory"] is False
