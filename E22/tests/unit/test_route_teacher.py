from __future__ import annotations

import torch
import pytest

from tests.integration.test_online_stage1_v2 import _v2_fixture


def test_install_copies_and_freezes_the_complete_route_scoring_snapshot() -> None:
    """A teacher must be an immutable value-copy, not a live Stage-1 view."""

    model, _ = _v2_fixture()
    digest = model.install_frozen_route_teacher()

    assert model.has_frozen_route_teacher
    assert len(digest) == 64
    assert model.route_teacher is not None
    assert model.route_teacher.state_sha256() == digest
    assert all(not parameter.requires_grad for parameter in model.route_teacher.parameters())
    assert not model.route_teacher.training
    for teacher_module, source_module in (
        (model.route_teacher.patch_projection, model.core.W_P),
        (model.route_teacher.historical_operator, model.historical_operator),
        (model.route_teacher.temporal_operator, model.temporal_operator),
        (model.route_teacher.variable_operator, model.variable_operator),
    ):
        teacher_state = teacher_module.state_dict()
        source_state = source_module.state_dict()
        assert teacher_state.keys() == source_state.keys()
        for name in teacher_state:
            assert torch.equal(teacher_state[name], source_state[name])
            assert teacher_state[name].data_ptr() != source_state[name].data_ptr()


def test_installed_teacher_stays_in_eval_mode_and_can_only_be_installed_once() -> None:
    """Parent training mode must not enable teacher dropout or replace its snapshot."""

    model, _ = _v2_fixture()
    model.install_frozen_route_teacher()

    model.train()

    assert model.training
    assert model.route_teacher is not None
    assert not model.route_teacher.training
    with pytest.raises(RuntimeError, match="already installed"):
        model.install_frozen_route_teacher()


def test_teacher_hash_rejects_a_nonfinite_state_tensor() -> None:
    """A corrupt serialized scoring snapshot must not receive a provenance hash."""

    model, _ = _v2_fixture()
    model.install_frozen_route_teacher()
    assert model.route_teacher is not None
    with torch.no_grad():
        model.route_teacher.patch_projection.weight[0, 0] = torch.nan

    with pytest.raises(ValueError, match="finite"):
        model.route_teacher.state_sha256()


def test_frozen_forecast_configuration_requires_an_installed_teacher() -> None:
    """Frozen-teacher mode must never silently fall back to mutable route scores."""

    model, _ = _v2_fixture()

    with pytest.raises(RuntimeError, match="install"):
        model.configure_forecast_finetuning(frozen_route_teacher=True)


def test_teacher_hv_configuration_requires_both_teacher_and_fusion() -> None:
    """E21 must not silently fall back when either frozen component is absent."""

    model, _ = _v2_fixture()
    with pytest.raises(RuntimeError, match="teacher"):
        model.configure_forecast_finetuning(forecast_fusion="teacher_hv")

    model.install_frozen_route_teacher()
    with pytest.raises(RuntimeError, match="fusion"):
        model.configure_forecast_finetuning(forecast_fusion="teacher_hv")


def test_forecast_hv_fusion_installs_once_on_the_model_dtype() -> None:
    """A late-installed E21 container must match its already-moved parent model."""

    model, _ = _v2_fixture()
    model.double()

    model.install_forecast_hv_fusion()

    assert model.has_forecast_hv_fusion
    assert model.forecast_fusion is not None
    assert all(parameter.requires_grad for parameter in model.forecast_fusion.parameters())
    assert all(parameter.dtype == torch.float64 for parameter in model.forecast_fusion.parameters())
    with pytest.raises(RuntimeError, match="already installed"):
        model.install_forecast_hv_fusion()
