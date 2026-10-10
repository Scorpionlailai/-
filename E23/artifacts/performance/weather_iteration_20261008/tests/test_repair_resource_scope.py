"""Forecast resource chains must not contaminate repair-only cost tables."""
import importlib.util
import json
from pathlib import Path

import pytest


def load_module():
    path=Path(__file__).resolve().parents[1]/'repair_resource_scope.py'
    spec=importlib.util.spec_from_file_location('repair_resource_scope',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def manifest(tmp_path,name,stage,**extra):
    p=tmp_path/(name+'.json');p.write_text(json.dumps(dict(stage=stage,**extra)));return p


def test_legacy_forecast_evaluation_uses_only_repair_cost(tmp_path):
    stage1=manifest(tmp_path,'imputer','repair',selection_metric='mse')
    stage2=manifest(tmp_path,'predictor','forecast',parent_selection=str(stage1))
    record={'training_resources':{'selection_manifest_paths':[str(stage1),str(stage2)]}}
    selected,paths=load_module().stage1_manifests(record)
    assert selected==stage1 and paths==[stage1]


def test_refinement_counts_coarse_parent_once_and_labels_leaf_selection(tmp_path):
    parent=manifest(tmp_path,'coarse','repair')
    child=manifest(tmp_path,'refinement_mae','repair',selection_metric='mae',parent_selection=str(parent))
    record={'training_resources':{'selection_manifest_paths':[str(parent),str(child)]}}
    selected,paths=load_module().stage1_manifests(record)
    assert selected==child and paths==[parent,child]
    assert json.loads(selected.read_text())['selection_metric']=='mae'


def test_missing_repair_manifest_is_not_reported_as_zero_cost(tmp_path):
    forecast=manifest(tmp_path,'predictor','forecast')
    with pytest.raises(ValueError,match='Stage1 repair'):
        load_module().stage1_manifests({'training_resources':{'selection_manifest_paths':[str(forecast)]}})


def test_shared_training_parent_is_counted_once_for_two_calibrations(tmp_path):
    coarse=manifest(tmp_path,'coarse','repair')
    a=manifest(tmp_path,'a','repair',parent_selection=str(coarse))
    b=manifest(tmp_path,'b','repair',parent_selection=str(coarse))
    ensemble=manifest(tmp_path,'ensemble','repair',parent_selection=str(b),additional_parent_selections=[str(a)])
    selected,paths=load_module().stage1_manifests({'training_resources':{'selection_manifest_paths':[str(coarse),str(a),str(b),str(ensemble)]}})
    assert selected==ensemble and paths==[coarse,a,b,ensemble]
