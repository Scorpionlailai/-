"""Multi-variable settings must be explicit and leave the old lock intact."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from integrations.e23.config import load_e23_settings


@pytest.mark.parametrize('count,topk', [(21,4),(21,8),(64,8)])
def test_explicit_scaling_settings_accept_candidate_budget(tmp_path,count,topk):
    document=yaml.safe_load(Path('/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_stage1_mean_and_dense_20261008/e23_v97_stage1_mean_and_dense.yaml').read_text())
    document['model']['variable_topk']=topk
    path=tmp_path/'settings.yaml'
    path.write_text(yaml.safe_dump(document))
    protocol=SimpleNamespace(training_seed=2023,bounds=SimpleNamespace(name='Scaling'),config=SimpleNamespace(
        data=SimpleNamespace(seq_len=96,pred_len=96,datasets={'Scaling':SimpleNamespace(n_vars=count)}),
        patch=SimpleNamespace(length=16,stride=8)))
    settings=load_e23_settings(path,protocol=protocol,allow_variable_scaling=True)
    assert settings.variable_topk == topk
    assert settings.variable_count == count
    from integrations.e23.train import _trial_identity
    identity = _trial_identity(settings)
    assert identity['global_dense_features'] == 5*count
    assert identity['global_dense_extra_parameters'] == 77952+385*count
    with pytest.raises(ValueError):
        load_e23_settings(path,protocol=protocol)


def test_dynamic_scaling_config_preserves_policy_and_requires_cap(tmp_path):
    from malr.models.e23.variable import DYNAMIC_POLICY
    document=yaml.safe_load(Path('/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_stage1_mean_and_dense_20261008/e23_v97_stage1_mean_and_dense.yaml').read_text())
    document['model']['variable_topk']=16
    document['model']['variable_selection']=DYNAMIC_POLICY
    path=tmp_path/'dynamic.yaml'
    protocol=SimpleNamespace(training_seed=2023,bounds=SimpleNamespace(name='Scaling'),config=SimpleNamespace(
        data=SimpleNamespace(seq_len=96,pred_len=96,datasets={'Scaling':SimpleNamespace(n_vars=21)}),
        patch=SimpleNamespace(length=16,stride=8)))
    path.write_text(yaml.safe_dump(document))
    settings=load_e23_settings(path,protocol=protocol,allow_variable_scaling=True)
    assert settings.variable_selection==DYNAMIC_POLICY
    assert settings.variable_candidate_limit==48
    with pytest.raises(ValueError):
        load_e23_settings(path,protocol=protocol)
    document['model']['variable_topk']=8
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError,match='cap16'):
        load_e23_settings(path,protocol=protocol,allow_variable_scaling=True)
