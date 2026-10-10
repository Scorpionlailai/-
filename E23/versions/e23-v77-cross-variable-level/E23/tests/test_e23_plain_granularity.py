from dataclasses import replace
from pathlib import Path
import pytest
import torch
import yaml
from integrations.e23 import train as training
from integrations.e23.config import load_e23_settings
from integrations.e23.forecast_sensitive import parameter_hash
from tests.integration.test_e23_stage1 import _protocol_settings,_batch

ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('p',[16,8,4])
def test_plain_removes_branch_preserves_base_head_and_initial_function(tmp_path,p):
    protocol,_=_protocol_settings()
    raw=yaml.safe_load((ROOT/f'configs/e23_v45_plain{p}_seed502.yaml').read_text())
    raw['training']['seed']=protocol.training_seed
    config=tmp_path/'plain.yaml';config.write_text(yaml.safe_dump(raw))
    s=load_e23_settings(config,protocol=protocol)
    torch.manual_seed(502);plain=training.build_forecast_model(protocol,s,device=torch.device('cpu')).eval()
    torch.manual_seed(502);hybrid=training.build_forecast_model(protocol,replace(s,point_refiner_arch='hybrid_v1',point_refiner_span=32),device=torch.device('cpu')).eval()
    assert plain.repair.point_refiner is None
    assert not any('point_refiner' in k for k in plain.repair.state_dict())
    assert parameter_hash(plain.predictor)==parameter_hash(hybrid.predictor)
    for k,v in plain.repair.state_dict().items():
        torch.testing.assert_close(v,hybrid.repair.state_dict()[k],rtol=0,atol=0)
    b=_batch('validation');starts=training._starts(b,stride=p,patches=96//p,device=torch.device('cpu'))
    with torch.no_grad():
        pa,oa=plain(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
        pb,ob=hybrid(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
    torch.testing.assert_close(pa,pb,rtol=0,atol=0)
    torch.testing.assert_close(oa.explicit.normalized_sequence,ob.explicit.normalized_sequence,rtol=0,atol=0)
    assert oa.explicit.coverage.eq(1).all()
    identity=training._trial_identity(s)
    assert identity['point_refiner']=='none' and identity['point_refiner_span']==0
    assert not any(k.startswith('hybrid_') for k in identity)
    # Fine geometry cannot silently reintroduce a dense branch.
    raw['model']['point_refiner_span']=32;config.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):load_e23_settings(config,protocol=protocol)
