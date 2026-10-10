"""A calibration ensemble shares coarse input and protects its observations."""
from types import SimpleNamespace
import pytest
import torch

from malr.models.e23.calibration_ensemble import RecentMeanEnsembleProjector
from malr.data.patching import patchify_preserving_values
from malr.models.explicit_repair import build_explicit_repair


def fixture():
    torch.manual_seed(31);values=torch.randn(2,96,3)
    mask=torch.rand_like(values)>.7;mask[1]=False
    validity=torch.ones_like(mask);validity[:,-2:]=False;mask &= validity
    visible=values.masked_fill(~mask,0)
    patches=patchify_preserving_values(visible,mask,16,8,validity=validity)
    explicit=build_explicit_repair(torch.randn_like(patches.values),patches.values,
        patches.effective_mask,patches.validity,torch.zeros(2,3),torch.ones(2,3),
        window_len=96,patch_len=16,stride=8)
    router=SimpleNamespace(latents=torch.randn(2,3,explicit.validity_patches.shape[2],64))
    return explicit,router,torch.randn(6,64,96),torch.zeros(2,3),torch.ones(2,3)


def test_zero_outputs_preserve_input_and_strict_state_roundtrip():
    model=RecentMeanEnsembleProjector(3,widths=(64,256),weights=(.5,.5))
    args=fixture();out,_=model(*args)
    assert torch.equal(out.normalized_sequence,args[0].normalized_sequence)
    assert model.weights.requires_grad is False
    clone=RecentMeanEnsembleProjector(3,widths=(64,256),weights=(.5,.5))
    clone.load_state_dict(model.state_dict(),strict=True)
    assert torch.equal(clone(*args)[0].normalized_sequence,out.normalized_sequence)


def test_nonzero_members_protect_observed_invalid_and_no_source_windows():
    model=RecentMeanEnsembleProjector(3,widths=(64,256),weights=(.5,.5))
    for member in model.members:
        with torch.no_grad():
            member.network[-1].bias.fill_(.2)
            member.global_dense_refiner.output.bias.fill_(.4)
            member.global_dense_refiner.guide_gate.bias.fill_(.3)
    args=fixture();explicit=args[0];out,_=model(*args)
    protected=explicit.sequence_observed | ~explicit.sequence_validity
    assert torch.equal(out.normalized_sequence[protected],explicit.normalized_sequence[protected])
    assert torch.equal(out.normalized_sequence[1],explicit.normalized_sequence[1])
    active=explicit.sequence_missing[0]
    assert not torch.equal(out.normalized_sequence[0][active],explicit.normalized_sequence[0][active])
    assert torch.equal(out.repaired_patches.masked_select(~out.validity_patches),torch.zeros_like(out.repaired_patches.masked_select(~out.validity_patches)))
    out.normalized_sequence.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize('widths,weights',[((64,),(.5,.5)),((64,256),(.5,-.5)),((64,256),(.5,.6)),((True,256),(.5,.5))])
def test_invalid_recipe_is_rejected(widths,weights):
    with pytest.raises(ValueError):RecentMeanEnsembleProjector(3,widths=widths,weights=weights)


def test_saved_weights_must_match_the_bound_recipe():
    from integrations.e23.calibration_ensemble import verify_ensemble_instance
    spec=dict(kind='shared_prefix_calibration_v1',member_widths=[64,256],weights=[.5,.5],patch_len=16,stride=8,no_source_policy='preserve_coarse')
    projector=RecentMeanEnsembleProjector(3,widths=(64,256),weights=(.5,.5))
    model=SimpleNamespace(recent_mean_projector=projector)
    verify_ensemble_instance(model,spec)
    projector.weights.copy_(torch.tensor([.6,.4]))
    with pytest.raises(ValueError,match='state differs'):verify_ensemble_instance(model,spec)
