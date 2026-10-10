"""Verification-only mapping; formal stack experiments always initialize from scratch."""
from .model import MultiExpertRepairEncoder


def copy_v29_r1_into_stack_reference(source:MultiExpertRepairEncoder,target:MultiExpertRepairEncoder)->None:
    if source.recovery_schedule!='iterative_tv_h_v1' or source.recovery_rounds!=1 or target.stacked is None or target.recovery_layers!=1 or target.expert_groups!=1:
        raise ValueError('reference mapping requires v29 r1 and a single-layer/single-group stack')
    target.base_encoder.load_state_dict(source.base_encoder.state_dict(),strict=True)
    target.decoder.load_state_dict(source.decoder.state_dict(),strict=True)
    target.historical.key_encoder.load_state_dict(source.historical.key_encoder.state_dict(),strict=True)
    target.historical.retrieval.load_state_dict(source.historical.retrieval.state_dict())
    target.historical.index=source.historical.index
    group=target.stacked.layers[0].groups[0]
    group.historical.load_state_dict({k:v for k,v in source.historical.state_dict().items() if not k.startswith('key_encoder.')},strict=True)
    for name in ('variable','temporal','router'):
        getattr(group,name).load_state_dict(getattr(source,name).state_dict(),strict=True)
    group.condition_embedding.load_state_dict(source.iterative.round_embedding.state_dict(),strict=True)
