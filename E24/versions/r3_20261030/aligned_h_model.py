"""H-only coordinate consistency. No diverse-key policy or new retrieval rules."""
import numpy as np
import torch
import bootstrap
import variants as original
from aligned_h_candidates import HistoricalCandidateProvider
from integrations.v125_ablation.model.historical import HistoricalExpert
from integrations.v125_ablation.adapter import V125Adapter
from integrations.physionet_baselines.losses import hidden_error,mean_or_zero

ORIGINAL_FACTORY=original.make_repair


class AlignedHistoricalExpert(HistoricalExpert):
    prepare_candidates=HistoricalCandidateProvider.prepare_candidates


class GlobalCoordinateAdapter(V125Adapter):
    def loss(self,truth,observed,starts):
        result,norm,positions=self.output(truth,observed,starts,training_coarse=True)
        valid=torch.from_numpy(self.data.valid_for(starts)).to(truth.device)
        candidate=result.explicit.standardized_sequence
        error=hidden_error(candidate,truth,observed,valid)
        hidden=~observed&valid
        tail=torch.where(hidden,candidate-truth,0.)[:,-32:]
        structure=tail.mean(1).square().mean()
        from malr.data.patching import patchify_with_state
        from malr.prototypes.retrieval import masked_pair_loss
        from integrations.physionet_baselines.losses import chunked_contrastive_loss
        # truth is already train-standardized. Hidden targets are supervised
        # labels only; query arithmetic is masked before entering the encoder.
        query=patchify_with_state(torch.where(observed&valid,truth,0.),observed,16,8)
        clean=patchify_with_state(torch.where(valid,truth,0.),valid,16,8)
        eligible=query.rho.lt(1.)&clean.effective_mask.any(-1)
        if eligible.any() and 'historical' not in self.disabled:
            encoder=self.core.historical.key_encoder
            q=encoder(query.values[eligible],query.effective_mask[eligible])
            t=encoder(clean.values[eligible],clean.effective_mask[eligible])
            retrieval=masked_pair_loss(q,t)+.5*chunked_contrastive_loss(q,t)/np.log(max(len(q),2))
        else:retrieval=error.sum()*0.
        from integrations.forecast_compare.protocol import P4Batch
        self._pending_batch=(P4Batch(norm.student,norm.target,observed,norm.mean,norm.std,
            truth.new_empty(len(truth),self.data.n_vars,0)),positions)
        return mean_or_zero(error.square())+.1*retrieval+structure


def make_repair(data,variant,seed=2023):
    if variant!='H_GLOBAL':raise ValueError(variant)
    model=ORIGINAL_FACTORY(data,'FULL',seed)
    model.__class__=GlobalCoordinateAdapter
    model.core.historical.__class__=AlignedHistoricalExpert
    model.core.historical.query_coordinates='train_global'
    # Keep legacy_v1 so original staging/rekey/EMA publication is unchanged.
    assert model.core.historical.key_policy=='legacy_v1'
    return model


def install():original.make_repair=make_repair
