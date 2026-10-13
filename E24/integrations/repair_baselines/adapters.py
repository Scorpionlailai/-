"""Keep official models and losses; isolate necessary protocol substitutions."""
import ast
import copy
import importlib
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
import yaml

from .data import ROOT


class BritsAdapter(nn.Module):
    def __init__(self, data):
        super().__init__()
        from integrations.forecast_compare.author_profiles import _module, _OriginalBRITS
        root = ROOT / 'BRITS'
        unused = {'utils', 'argparse', 'data_loader', 'ipdb', 'sklearn'}
        class Compatibility(ast.NodeTransformer):
            def visit_Import(self, node):
                node.names = [a for a in node.names if a.name.split('.')[0] not in unused]
                return node if node.names else None
            def visit_ImportFrom(self, node):
                return None if node.module.split('.')[0] in unused else node
            def visit_Constant(self, node):
                return ast.copy_location(ast.Constant(data.n_vars), node) if node.value == 35 else node
            def visit_If(self, node):
                return None if ast.unparse(node.test) == 'torch.cuda.is_available()' else self.generic_visit(node)
        def compatible(path):
            tree = Compatibility().visit(ast.parse(path.read_text()))
            return ast.unparse(ast.fix_missing_locations(tree))
        path = root / 'models/rits.py'
        text = compatible(path).replace('SEQ_LEN = 48', 'SEQ_LEN = 96')
        text = text.replace('torch.zeros((values.size()[0], self.rnn_hid_size))',
                            'values.new_zeros((values.size()[0], self.rnn_hid_size))')
        _module('_repair_brits_rits', path, text=text)
        path = root / 'models/brits.py'
        text = compatible(path).replace('import rits', 'import _repair_brits_rits as rits')
        text = text.replace('torch.LongTensor(indices)', 'tensor_.new_tensor(list(indices), dtype=torch.long)')
        module = _module('_repair_brits', path, text=text)
        self.config = dict(hidden_size=108, impute_weight=.3, label_weight=0., seq_len=96, features=data.n_vars,
                           variant='BRITS-accelerated-v1', execution='scripted_recurrence')
        self.core = _OriginalBRITS(module.Model(108, .3, 0.))
        from .fast_brits import ScriptedCore
        self.core = ScriptedCore(self.core.model)

    def forward(self, values, observed, starts):
        result = self.core(values, mask=observed.to(values.dtype))
        return torch.where(observed, values, result)

    def loss(self, truth, observed, starts):
        return self.core.original_loss(torch.where(observed, truth, 0.), observed.to(truth.dtype))


class ImputeFormerAdapter(nn.Module):
    def __init__(self, data):
        super().__init__()
        sys.path.insert(0, str(ROOT / 'ImputeFormer/experiments'))
        from imputeformer.models.ImputeFormer import ImputeFormerModel
        cfg = yaml.safe_load((ROOT / 'ImputeFormer/config/imputation/imputeformer_la.yaml').read_text())
        keys = ('input_dim', 'output_dim', 'input_embedding_dim', 'learnable_embedding_dim',
                'feed_forward_dim', 'num_temporal_heads', 'num_layers', 'dim_proj', 'dropout')
        self.config = {k: cfg[k] for k in keys}
        self.config.update(num_nodes=data.n_vars, windows=96)
        self.core = ImputeFormerModel(**self.config)
        self.f1_loss_weight = cfg['f1_loss_weight']
        # Exactly the author's datetime_encoded(['day']): sin/cos of absolute
        # timestamp, two exogenous channels plus one value channel per node.
        phase = data.timestamps.astype('datetime64[ns]').astype(np.int64) * (2 * np.pi / 86400000000000)
        self.time_marks = np.stack([np.sin(phase), np.cos(phase)], axis=-1).astype(np.float32)

    def raw(self, values, observed, starts):
        u = torch.from_numpy(np.stack([self.time_marks[s:s+96] for s in starts])).to(values.device)
        return self.core(values.unsqueeze(-1), u, observed.unsqueeze(-1)).squeeze(-1)

    def forward(self, values, observed, starts):
        return torch.where(observed, values, self.raw(values, observed, starts))

    def loss(self, truth, observed, starts):
        output = self.raw(torch.where(observed, truth, 0.), observed, starts)
        loss = (output[~observed] - truth[~observed]).abs().mean()
        # Official train_shared_step uses original_mask (all source coordinates
        # are valid before dynamic whitening) for Freg. FFT includes batch axis.
        transformed = torch.fft.fftn(output.float().unsqueeze(-1))
        freg = (transformed.abs().flatten(1).sum(1) / transformed.numel()).mean()
        return loss + self.f1_loss_weight * freg


class CsdiAdapter(nn.Module):
    def __init__(self, data, device):
        super().__init__()
        sys.path.insert(0, str(ROOT / 'CSDI'))
        module = importlib.import_module('main_model')
        if Path(module.__file__).resolve() != ROOT / 'CSDI/main_model.py':
            raise RuntimeError('wrong CSDI source')
        self.config = yaml.safe_load((ROOT / 'CSDI/config/base.yaml').read_text())
        self.core = module.CSDI_base(data.n_vars, copy.deepcopy(self.config), device)
        self.n_samples = 5
        self.sample_chunk_size = 5
        self.config['sampling'] = dict(samples=5, training_samples=1, sample_chunk_size=5, aggregation='median',
                                      variant='CSDI', diffusion_steps=50)

    def side(self, truth, observed):
        mask = observed.permute(0, 2, 1).to(truth.dtype)
        times = torch.arange(96, device=truth.device, dtype=truth.dtype).expand(len(truth), -1)
        return mask, self.core.get_side_info(times, mask)

    def loss(self, truth, observed, starts):
        mask, side = self.side(truth, observed)
        return self.core.calc_loss(truth.permute(0, 2, 1), mask, torch.ones_like(mask), side, 1)

    def forward(self, values, observed, starts):
        mask, side = self.side(values, observed)
        from .fast_csdi import sample
        samples = sample(self.core, values.permute(0, 2, 1), mask, side, self.n_samples, self.sample_chunk_size)
        candidate = samples.median(dim=1).values.permute(0, 2, 1)
        return torch.where(observed, values, candidate)


def build_adapter(name, data, device):
    if name == 'TimesNet':
        from .timesnet import TimesNetAdapter
        return TimesNetAdapter(data).to(device)
    if name == 'BRITS':
        return BritsAdapter(data).to(device)
    if name == 'ImputeFormer':
        return ImputeFormerAdapter(data).to(device)
    if name == 'CSDI':
        return CsdiAdapter(data, device).to(device)
    raise ValueError(name)


def training_profile(name):
    common = dict(learning_rate=.001, weight_decay=0., precision='float32', max_batches=None,
                  patience=None, grad_clip=None, selection='final_epoch', seed=2023)
    if name == 'TimesNet':
        return dict(common, epochs=10, batch_size=16, patience=3, selection='mse', schedule='type1',
                    source='TimesNet dataset-specific imputation scripts + run.py defaults',
                    parameter_status='author_dataset_specific', loss='hidden_mse')
    if name == 'BRITS':
        return dict(common, epochs=200, batch_size=64, schedule='constant', patience=20, selection='mae',
                    source='BRITS/README.md example + main.py optimizer; no author validation checkpoint selection',
                    parameter_status='user_authorized_accelerated_base', loss='author_observed_l1_and_bidirectional_consistency',
                    variant='BRITS-accelerated-v1', changes_from_author=dict(epochs='1000->max200',
                        checkpoint='final_epoch->validation_mae', patience='none->20', execution='eager->scripted'))
    if name == 'ImputeFormer':
        return dict(common, epochs=300, batch_size=32, patience=20, selection='mae',
                    schedule='magic', precision='float16_mixed', max_batches=300, grad_clip=5.,
                    source='ImputeFormer/config/imputation/imputeformer_la.yaml + official training entry',
                    parameter_status='author_LA_base_not_target_dataset_optimum', loss='hidden_l1_plus_author_Freg_0.01')
    if name == 'CSDI':
        return dict(common, epochs=100, batch_size=16, weight_decay=1e-6, schedule='multistep', milestones=[75, 90],
                    source='CSDI/config/base.yaml + utils.py (saves final epoch, no best checkpoint saved)',
                    parameter_status='user_authorized_accelerated_base', loss='hidden_diffusion_noise_mse',
                    variant='CSDI', changes_from_author=dict(epochs='200->100', milestones='150,180->75,90', stage2_training_samples=1, inference_samples='100->5',
                        sampling='sequential->parallel_chunks5', architecture='unchanged', diffusion_steps='50 unchanged'))
    raise ValueError(name)
