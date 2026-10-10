"""Versioned author configurations; legacy adapters and upstream files stay intact."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import torch
from torch import nn

from .crib_p4 import _load_crib_class
from .saits_model import build_official_saits
from .t1_p4 import T1_SETTINGS, load_official_t1

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILES = ROOT / 'configs/baselines/etth1_p4_author_20261002.json'


def load_author_profiles(path: Path = DEFAULT_PROFILES) -> dict:
    profile = json.loads(Path(path).read_text())
    if profile.get('schema_version') != 1:
        raise ValueError('unsupported author profile schema')
    for name, cfg in profile['models'].items():
        train = cfg['training']
        if train['epochs'] < 1 or train['batch_size'] < 1 or train['learning_rate'] <= 0:
            raise ValueError(f'invalid training configuration: {name}')
        if train['selection_metric'] not in ('mse', 'mae'):
            raise ValueError(f'unsupported selection metric: {name}')
    return profile


def epoch_end_learning_rate(training: dict, completed_epoch: int) -> float:
    base = training['learning_rate']
    if training['lradj'] == 'constant':
        return base
    if training['lradj'] in ('ctf_type1', 's4m_type1'):
        return base * 0.5 ** (completed_epoch - 1)
    raise ValueError('unsupported author learning-rate schedule')


def author_coifnet_loss(prediction, values, mask, target, *, weight: float):
    history, future = prediction[:, :96], prediction[:, 96:]
    if history.shape != values.shape or future.shape != target.shape:
        raise ValueError('CoIFNet output shape mismatch')
    observed = ((history - values).abs() * mask).sum() / mask.sum().clamp_min(1)
    return weight * (future - target).abs().mean() + (1 - weight) * observed


def _module(name: str, source: Path, *, text: str | None = None) -> ModuleType:
    module = ModuleType(name)
    module.__file__ = str(source.resolve())
    sys.modules[name] = module
    exec(compile(source.read_text() if text is None else text, str(source), 'exec'), module.__dict__)
    return module


class _OriginalBRITS(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @staticmethod
    def _direction(values, mask):
        delta = torch.ones_like(values)
        for t in range(1, values.shape[1]):
            delta[:, t] = 1 + (1 - mask[:, t]) * delta[:, t - 1]
        return {'values': values, 'masks': mask, 'deltas': delta,
                'evals': torch.zeros_like(values), 'eval_masks': torch.zeros_like(mask)}

    def _result(self, values, mask):
        values = torch.where(mask.bool(), values, torch.zeros_like(values))
        return self.model({
            'forward': self._direction(values, mask),
            'backward': self._direction(values.flip(1), mask.flip(1)),
            'labels': values.new_zeros(values.shape[0]),
            'is_train': values.new_zeros(values.shape[0]),
        })

    def forward(self, values, *, mask):
        return self._result(values, mask)['imputations']

    def original_loss(self, values, mask):
        return self._result(values, mask)['loss']


def _original_brits(root: Path, cfg: dict) -> nn.Module:
    # Only compatibility/task constants change; original loss equations are retained.
    rits_path, brits_path = root / 'models/rits.py', root / 'models/brits.py'
    unused = {'utils', 'argparse', 'data_loader', 'ipdb', 'sklearn'}

    def remove_unused(source):
        tree = ast.parse(source)
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.Import) and any(a.name.split('.')[0] in unused for a in node.names)
            or isinstance(node, ast.ImportFrom) and node.module.split('.')[0] in unused)]
        class DeviceSafe(ast.NodeTransformer):
            def visit_If(self, node):
                if ast.unparse(node.test) == 'torch.cuda.is_available()':
                    return None  # Tensors are created on the input's device below.
                return self.generic_visit(node)
        tree = DeviceSafe().visit(tree)
        return ast.unparse(tree)

    text = remove_unused(rits_path.read_text())
    text = text.replace('SEQ_LEN = 48', 'SEQ_LEN = 96')
    # Every literal 35 in RITS is an input feature count.
    class Features(ast.NodeTransformer):
        def visit_Constant(self, node):
            return ast.copy_location(ast.Constant(7), node) if node.value == 35 else node
    text = ast.unparse(Features().visit(ast.parse(text)))
    text = text.replace('torch.zeros((values.size()[0], self.rnn_hid_size))',
                        'values.new_zeros((values.size()[0], self.rnn_hid_size))')
    _module('_paper_brits_rits', rits_path, text=text)
    text = remove_unused(brits_path.read_text()).replace('import rits', 'import _paper_brits_rits as rits')
    text = text.replace('torch.LongTensor(indices)', 'tensor_.new_tensor(list(indices), dtype=torch.long)')
    brits = _module('_paper_brits', brits_path, text=text)
    return _OriginalBRITS(brits.Model(cfg['hidden_size'], cfg['impute_weight'], cfg['label_weight']))


class _SaitsGlobal(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, values, *, mask):
        return self.model.impute({'X': values, 'missing_mask': mask})[0]

    def original_loss(self, values, mask, target):
        result = self.model({'X': values, 'missing_mask': mask, 'X_holdout': target,
                             'indicating_mask': 1 - mask}, 'train')
        return result['reconstruction_loss'] + result['imputation_loss']


def build_author_model(name: str, profile: dict, root: Path, device: torch.device) -> nn.Module:
    root = Path(root).resolve()
    cfg, train = profile['model'], profile['training']
    if name == 'CRIB':
        args = SimpleNamespace(**cfg, patch_num=12, pred_len=96, var_num=7,
                               output_attention=True, activation='relu',
                               batch_size=train['batch_size'], device=device)
        model = _load_crib_class(root)(args)
    elif name == 'CoIFNet':
        from .coifnet_p4 import _load_coifnet_class
        from omegaconf import OmegaConf
        official = OmegaConf.load(root / 'conf/model/CoIFNet.yaml')
        official.update(cfg)
        options = OmegaConf.create({'model': OmegaConf.to_container(official, resolve=True),
                                    'dataset': {'channels': 7}, 'device': str(device)})
        model = _load_coifnet_class(root)(options, seq_len=96, pred_len=192)
    elif name == 'BiTGraph':
        if device.type != 'cuda' or device.index not in (None, 0):
            raise ValueError('official BiTGraph requires CUDA device 0')
        directory = root / 'models/BiaTCGNet'
        layer_path = directory / 'BiaTCGNet_layer.py'
        text = layer_path.read_text()
        if text.count('torch.topk(A, k=5,') != 1:
            raise ValueError('BiTGraph nconv compatibility patch no longer matches upstream')
        _module('_paper_bitgraph_layer', layer_path,
                text=text.replace('torch.topk(A, k=5,', f"torch.topk(A, k={cfg['effective_topk']},"))
        main_path = directory / 'BiaTCGNet.py'
        module = _module('_paper_bitgraph', main_path, text=main_path.read_text().replace(
            'from models.BiaTCGNet.BiaTCGNet_layer import *', 'from _paper_bitgraph_layer import *'))
        model = module.Model(True, True, 2, 7, [2, 3, 6, 7], 'cuda:0', predefined_A=None,
                             dropout=cfg['dropout'], subgraph_size=cfg['effective_topk'],
                             node_dim=cfg['node_dim'], dilation_exponential=1,
                             conv_channels=cfg['conv_channels'], residual_channels=cfg['residual_channels'],
                             skip_channels=cfg['skip_channels'], end_channels=cfg['end_channels'],
                             seq_length=96, in_dim=1, out_len=96, out_dim=1, layers=cfg['layers'],
                             propalpha=0.05, tanhalpha=3, layer_norm_affline=True)
    elif name == 'ChannelTokenFormer':
        sys.path.insert(0, str(root))
        module = _module('_paper_ctf', root / 'models/ChannelTokenFormer_missing.py')
        args = SimpleNamespace(task_name='long_term_forecast', features='M', activation='gelu', seq_len=96,
                               pred_len=96, use_norm=1, patch_lens=[cfg['patch_len']] * 7,
                               sampling_periods=[1] * 7, enc_in=7,
                               batch_size=train['batch_size'], **{k: v for k, v in cfg.items() if k != 'patch_len'})
        model = module.Model(args)
    elif name == 'T1':
        if any(cfg[k] != T1_SETTINGS[k] for k in cfg):
            source = root / 'models/T1.py'
            module = _module('_paper_t1', source)
            model = module.Model(SimpleNamespace(**{**T1_SETTINGS, **cfg}))
        else:
            model = load_official_t1(root, device=device)
    elif name == 'SAITS':
        sys.path.insert(0, str(root))
        model = _SaitsGlobal(build_official_saits(root, device=device))
        # SAITS-base dimensions are pinned by the existing source-checked builder.
        from .saits_model import SAITS_SETTINGS
        if any(SAITS_SETTINGS[k] != v for k, v in cfg.items()):
            raise ValueError('this adapter supports the pinned public SAITS-base dimensions')
    elif name == 'BRITS':
        model = _original_brits(root, cfg)
    else:
        raise ValueError(f'{name} requires its dedicated model environment')
    return model.to(device)
