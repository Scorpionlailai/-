"""Scriptable BRITS equations; retain original parameters and loss reductions."""
from typing import Tuple
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class FeatureRegression(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.W, self.b = original.W, original.b
        self.register_buffer('m', original.m)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.W * self.m, self.b)


class TemporalDecay(nn.Module):
    __constants__ = ['diag']

    def __init__(self, original):
        super().__init__()
        self.W, self.b, self.diag = original.W, original.b, original.diag
        if original.diag:
            self.register_buffer('m', original.m)
        else:
            self.m = torch.empty(0)

    def forward(self, delta: Tensor) -> Tensor:
        weight = self.W * self.m if self.diag else self.W
        return torch.exp(-F.relu(F.linear(delta, weight, self.b)))


class RITS(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.rnn_hid_size = original.rnn_hid_size
        self.impute_weight = original.impute_weight
        self.rnn_cell = original.rnn_cell
        self.temp_decay_h = TemporalDecay(original.temp_decay_h)
        self.temp_decay_x = TemporalDecay(original.temp_decay_x)
        self.hist_reg = original.hist_reg
        self.feat_reg = FeatureRegression(original.feat_reg)
        self.weight_combine = original.weight_combine
        self.dropout, self.out = original.dropout, original.out

    def forward(self, values: Tensor, masks: Tensor, deltas: Tensor) -> Tuple[Tensor, Tensor]:
        h = values.new_zeros((values.size(0), self.rnn_hid_size))
        c = torch.zeros_like(h)
        loss = values.new_zeros(())
        imputations = torch.jit.annotate(list[Tensor], [])
        for step in range(values.size(1)):
            x, mask, delta = values[:, step], masks[:, step], deltas[:, step]
            gamma_h, gamma_x = self.temp_decay_h(delta), self.temp_decay_x(delta)
            h = h * gamma_h
            history = self.hist_reg(h)
            loss = loss + torch.sum(torch.abs(x - history) * mask) / (torch.sum(mask) + 1e-5)
            completed_history = mask * x + (1 - mask) * history
            features = self.feat_reg(completed_history)
            loss = loss + torch.sum(torch.abs(x - features) * mask) / (torch.sum(mask) + 1e-5)
            alpha = self.weight_combine(torch.cat([gamma_x, mask], dim=1))
            combined = alpha * features + (1 - alpha) * history
            loss = loss + torch.sum(torch.abs(x - combined) * mask) / (torch.sum(mask) + 1e-5)
            completed = mask * x + (1 - mask) * combined
            h, c = self.rnn_cell(torch.cat([completed, mask], dim=1), (h, c))
            imputations.append(completed.unsqueeze(1))
        return loss * self.impute_weight, torch.cat(imputations, dim=1)


class BRITS(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.rits_f, self.rits_b = RITS(original.rits_f), RITS(original.rits_b)

    def forward(self, values: Tensor, mask: Tensor, forward_delta: Tensor, backward_delta: Tensor) -> Tuple[Tensor, Tensor]:
        loss_f, repaired_f = self.rits_f(values, mask, forward_delta)
        loss_b, repaired_b = self.rits_b(values.flip(1), mask.flip(1), backward_delta)
        repaired_b = repaired_b.flip(1)
        consistency = torch.abs(repaired_f - repaired_b).mean() * .1
        return loss_f + loss_b + consistency, (repaired_f + repaired_b) / 2


class ScriptedCore(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.model = torch.jit.script(BRITS(original))

    @staticmethod
    def deltas(mask):
        delta = torch.ones_like(mask)
        for step in range(1, mask.shape[1]):
            delta[:, step] = 1 + (1 - mask[:, step]) * delta[:, step-1]
        return delta

    def result(self, values, mask):
        values = torch.where(mask.bool(), values, 0.)
        return self.model(values, mask, self.deltas(mask), self.deltas(mask.flip(1)))

    def forward(self, values, *, mask):
        return self.result(values, mask)[1]

    def original_loss(self, values, mask):
        return self.result(values, mask)[0]
