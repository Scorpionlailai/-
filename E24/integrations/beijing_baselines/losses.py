"""Finite placeholders are never repair supervision."""


def hidden_error(candidate, truth, observed, valid):
    if not (candidate.shape == truth.shape == observed.shape == valid.shape):
        raise ValueError('repair supervision shapes differ')
    return (candidate - truth)[~observed.bool() & valid.bool()]
