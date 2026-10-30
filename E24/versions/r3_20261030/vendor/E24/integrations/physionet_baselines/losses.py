"""Finite placeholders are never repair supervision."""


def hidden_error(candidate, truth, observed, valid):
    if not (candidate.shape == truth.shape == observed.shape == valid.shape):
        raise ValueError('repair supervision shapes differ')
    return (candidate - truth)[~observed.bool() & valid.bool()]


def chunked_contrastive_loss(queries,clean_keys,*,temperature=1.,chunk_size=512):
    """Original full-negative cross entropy, with checkpointed query-row blocks."""
    import torch
    from torch.nn import functional as F
    from torch.utils.checkpoint import checkpoint
    if queries.shape!=clean_keys.shape or queries.ndim!=2:
        raise ValueError('queries and clean keys must have matching [batch,dim] shape')
    if temperature<=0 or chunk_size<=0:raise ValueError('positive temperature and chunk size required')
    n=len(queries)
    if not n:return queries.sum()*0.
    teacher=clean_keys.detach()
    total=queries.new_zeros(())
    for begin in range(0,n,chunk_size):
        rows=queries[begin:begin+chunk_size]
        def block(q,k,start=begin):
            logits=q@k.transpose(0,1)/temperature
            labels=torch.arange(start,start+len(q),device=q.device)
            return F.cross_entropy(logits,labels,reduction='sum')
        value=(checkpoint(block,rows,teacher,use_reentrant=False,preserve_rng_state=False)
               if torch.is_grad_enabled() and rows.requires_grad else block(rows,teacher))
        total=total+value
    return total/n


def mean_or_zero(errors):
    return errors.mean() if errors.numel() else errors.sum()
