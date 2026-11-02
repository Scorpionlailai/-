"""Variant switches do not construct/delete modules or consume random numbers."""
DISABLED = {
    'A0': (), 'A1': ('historical','variable','temporal'),
    'A2': ('historical',), 'A3': ('variable',), 'A4': ('temporal',),
    'A5': (), 'A6': (), 'A7': (),
}


def configure(core, variant):
    disabled = frozenset(DISABLED[variant])
    core.iterative.neutral_quality = variant == 'A5'
    core.iterative.parallel = variant == 'A7'
    core.point_candidate_readout.neutral_quality = variant == 'A5'
    core.point_candidate_readout.patch_shared = variant == 'A6'
    for name in disabled:
        getattr(core, name).requires_grad_(False)
        core.router.residual[name].requires_grad_(False)
        core.router.scales[name].requires_grad_(False)
        core.point_router.quality[name].requires_grad_(False)
        core.point_candidate_readout.quality[name].requires_grad_(False)
    return disabled
