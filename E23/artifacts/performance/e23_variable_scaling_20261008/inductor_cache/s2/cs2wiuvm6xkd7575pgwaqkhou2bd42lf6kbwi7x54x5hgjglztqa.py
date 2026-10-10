
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 32},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*i1', 'in_ptr2': '*i1', 'in_ptr3': '*i1', 'in_ptr4': '*i1', 'in_ptr5': '*i1', 'in_ptr6': '*i1', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_any_cat_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 1, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_any_cat_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32256
    r0_numel = 26
    R0_BLOCK: tl.constexpr = 32
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x3 = xindex
    x0 = (xindex % 12)
    x1 = xindex // 12
    tmp0 = r0_2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (4*x3 + (r0_2)), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(x3, [XBLOCK, R0_BLOCK])), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp7 = tmp5 & tmp6
    tmp8 = tl.load(in_ptr2 + (tl.broadcast_to(1 + 4*x3, [XBLOCK, R0_BLOCK])), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp9 = tmp7 & tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1, 1], 16, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tmp12 & tmp14
    tmp16 = tl.load(in_ptr3 + (12*x3 + ((-4) + r0_2)), xmask & r0_mask & tmp15, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp17 = tl.load(in_ptr4 + (tl.broadcast_to(x3, [XBLOCK, R0_BLOCK])), xmask & r0_mask & tmp15, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp18 = tmp16 & tmp17
    tmp19 = tl.load(in_ptr2 + (tl.broadcast_to(2 + 4*x3, [XBLOCK, R0_BLOCK])), xmask & r0_mask & tmp15, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp20 = tmp18 & tmp19
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp15, tmp20, tmp21)
    tmp23 = tmp0 >= tmp13
    tmp24 = tl.full([1, 1], 26, tl.int64)
    tmp25 = tmp0 < tmp24
    tmp26 = tl.load(in_ptr5 + (x0 + 12*((-16) + r0_2) + 120*x1), xmask & r0_mask & tmp23, other=0.0).to(tl.int1)
    tmp27 = tl.load(in_ptr6 + (tl.broadcast_to(x3, [XBLOCK, R0_BLOCK])), xmask & r0_mask & tmp23, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp28 = tmp26 & tmp27
    tmp29 = tl.load(in_ptr2 + (tl.broadcast_to(3 + 4*x3, [XBLOCK, R0_BLOCK])), xmask & r0_mask & tmp23, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp30 = tmp28 & tmp29
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp23, tmp30, tmp31)
    tmp33 = tl.where(tmp15, tmp22, tmp32)
    tmp34 = tl.where(tmp4, tmp11, tmp33)
    tmp35 = tmp34.to(tl.int64)
    tmp36 = (tmp35 != 0)
    tmp37 = tl.broadcast_to(tmp36, [XBLOCK, R0_BLOCK])
    tmp39 = tl.where(r0_mask & xmask, tmp37, False)
    tmp40 = triton_helpers.any(tmp39, 1)[:, None]
    tl.store(out_ptr0 + (r0_2 + 26*x3), tmp34, xmask & r0_mask)
    tl.store(out_ptr1 + (x3), tmp40, xmask)
