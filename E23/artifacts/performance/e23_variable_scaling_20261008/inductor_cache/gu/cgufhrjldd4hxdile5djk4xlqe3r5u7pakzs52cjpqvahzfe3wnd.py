
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2097152, 'r0_': 32},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i1', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_cat_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 4, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax_cat_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1774080
    r0_numel = 27
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
    r0_3 = r0_index
    x0 = (xindex % 16)
    x2 = xindex // 128
    x5 = xindex
    tmp0 = r0_3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (tl.broadcast_to(4*x0 + 64*x2, [XBLOCK, R0_BLOCK])), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = 1e-08
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tl_math.log(tmp7)
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp4, tmp8, tmp9)
    tmp11 = tmp0 >= tmp3
    tmp12 = tl.full([1, 1], 27, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr1 + (26*x2 + ((-1) + r0_3)), xmask & r0_mask & tmp11, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp15 = tmp14 == 0
    tmp16 = tl.load(in_ptr2 + (26*x5 + ((-1) + r0_3)), xmask & r0_mask & tmp11, eviction_policy='evict_last', other=0.0)
    tmp17 = 0.35355339059327373
    tmp18 = tmp16 * tmp17
    tmp19 = tl.load(in_ptr3 + (26*x0 + 416*x2 + ((-1) + r0_3)), r0_mask & xmask & tmp11, eviction_policy='evict_last', other=0.0)
    tmp20 = 1e-08
    tmp21 = triton_helpers.maximum(tmp19, tmp20)
    tmp22 = tl_math.log(tmp21)
    tmp23 = tmp18 + tmp22
    tmp24 = float("-inf")
    tmp25 = tl.where(tmp15, tmp24, tmp23)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp11, tmp25, tmp26)
    tmp28 = tl.where(tmp4, tmp10, tmp27)
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, R0_BLOCK])
    tmp31 = tl.broadcast_to(tmp29, [XBLOCK, R0_BLOCK])
    tmp33 = tl.where(r0_mask & xmask, tmp31, float("-inf"))
    tmp34 = triton_helpers.max2(tmp33, 1)[:, None]
    tmp35 = tmp29 - tmp34
    tmp36 = tl_math.exp(tmp35)
    tmp37 = tl.broadcast_to(tmp36, [XBLOCK, R0_BLOCK])
    tmp39 = tl.where(r0_mask & xmask, tmp37, 0)
    tmp40 = tl.sum(tmp39, 1)[:, None]
    tmp41 = tmp28 - tmp34
    tmp42 = tl_math.exp(tmp41)
    tmp43 = (tmp42 / tmp40)
    tl.store(out_ptr2 + (r0_3 + 27*x5), tmp43, xmask & r0_mask)
