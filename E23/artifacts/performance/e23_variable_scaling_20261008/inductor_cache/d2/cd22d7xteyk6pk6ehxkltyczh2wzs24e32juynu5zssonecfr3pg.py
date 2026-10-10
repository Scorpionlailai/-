
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 524288, 'r0_': 64},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_bitwise_not_masked_fill_native_layer_norm_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 4, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_bitwise_not_masked_fill_native_layer_norm_1(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 516096
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    x4 = xindex
    x0 = (xindex % 16)
    x1 = ((xindex // 16) % 12)
    r0_3 = r0_index
    x2 = xindex // 192
    tmp0 = tl.load(in_ptr0 + (x4), None, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tmp0 == 0
    tmp2 = x0 + 8*x1
    tmp3 = tl.full([1, 1], 96, tl.int64)
    tmp4 = tmp2 < tmp3
    tmp5 = tl.load(in_ptr1 + (x0 + 8*x1 + 96*r0_3 + 6144*x2), tmp4, other=0.0)
    tmp6 = 0.0
    tmp7 = tl.where(tmp1, tmp6, tmp5)
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp10 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp12 = tl.sum(tmp10, 1)[:, None]
    tmp13 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = (tmp12 / tmp14)
    tmp16 = tmp8 - tmp15
    tmp17 = tmp16 * tmp16
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
    tmp20 = tl.sum(tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp15, None)
    tl.store(out_ptr1 + (x4), tmp20, None)
