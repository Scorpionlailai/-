
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i1', 'in_ptr1': '*i1', 'in_ptr2': '*i1', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_scalar_tensor_where_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 4, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_scalar_tensor_where_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 387072
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    x4 = xindex
    x5 = xindex // 12
    r0_3 = r0_index
    x2 = xindex // 3024
    x6 = (xindex % 3024)
    tmp0 = tl.load(in_ptr0 + (x4), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x5), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp3 = tl.load(in_ptr2 + (2 + 4*x5), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp5 = tl.load(in_ptr3 + (r0_3 + 64*x4), xmask, other=0.0)
    tmp31 = tl.load(in_ptr4 + (r0_3), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr5 + (r0_3), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_out_ptr0 + (r0_3 + 64*x4), xmask, other=0.0)
    tmp36 = tl.load(in_ptr6 + (r0_3), None, eviction_policy='evict_last')
    tmp2 = tmp0 & tmp1
    tmp4 = tmp2 & tmp3
    tmp6 = 0.0
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp10 = tl.where(xmask, tmp8, 0)
    tmp11 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp13 = tl.where(xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp15 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = (tmp14 / tmp16)
    tmp18 = tmp8 - tmp17
    tmp19 = tmp18 * tmp18
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
    tmp22 = tl.where(xmask, tmp20, 0)
    tmp23 = tl.sum(tmp22, 1)[:, None]
    tmp24 = tmp7 - tmp17
    tmp25 = 64.0
    tmp26 = (tmp23 / tmp25)
    tmp27 = 1e-05
    tmp28 = tmp26 + tmp27
    tmp29 = libdevice.rsqrt(tmp28)
    tmp30 = tmp24 * tmp29
    tmp32 = tmp30 * tmp31
    tmp34 = tmp32 + tmp33
    tmp37 = tmp35 + tmp36
    tmp38 = tmp34 + tmp37
    tl.store(in_out_ptr0 + (r0_3 + 64*x4), tmp38, xmask)
