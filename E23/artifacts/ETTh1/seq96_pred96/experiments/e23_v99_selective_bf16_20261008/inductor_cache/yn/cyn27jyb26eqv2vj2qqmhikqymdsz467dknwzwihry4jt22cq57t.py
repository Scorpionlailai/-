
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 262144, 'r0_': 128},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_cat_native_layer_norm_27', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 2, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_cat_native_layer_norm_27(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 172032
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x3 = xindex
    x0 = (xindex % 16)
    x1 = xindex // 16
    tmp13_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp13_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp13_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = r0_2
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 64, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (64*x3 + (r0_2)), r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tmp0 >= tmp3
        tmp7 = tl.full([1, 1], 128, tl.int64)
        tmp8 = tmp0 < tmp7
        tmp9 = tl.load(in_ptr1 + (8*x0 + 128*(((((-64) + r0_2) // 8) % 8)) + 1024*x1 + ((((-64) + r0_2) % 8))), r0_mask & tmp6, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tl.where(tmp4, tmp5, tmp9)
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
        tmp13_mean_next, tmp13_m2_next, tmp13_weight_next = triton_helpers.welford_reduce(
            tmp12, tmp13_mean, tmp13_m2, tmp13_weight, roffset == 0
        )
        tmp13_mean = tl.where(r0_mask, tmp13_mean_next, tmp13_mean)
        tmp13_m2 = tl.where(r0_mask, tmp13_m2_next, tmp13_m2)
        tmp13_weight = tl.where(r0_mask, tmp13_weight_next, tmp13_weight)
    tmp16, tmp17, tmp18 = triton_helpers.welford(tmp13_mean, tmp13_m2, tmp13_weight, 1)
    tmp13 = tmp16[:, None]
    tmp14 = tmp17[:, None]
    tmp15 = tmp18[:, None]
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp38 = tl.load(in_ptr2 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp40 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp19 = r0_2
        tmp20 = tl.full([1, 1], 0, tl.int64)
        tmp21 = tmp19 >= tmp20
        tmp22 = tl.full([1, 1], 64, tl.int64)
        tmp23 = tmp19 < tmp22
        tmp24 = tl.load(in_ptr0 + (64*x3 + (r0_2)), r0_mask & tmp23, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp25 = tmp19 >= tmp22
        tmp26 = tl.full([1, 1], 128, tl.int64)
        tmp27 = tmp19 < tmp26
        tmp28 = tl.load(in_ptr1 + (8*x0 + 128*(((((-64) + r0_2) // 8) % 8)) + 1024*x1 + ((((-64) + r0_2) % 8))), r0_mask & tmp25, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp29 = tl.where(tmp23, tmp24, tmp28)
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tmp30 - tmp13
        tmp32 = 128.0
        tmp33 = (tmp14 / tmp32)
        tmp34 = 1e-05
        tmp35 = tmp33 + tmp34
        tmp36 = libdevice.rsqrt(tmp35)
        tmp37 = tmp31 * tmp36
        tmp39 = tmp37 * tmp38
        tmp41 = tmp39 + tmp40
        tmp42 = tmp41.to(tl.float32)
        tl.store(out_ptr2 + (r0_2 + 128*x3), tmp42, r0_mask)
