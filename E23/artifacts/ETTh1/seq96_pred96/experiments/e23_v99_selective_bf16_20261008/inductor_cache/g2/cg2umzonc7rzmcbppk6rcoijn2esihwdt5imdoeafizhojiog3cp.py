
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*i1', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24600576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 143)
    x4 = xindex // 143
    x1 = ((xindex // 143) % 16)
    x2 = ((xindex // 2288) % 12)
    x3 = xindex // 27456
    x5 = xindex // 2288
    x6 = (xindex % 2288)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4), tmp4, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp6 = tmp5 == 0
    tmp7 = x1 + 8*x2
    tmp8 = tl.full([1], 96, tl.int64)
    tmp9 = tmp7 < tmp8
    tmp10 = tmp9 & tmp4
    tmp11 = tl.load(in_ptr1 + (x1 + 8*x2 + 96*(x0) + 6144*x3), tmp10, eviction_policy='evict_last', other=0.0)
    tmp12 = 0.0
    tmp13 = tl.where(tmp6, tmp12, tmp11)
    tmp14 = tl.load(in_ptr2 + (x4), tmp4, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 - tmp14
    tmp16 = tl.load(in_ptr3 + (x4), tmp4, eviction_policy='evict_last', other=0.0)
    tmp17 = 64.0
    tmp18 = (tmp16 / tmp17)
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.rsqrt(tmp20)
    tmp22 = tmp15 * tmp21
    tmp23 = tl.load(in_ptr4 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 * tmp23
    tmp25 = tl.load(in_ptr5 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp26 = tmp24 + tmp25
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp4, tmp26, tmp27)
    tmp29 = tmp0 >= tmp3
    tmp30 = tl.full([1], 128, tl.int64)
    tmp31 = tmp0 < tmp30
    tmp32 = tmp29 & tmp31
    tmp33 = tl.load(in_ptr6 + (64*x5 + ((-64) + x0)), tmp32, eviction_policy='evict_last', other=0.0)
    tmp34 = tmp0 >= tmp30
    tmp35 = tl.full([1], 133, tl.int64)
    tmp36 = tmp0 < tmp35
    tmp37 = tmp34 & tmp36
    tmp38 = tl.load(in_ptr0 + (x4), tmp37, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp39 = tmp38 == 0
    tmp40 = x1 + 8*x2
    tmp41 = tl.full([1], 96, tl.int64)
    tmp42 = tmp40 < tmp41
    tmp43 = tmp42 & tmp37
    tmp44 = tl.load(in_ptr7 + (x1 + 8*x2 + 96*((-128) + x0) + 480*x3), tmp43, eviction_policy='evict_last', other=0.0)
    tmp45 = 0.0
    tmp46 = tl.where(tmp39, tmp45, tmp44)
    tmp47 = tl.full(tmp46.shape, 0.0, tmp46.dtype)
    tmp48 = tl.where(tmp37, tmp46, tmp47)
    tmp49 = tmp0 >= tmp35
    tmp50 = tl.full([1], 134, tl.int64)
    tmp51 = tmp0 < tmp50
    tmp52 = tmp49 & tmp51
    tmp53 = x1
    tmp54 = tmp53.to(tl.float32)
    tmp55 = 8.0
    tmp56 = tmp54 < tmp55
    tmp57 = 0.13333333333333333
    tmp58 = tmp54 * tmp57
    tmp59 = -1.0
    tmp60 = tmp58 + tmp59
    tmp61 = 15 + ((-1)*x1)
    tmp62 = tmp61.to(tl.float32)
    tmp63 = tmp62 * tmp57
    tmp64 = 1.0
    tmp65 = tmp64 - tmp63
    tmp66 = tl.where(tmp56, tmp60, tmp65)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp52, tmp66, tmp67)
    tmp69 = tmp0 >= tmp50
    tmp70 = tl.full([1], 135, tl.int64)
    tmp71 = tmp0 < tmp70
    tmp72 = tmp69 & tmp71
    tmp73 = tl.load(in_ptr8 + (x4), tmp72, eviction_policy='evict_last', other=0.0)
    tmp74 = tl.load(in_ptr9 + (x3), tmp72, eviction_policy='evict_last', other=0.0)
    tmp75 = tmp73 * tmp74
    tmp76 = tmp74 * tmp74
    tmp77 = 0.010000000000000002
    tmp78 = tmp76 + tmp77
    tmp79 = libdevice.sqrt(tmp78)
    tmp80 = (tmp75 / tmp79)
    tmp81 = tl.full(tmp80.shape, 0.0, tmp80.dtype)
    tmp82 = tl.where(tmp72, tmp80, tmp81)
    tmp83 = tmp0 >= tmp70
    tmp84 = tl.full([1], 139, tl.int64)
    tmp85 = tmp0 < tmp84
    tmp86 = tmp83 & tmp85
    tmp87 = tl.load(in_ptr10 + (4*x4 + ((-135) + x0)), tmp86, eviction_policy='evict_last', other=0.0)
    tmp88 = tmp0 >= tmp84
    tmp89 = tl.full([1], 143, tl.int64)
    tmp90 = tmp0 < tmp89
    tmp91 = tl.load(in_ptr11 + (4*x5 + ((-139) + x0)), tmp88, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp92 = tmp91.to(tl.float32)
    tmp93 = tl.full(tmp92.shape, 0.0, tmp92.dtype)
    tmp94 = tl.where(tmp88, tmp92, tmp93)
    tmp95 = tl.where(tmp86, tmp87, tmp94)
    tmp96 = tl.where(tmp72, tmp82, tmp95)
    tmp97 = tl.where(tmp52, tmp68, tmp96)
    tmp98 = tl.where(tmp37, tmp48, tmp97)
    tmp99 = tl.where(tmp32, tmp33, tmp98)
    tmp100 = tl.where(tmp4, tmp28, tmp99)
    tl.store(out_ptr0 + (x6 + 2304*x5), tmp100, None)
