
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*i1', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_cat_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_cat_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24600576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 143)
    x6 = xindex // 143
    x1 = ((xindex // 143) % 16)
    x2 = ((xindex // 2288) % 12)
    x3 = xindex // 27456
    x5 = xindex // 2288
    x4 = (xindex % 2288)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x6), tmp4, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp6 = tmp5 == 0
    tmp7 = tl.load(in_ptr1 + (x1 + 8*x2 + 104*(x0) + 6656*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp8 = 0.0
    tmp9 = tl.where(tmp6, tmp8, tmp7)
    tmp10 = tl.load(in_ptr2 + (x6), tmp4, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 - tmp10
    tmp12 = tl.load(in_ptr3 + (x6), tmp4, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 * tmp12
    tmp14 = tl.load(in_ptr4 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 * tmp14
    tmp16 = tl.load(in_ptr5 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp4, tmp17, tmp18)
    tmp20 = tmp0 >= tmp3
    tmp21 = tl.full([1], 128, tl.int64)
    tmp22 = tmp0 < tmp21
    tmp23 = tmp20 & tmp22
    tmp24 = tl.load(in_ptr6 + (64*x5 + ((-64) + x0)), tmp23, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp0 >= tmp21
    tmp26 = tl.full([1], 133, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tmp25 & tmp27
    tmp29 = tl.load(in_ptr0 + (x6), tmp28, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp30 = tmp29 == 0
    tmp31 = x1 + 8*x2
    tmp32 = tl.full([1], 96, tl.int64)
    tmp33 = tmp31 < tmp32
    tmp34 = tmp33 & tmp28
    tmp35 = tl.load(in_ptr7 + (x1 + 8*x2 + 96*((-128) + x0) + 480*x3), tmp34, eviction_policy='evict_last', other=0.0)
    tmp36 = 0.0
    tmp37 = tl.where(tmp30, tmp36, tmp35)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp28, tmp37, tmp38)
    tmp40 = tmp0 >= tmp26
    tmp41 = tl.full([1], 134, tl.int64)
    tmp42 = tmp0 < tmp41
    tmp43 = tmp40 & tmp42
    tmp44 = x1
    tmp45 = tmp44.to(tl.float32)
    tmp46 = 8.0
    tmp47 = tmp45 < tmp46
    tmp48 = 0.13333333333333333
    tmp49 = tmp45 * tmp48
    tmp50 = -1.0
    tmp51 = tmp49 + tmp50
    tmp52 = 15 + ((-1)*x1)
    tmp53 = tmp52.to(tl.float32)
    tmp54 = tmp53 * tmp48
    tmp55 = 1.0
    tmp56 = tmp55 - tmp54
    tmp57 = tl.where(tmp47, tmp51, tmp56)
    tmp58 = tl.full(tmp57.shape, 0.0, tmp57.dtype)
    tmp59 = tl.where(tmp43, tmp57, tmp58)
    tmp60 = tmp0 >= tmp41
    tmp61 = tl.full([1], 135, tl.int64)
    tmp62 = tmp0 < tmp61
    tmp63 = tmp60 & tmp62
    tmp64 = tl.load(in_ptr8 + (x6), tmp63, eviction_policy='evict_last', other=0.0)
    tmp65 = tl.load(in_ptr9 + (x3), tmp63, eviction_policy='evict_last', other=0.0)
    tmp66 = tmp64 * tmp65
    tmp67 = tmp65 * tmp65
    tmp68 = 0.010000000000000002
    tmp69 = tmp67 + tmp68
    tmp70 = libdevice.sqrt(tmp69)
    tmp71 = (tmp66 / tmp70)
    tmp72 = tl.full(tmp71.shape, 0.0, tmp71.dtype)
    tmp73 = tl.where(tmp63, tmp71, tmp72)
    tmp74 = tmp0 >= tmp61
    tmp75 = tl.full([1], 139, tl.int64)
    tmp76 = tmp0 < tmp75
    tmp77 = tmp74 & tmp76
    tmp78 = tl.load(in_ptr10 + (4*x6 + ((-135) + x0)), tmp77, eviction_policy='evict_last', other=0.0)
    tmp79 = tmp0 >= tmp75
    tmp80 = tl.full([1], 143, tl.int64)
    tmp81 = tmp0 < tmp80
    tmp82 = tl.load(in_ptr11 + (4*x5 + ((-139) + x0)), tmp79, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp83 = tmp82.to(tl.float32)
    tmp84 = tl.full(tmp83.shape, 0.0, tmp83.dtype)
    tmp85 = tl.where(tmp79, tmp83, tmp84)
    tmp86 = tl.where(tmp77, tmp78, tmp85)
    tmp87 = tl.where(tmp63, tmp73, tmp86)
    tmp88 = tl.where(tmp43, tmp59, tmp87)
    tmp89 = tl.where(tmp28, tmp39, tmp88)
    tmp90 = tl.where(tmp23, tmp24, tmp89)
    tmp91 = tl.where(tmp4, tmp19, tmp90)
    tmp92 = tmp91.to(tl.float32)
    tl.store(out_ptr1 + (x4 + 2304*x5), tmp92, None)
