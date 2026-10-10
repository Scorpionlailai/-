
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i1', 'in_ptr2': '*i1', 'in_ptr3': '*i1', 'in_ptr4': '*fp32', 'in_ptr5': '*i1', 'in_ptr6': '*i1', 'in_ptr7': '*fp32', 'in_ptr8': '*i1', 'in_ptr9': '*i1', 'in_ptr10': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 15, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4472832
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 26)
    x4 = xindex // 26
    x5 = xindex // 416
    x2 = ((xindex // 416) % 12)
    x3 = xindex // 4992
    x7 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (1 + 4*x4), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (4*x5 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp7 = tl.load(in_ptr2 + (x5), tmp4, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp8 = tmp6 & tmp7
    tmp9 = tl.load(in_ptr3 + (1 + 4*x5), tmp4, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp10 = tmp8 & tmp9
    tmp11 = tl.load(in_ptr4 + (4*x5 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp12 = 0.0
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = tmp5 * tmp13
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp4, tmp14, tmp15)
    tmp17 = tmp0 >= tmp3
    tmp18 = tl.full([1], 16, tl.int64)
    tmp19 = tmp0 < tmp18
    tmp20 = tmp17 & tmp19
    tmp21 = tl.load(in_ptr0 + (2 + 4*x4), tmp20, eviction_policy='evict_last', other=0.0)
    tmp22 = tl.load(in_ptr5 + (12*x5 + ((-4) + x0)), tmp20, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp23 = tl.load(in_ptr6 + (x5), tmp20, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp24 = tmp22 & tmp23
    tmp25 = tl.load(in_ptr3 + (2 + 4*x5), tmp20, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp26 = tmp24 & tmp25
    tmp27 = tl.load(in_ptr7 + (12*x5 + ((-4) + x0)), tmp20, eviction_policy='evict_last', other=0.0)
    tmp28 = 0.0
    tmp29 = tl.where(tmp26, tmp27, tmp28)
    tmp30 = tmp21 * tmp29
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp20, tmp30, tmp31)
    tmp33 = tmp0 >= tmp18
    tmp34 = tl.full([1], 26, tl.int64)
    tmp35 = tmp0 < tmp34
    tmp36 = tl.load(in_ptr0 + (3 + 4*x4), tmp33, eviction_policy='evict_last', other=0.0)
    tmp37 = tl.load(in_ptr8 + (x2 + 12*((-16) + x0) + 120*x3), tmp33, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp38 = tl.load(in_ptr9 + (x5), tmp33, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp39 = tmp37 & tmp38
    tmp40 = tl.load(in_ptr3 + (3 + 4*x5), tmp33, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp41 = tmp39 & tmp40
    tmp42 = tl.load(in_ptr10 + (10*x5 + ((-16) + x0)), tmp33, eviction_policy='evict_last', other=0.0)
    tmp43 = 0.0
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp36 * tmp44
    tmp46 = tl.full(tmp45.shape, 0.0, tmp45.dtype)
    tmp47 = tl.where(tmp33, tmp45, tmp46)
    tmp48 = tl.where(tmp20, tmp32, tmp47)
    tmp49 = tl.where(tmp4, tmp16, tmp48)
    tl.store(out_ptr0 + (x7), tmp49, None)
