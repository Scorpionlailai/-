
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*i1', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7741440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 15)
    x4 = xindex // 15
    x1 = ((xindex // 15) % 16)
    x2 = ((xindex // 240) % 12)
    x3 = xindex // 2880
    x5 = xindex // 240
    x6 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 5, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4), tmp4, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp6 = tmp5 == 0
    tmp7 = x1 + 8*x2
    tmp8 = tl.full([1], 96, tl.int64)
    tmp9 = tmp7 < tmp8
    tmp10 = tmp9 & tmp4
    tmp11 = tl.load(in_ptr1 + (x1 + 8*x2 + 96*(x0) + 480*x3), tmp10, eviction_policy='evict_last', other=0.0)
    tmp12 = 0.0
    tmp13 = tl.where(tmp6, tmp12, tmp11)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp4, tmp13, tmp14)
    tmp16 = tmp0 >= tmp3
    tmp17 = tl.full([1], 6, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = x1
    tmp21 = tmp20.to(tl.float32)
    tmp22 = 8.0
    tmp23 = tmp21 < tmp22
    tmp24 = 0.13333333333333333
    tmp25 = tmp21 * tmp24
    tmp26 = -1.0
    tmp27 = tmp25 + tmp26
    tmp28 = 15 + ((-1)*x1)
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp29 * tmp24
    tmp31 = 1.0
    tmp32 = tmp31 - tmp30
    tmp33 = tl.where(tmp23, tmp27, tmp32)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp19, tmp33, tmp34)
    tmp36 = tmp0 >= tmp17
    tmp37 = tl.full([1], 7, tl.int64)
    tmp38 = tmp0 < tmp37
    tmp39 = tmp36 & tmp38
    tmp40 = tl.load(in_ptr2 + (x4), tmp39, eviction_policy='evict_last', other=0.0)
    tmp41 = tl.load(in_ptr3 + (x3), tmp39, eviction_policy='evict_last', other=0.0)
    tmp42 = tmp40 * tmp41
    tmp43 = tmp41 * tmp41
    tmp44 = 0.010000000000000002
    tmp45 = tmp43 + tmp44
    tmp46 = libdevice.sqrt(tmp45)
    tmp47 = (tmp42 / tmp46)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp39, tmp47, tmp48)
    tmp50 = tmp0 >= tmp37
    tmp51 = tl.full([1], 11, tl.int64)
    tmp52 = tmp0 < tmp51
    tmp53 = tmp50 & tmp52
    tmp54 = tl.load(in_ptr4 + (4*x4 + ((-7) + x0)), tmp53, eviction_policy='evict_last', other=0.0)
    tmp55 = tmp0 >= tmp51
    tmp56 = tl.full([1], 15, tl.int64)
    tmp57 = tmp0 < tmp56
    tmp58 = tl.load(in_ptr5 + (4*x5 + ((-11) + x0)), tmp55, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp59 = tmp58.to(tl.float32)
    tmp60 = tl.full(tmp59.shape, 0.0, tmp59.dtype)
    tmp61 = tl.where(tmp55, tmp59, tmp60)
    tmp62 = tl.where(tmp53, tmp54, tmp61)
    tmp63 = tl.where(tmp39, tmp49, tmp62)
    tmp64 = tl.where(tmp19, tmp35, tmp63)
    tmp65 = tl.where(tmp4, tmp15, tmp64)
    tl.store(out_ptr0 + (x6), tmp65, None)
