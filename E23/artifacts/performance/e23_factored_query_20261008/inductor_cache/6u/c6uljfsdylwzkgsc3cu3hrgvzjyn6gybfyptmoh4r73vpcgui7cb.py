
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_min_div_ge_masked_fill_slice_backward_where_40', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_min_div_ge_masked_fill_slice_backward_where_40(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2064384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 4)
    x1 = xindex // 4
    x2 = xindex
    tmp17 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr5 + (7 + x0 + 16*x1), None)
    tmp0 = x0
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (4*x1), tmp2, eviction_policy='evict_last', other=0.0)
    tmp4 = 1e-08
    tmp5 = tmp3 >= tmp4
    tmp6 = tl.load(in_ptr1 + (x1), tmp2, eviction_policy='evict_last', other=0.0)
    tmp7 = triton_helpers.maximum(tmp3, tmp4)
    tmp8 = (tmp6 / tmp7)
    tmp9 = 0.0
    tmp10 = tl.where(tmp5, tmp8, tmp9)
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp2, tmp10, tmp11)
    tmp13 = 0.0
    tmp14 = tl.where(tmp2, tmp12, tmp13)
    tmp15 = tl.full([1], 3, tl.int32)
    tmp16 = tmp0 == tmp15
    tmp18 = tl.where(tmp16, tmp17, tmp13)
    tmp19 = tmp14 + tmp18
    tmp20 = tl.full([1], 2, tl.int32)
    tmp21 = tmp0 == tmp20
    tmp23 = tl.where(tmp21, tmp22, tmp13)
    tmp24 = tmp19 + tmp23
    tmp25 = tl.full([1], 1, tl.int32)
    tmp26 = tmp0 == tmp25
    tmp28 = tl.where(tmp26, tmp27, tmp13)
    tmp29 = tmp24 + tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr0 + (x2), tmp31, None)
