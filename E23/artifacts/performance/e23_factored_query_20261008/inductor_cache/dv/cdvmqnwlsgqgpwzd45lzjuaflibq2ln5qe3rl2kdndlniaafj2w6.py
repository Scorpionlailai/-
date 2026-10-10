
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i1', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_bitwise_not_gelu_gelu_backward_masked_fill_where_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_bitwise_not_gelu_gelu_backward_masked_fill_where_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33030144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x5 = xindex // 64
    x1 = ((xindex // 64) % 16)
    x2 = ((xindex // 1024) % 12)
    x6 = (xindex % 1024)
    x7 = xindex // 1024
    x0 = (xindex % 64)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x5), None, eviction_policy='evict_last').to(tl.int1)
    tmp8 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0 + 64*x7), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_out_ptr0 + (x4), None)
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0 == 0
    tmp2 = x1 + 8*x2
    tmp3 = tl.full([1], 96, tl.int64)
    tmp4 = tmp2 < tmp3
    tmp5 = tl.load(in_ptr1 + (x6 + 512*x7), tmp4, other=0.0)
    tmp6 = 0.0
    tmp7 = tl.where(tmp1, tmp6, tmp5)
    tmp9 = tl.where(tmp0, tmp7, tmp8)
    tmp11 = tmp9 + tmp10
    tmp14 = tmp12 + tmp13
    tmp15 = tmp11 + tmp14
    tmp16 = 0.5
    tmp17 = tmp15 * tmp16
    tmp18 = 0.7071067811865476
    tmp19 = tmp15 * tmp18
    tmp20 = libdevice.erf(tmp19)
    tmp21 = 1.0
    tmp22 = tmp20 + tmp21
    tmp23 = tmp17 * tmp22
    tmp24 = tmp22 * tmp16
    tmp25 = tmp15 * tmp15
    tmp26 = -0.5
    tmp27 = tmp25 * tmp26
    tmp28 = tl_math.exp(tmp27)
    tmp29 = 0.3989422804014327
    tmp30 = tmp28 * tmp29
    tmp31 = tmp15 * tmp30
    tmp32 = tmp24 + tmp31
    tl.store(out_ptr0 + (x4), tmp23, None)
    tl.store(out_ptr1 + (x4), tmp32, None)
