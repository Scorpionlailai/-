
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sum_36', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_sum_36(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 172032
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 16
    tmp0 = tl.load(in_ptr0 + (26*x2), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (26*x2), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (4*x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (1 + 26*x2), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr1 + (1 + 26*x2), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr2 + (1 + 4*x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + (2 + 26*x2), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr1 + (2 + 26*x2), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr2 + (2 + 4*x1), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr0 + (3 + 26*x2), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr1 + (3 + 26*x2), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr2 + (3 + 4*x1), None, eviction_policy='evict_last')
    tmp1 = 1e-08
    tmp2 = tmp0 >= tmp1
    tmp4 = triton_helpers.maximum(tmp0, tmp1)
    tmp5 = (tmp3 / tmp4)
    tmp6 = 0.0
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp9 = tmp7 * tmp8
    tmp11 = tmp10 >= tmp1
    tmp13 = triton_helpers.maximum(tmp10, tmp1)
    tmp14 = (tmp12 / tmp13)
    tmp15 = tl.where(tmp11, tmp14, tmp6)
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 + tmp17
    tmp20 = tmp19 >= tmp1
    tmp22 = triton_helpers.maximum(tmp19, tmp1)
    tmp23 = (tmp21 / tmp22)
    tmp24 = tl.where(tmp20, tmp23, tmp6)
    tmp26 = tmp24 * tmp25
    tmp27 = tmp18 + tmp26
    tmp29 = tmp28 >= tmp1
    tmp31 = triton_helpers.maximum(tmp28, tmp1)
    tmp32 = (tmp30 / tmp31)
    tmp33 = tl.where(tmp29, tmp32, tmp6)
    tmp35 = tmp33 * tmp34
    tmp36 = tmp27 + tmp35
    tl.store(out_ptr0 + (x2), tmp36, None)
