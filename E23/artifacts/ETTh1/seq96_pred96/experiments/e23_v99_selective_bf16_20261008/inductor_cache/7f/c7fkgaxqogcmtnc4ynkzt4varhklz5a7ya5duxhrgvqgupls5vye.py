
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 65536, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_unfold_backward_59', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_unfold_backward_59(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 57344
    xnumel = 192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y1 = yindex // 64
    y0 = (yindex % 64)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (x2 + 192*y1), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp2 = tl.load(in_ptr1 + (x2 + 192*y1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (y0 + 144*x2 + 27648*y1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x2 + 192*y1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (8*(x2 // 16) + 104*y3 + ((x2 % 16))), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2 + 192*y1), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2 + 192*y1), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 == 0
    tmp3 = 0.015625
    tmp4 = tmp2 * tmp3
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp6 * tmp7
    tmp9 = 64.0
    tmp10 = tmp8 * tmp9
    tmp12 = tmp10 - tmp11
    tmp14 = 0.0
    tmp15 = tl.where(tmp1, tmp14, tmp13)
    tmp17 = tmp15 - tmp16
    tmp18 = tmp17 * tmp2
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 - tmp20
    tmp22 = tmp4 * tmp21
    tmp23 = tl.where(tmp1, tmp14, tmp22)
    tl.atomic_add(out_ptr0 + (8*(x2 // 16) + 104*y3 + ((x2 % 16))), tmp23, xmask, sem='relaxed')
