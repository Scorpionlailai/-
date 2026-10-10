# AOT ID: ['0_backward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_factored_query_20261008/inductor_cache/4l/c4l4g7lxqr32bsj77xuyih7tmpgioqs5gy42bj4djtsgt7i2roiv.py
# Topologically Sorted Source Nodes: [visible, invert_6, and__6, eligible], Original ATen: [aten.masked_fill, aten.bitwise_not, aten.bitwise_and, aten.where, aten.sum]
# Source node to ATen node mapping:
#   and__6 => bitwise_and_6
#   eligible => bitwise_and_7
#   invert_6 => bitwise_not_6
#   visible => full_default
# Graph fragment:
#   %full_default : [num_users=14] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %bitwise_not_6 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_not.default](args = (%primals_2,), kwargs = {})
#   %bitwise_and_6 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%primals_3, %bitwise_not_6), kwargs = {})
#   %bitwise_and_7 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and_6, %unsqueeze_44), kwargs = {})
#   %where_18 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and_7, %tangents_1, %full_default), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_40, [0], True), kwargs = {})
triton_red_fused_bitwise_and_bitwise_not_masked_fill_sum_where_0 = async_compile.triton('triton_red_fused_bitwise_and_bitwise_not_masked_fill_sum_where_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 64, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*i1', 'in_ptr2': '*i1', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_bitwise_and_bitwise_not_masked_fill_sum_where_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_bitwise_and_bitwise_not_masked_fill_sum_where_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 63
    r0_numel = 8192
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index
        r0_2 = r0_index // 16
        tmp0 = tl.load(in_ptr0 + (r0_3 + 8192*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (r0_3 + 8192*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.int1)
        tmp4 = tl.load(in_ptr2 + (r0_2 + 512*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
        tmp6 = tl.load(in_ptr3 + (r0_3 + 8192*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp1 == 0
        tmp3 = tmp0 & tmp2
        tmp5 = tmp3 & tmp4
        tmp7 = 0.0
        tmp8 = tl.where(tmp5, tmp6, tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(r0_mask & xmask, tmp11, _tmp10)
        tl.store(out_ptr0 + (r0_3 + 8192*x0), tmp8, r0_mask & xmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp10, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_factored_query_20261008/inductor_cache/ix/cixv7snxxdeomexeskzpfkgbtwqigbe3q3kbzykpzr2aea5yhq4v.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_40, [0], True), kwargs = {})
triton_per_fused_sum_1 = async_compile.triton('triton_per_fused_sum_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_sum_1(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 63
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp4, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_factored_query_20261008/inductor_cache/w2/cw2l73rtgwijqzeuf6m3izqjuloupxq5t5ofvjivuouz6cz4rdy4.py
# Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.gelu, aten.gelu_backward]
# Source node to ATen node mapping:
#   input_3 => add_27, erf_1, mul_23
# Graph fragment:
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_37, 0.7071067811865476), kwargs = {})
#   %erf_1 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_23,), kwargs = {})
#   %add_27 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_1, 1), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_27, 0.5), kwargs = {})
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_37, %view_37), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_27, -0.5), kwargs = {})
#   %exp_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_28,), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp_1, 0.3989422804014327), kwargs = {})
#   %mul_30 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_37, %mul_29), kwargs = {})
#   %add_29 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %mul_30), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_42, %add_29), kwargs = {})
triton_poi_fused_gelu_gelu_backward_2 = async_compile.triton('triton_poi_fused_gelu_gelu_backward_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_gelu_backward_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33030144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = 0.7071067811865476
    tmp3 = tmp1 * tmp2
    tmp4 = libdevice.erf(tmp3)
    tmp5 = 1.0
    tmp6 = tmp4 + tmp5
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp9 = tmp1 * tmp1
    tmp10 = -0.5
    tmp11 = tmp9 * tmp10
    tmp12 = tl_math.exp(tmp11)
    tmp13 = 0.3989422804014327
    tmp14 = tmp12 * tmp13
    tmp15 = tmp1 * tmp14
    tmp16 = tmp8 + tmp15
    tmp17 = tmp0 * tmp16
    tl.store(in_out_ptr0 + (x0), tmp17, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_factored_query_20261008/inductor_cache/p3/cp3dqxk74lg4bi5cyeyti3ltqdhufxi5zhdl3lf6rijjipvaup2p.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_43, [0], True), kwargs = {})
triton_red_fused_sum_3 = async_compile.triton('triton_red_fused_sum_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 512},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_sum_3(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 86016
    r0_numel = 384
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 64)
    x1 = xindex // 64
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 64*r0_2 + 24576*x1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_factored_query_20261008/inductor_cache/2v/c2vsaick6rnwvxhqouso35xnonjrpygdhppdfqn4gsrsjj6qntvw.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_43, [0], True), kwargs = {})
triton_red_fused_sum_4 = async_compile.triton('triton_red_fused_sum_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 64, 'r0_': 2048},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_sum_4(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 64
    r0_numel = 1344
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 64*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_factored_query_20261008/inductor_cache/el/cel6fbssqmtdi6fhwjr2jqhpourgbly4tvpovnefllfwcuk5qi52.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.native_layer_norm_backward, aten.native_layer_norm]
# Source node to ATen node mapping:
#   input_1 => mul_20, sub_8
# Graph fragment:
#   %mul_33 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_45, %primals_52), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_33, 128), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_33, [4], True), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_7, %getitem_11), kwargs = {})
#   %mul_20 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %rsqrt_5), kwargs = {})
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_33, %mul_20), kwargs = {})
#   %sum_5 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_35, [4], True), kwargs = {})
#   %mul_36 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_20, %sum_5), kwargs = {})
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_34, %sum_4), kwargs = {})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_10, %mul_36), kwargs = {})
#   %div_4 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_5, 128), kwargs = {})
#   %mul_37 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_4, %sub_11), kwargs = {})
triton_per_fused_native_layer_norm_native_layer_norm_backward_5 = async_compile.triton('triton_per_fused_native_layer_norm_native_layer_norm_backward_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 524288, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_native_layer_norm_backward_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 516096
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 128*x0), None)
    tmp1 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (r0_1 + 128*x0), None)
    tmp7 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.sum(tmp3, 1)[:, None]
    tmp8 = tmp6 - tmp7
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
    tmp14 = tl.sum(tmp12, 1)[:, None]
    tmp15 = 0.0078125
    tmp16 = tmp9 * tmp15
    tmp17 = 128.0
    tmp18 = tmp2 * tmp17
    tmp19 = tmp18 - tmp5
    tmp20 = tmp10 * tmp14
    tmp21 = tmp19 - tmp20
    tmp22 = tmp16 * tmp21
    tl.store(out_ptr2 + (r0_1 + 128*x0), tmp22, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_factored_query_20261008/inductor_cache/x4/cx4efolpfs6cte4mmnksk4ti3zzxacbzcr2oa6qohnm6xdtprugg.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   input_1 => mul_20, sub_8
# Graph fragment:
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_7, %getitem_11), kwargs = {})
#   %mul_20 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %rsqrt_5), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_45, %mul_20), kwargs = {})
#   %sum_6 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_38, [0, 1, 2, 3]), kwargs = {})
#   %sum_7 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_45, [0, 1, 2, 3]), kwargs = {})
triton_red_fused_native_layer_norm_native_layer_norm_backward_6 = async_compile.triton('triton_red_fused_native_layer_norm_native_layer_norm_backward_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 262144, 'r0_': 512},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_native_layer_norm_backward_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_layer_norm_native_layer_norm_backward_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 172032
    r0_numel = 384
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 128)
    x1 = xindex // 128
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    _tmp11 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_2 + 49152*x1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + 128*r0_2 + 49152*x1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r0_2 + 384*x1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr3 + (r0_2 + 384*x1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp5 = tmp3 * tmp4
        tmp6 = tmp0 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask, tmp9, _tmp8)
        tmp10 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(r0_mask, tmp12, _tmp11)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, None)
    tl.store(out_ptr1 + (x3), tmp11, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_factored_query_20261008/inductor_cache/ke/ckedkxv7pkauwn67rk4wo7t2dg2b5xpjxc3ucunv2tozm3rrjcjy.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   input_1 => mul_20, sub_8
# Graph fragment:
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_7, %getitem_11), kwargs = {})
#   %mul_20 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %rsqrt_5), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_45, %mul_20), kwargs = {})
#   %sum_6 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_38, [0, 1, 2, 3]), kwargs = {})
triton_red_fused_native_layer_norm_native_layer_norm_backward_7 = async_compile.triton('triton_red_fused_native_layer_norm_native_layer_norm_backward_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r0_': 2048},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_native_layer_norm_backward_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_layer_norm_native_layer_norm_backward_7(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 1344
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_factored_query_20261008/inductor_cache/4k/c4kdlomgys2oqdin44dgru4ubwouot7ov6ltjajce742qq7nfjwh.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
# Source node to ATen node mapping:
# Graph fragment:
#   %clone_11 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_33,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_8 = async_compile.triton('triton_poi_fused_clone_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33030144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 8)
    x1 = ((xindex // 8) % 16)
    x2 = ((xindex // 128) % 8)
    x3 = xindex // 1024
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (64 + x0 + 8*x2 + 128*x1 + 2048*x3), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_factored_query_20261008/inductor_cache/um/cum5k2vwyz4fc6wozr4ry5pn6ndphtg7rn2wvgdplvlfcgueugrt.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.slice_backward, aten.add, aten._softmax_backward_data]
# Source node to ATen node mapping:
# Graph fragment:
#   %full_default_14 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([128, 21, 12, 8, 16, 27], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_scatter_default : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_14, %view_49, 5, 1, 9223372036854775807), kwargs = {})
#   %add_30 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tangents_2, %slice_scatter_default), kwargs = {})
#   %mul_39 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_30, %div_3), kwargs = {})
#   %sum_8 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_39, [-1], True), kwargs = {})
triton_per_fused__softmax_backward_data_add_slice_backward_9 = async_compile.triton('triton_per_fused__softmax_backward_data_add_slice_backward_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4194304, 'r0_': 32},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_add_slice_backward_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax_backward_data_add_slice_backward_9(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 4128768
    r0_numel = 27
    R0_BLOCK: tl.constexpr = 32
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 27*x0), r0_mask, other=0.0)
    tmp8 = tl.load(in_ptr2 + (r0_1 + 27*x0), r0_mask, other=0.0)
    tmp1 = r0_1
    tmp2 = tl.full([1, 1], 1, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.load(in_ptr1 + ((-1) + r0_1 + 26*x0), r0_mask & tmp3, other=0.0)
    tmp5 = 0.0
    tmp6 = tl.where(tmp3, tmp4, tmp5)
    tmp7 = tmp0 + tmp6
    tmp9 = tmp7 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp12 = tl.where(r0_mask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp13, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_factored_query_20261008/inductor_cache/l3/cl3i4mlsmgxgdcp5gwh3rmrs6c5tpc4eprfobeuovog4wn44mdi4.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_9 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%slice_24, [3], True), kwargs = {})
triton_per_fused_sum_10 = async_compile.triton('triton_per_fused_sum_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 524288, 'r0_': 8},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_sum_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 516096
    r0_numel = 8
    R0_BLOCK: tl.constexpr = 8
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 16)
    x1 = xindex // 16
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (27*x0 + 432*r0_2 + 3456*x1), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x0 + 16*r0_2 + 128*x1), None)
    tmp3 = tl.load(in_ptr2 + (27*x0 + 432*r0_2 + 3456*x1), None, eviction_policy='evict_last')
    tmp1 = -tmp0
    tmp4 = tl.full([1, 1], 0, tl.int64)
    tmp5 = tl.full([1, 1], 1, tl.int64)
    tmp6 = tmp4 >= tmp5
    tmp7 = tl.load(in_ptr3 + ((-1) + 26*x0 + 416*r0_2 + 3328*x1), tmp6, eviction_policy='evict_last', other=0.0)
    tmp8 = 0.0
    tmp9 = tl.where(tmp6, tmp7, tmp8)
    tmp10 = tmp3 + tmp9
    tmp11 = tmp10 * tmp0
    tmp12 = libdevice.fma(tmp1, tmp2, tmp11)
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp15 = tl.sum(tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_factored_query_20261008/inductor_cache/bd/cbdpzm3qfu6wkfwzmfyos5n24bduxbfsezyaczltisqghlzs4tzg.py
# Topologically Sorted Source Nodes: [visible, invert_5], Original ATen: [aten.masked_fill, aten.bitwise_not, aten.div]
# Source node to ATen node mapping:
#   invert_5 => bitwise_not_5
#   visible => full_default
# Graph fragment:
#   %full_default : [num_users=14] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %bitwise_not_5 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_not.default](args = (%unsqueeze_42,), kwargs = {})
#   %where_20 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_not_5, %full_default, %slice_25), kwargs = {})
#   %div_7 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%where_20, 2.8284271247461903), kwargs = {})
triton_poi_fused_bitwise_not_div_masked_fill_11 = async_compile.triton('triton_poi_fused_bitwise_not_div_masked_fill_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i1', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_not_div_masked_fill_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_bitwise_not_div_masked_fill_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 107347968
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 26)
    x2 = xindex // 3328
    x4 = xindex // 26
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 26*x2), None, eviction_policy='evict_last').to(tl.int1)
    tmp2 = tl.load(in_ptr1 + (1 + x0 + 27*x4), None)
    tmp4 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (1 + x0 + 27*x4), None)
    tmp1 = tmp0 == 0
    tmp3 = -tmp2
    tmp6 = 1 + x0
    tmp7 = tl.full([1], 1, tl.int64)
    tmp8 = tmp6 >= tmp7
    tmp9 = tl.load(in_out_ptr0 + (x3), tmp8, other=0.0)
    tmp10 = 0.0
    tmp11 = tl.where(tmp8, tmp9, tmp10)
    tmp12 = tmp5 + tmp11
    tmp13 = tmp12 * tmp2
    tmp14 = libdevice.fma(tmp3, tmp4, tmp13)
    tmp15 = tl.where(tmp1, tmp10, tmp14)
    tmp16 = 0.35355339059327373
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x3), tmp15, None)
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_factored_query_20261008/inductor_cache/hm/chmfmh4l4olkd2isresbgup3b6jrtotsa3d3vwbyklviznfiibwo.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_10 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%where_20, [3], True), kwargs = {})
triton_per_fused_sum_12 = async_compile.triton('triton_per_fused_sum_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16777216, 'r0_': 8},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_sum_12(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 13418496
    r0_numel = 8
    R0_BLOCK: tl.constexpr = 8
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 416)
    x1 = xindex // 416
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 416*r0_2 + 3328*x1), None)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.sum(tmp1, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp3, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_factored_query_20261008/inductor_cache/al/calwt2fdrpil65qlypd2ovzlpzpy6e3ikvanu4oy7a4feogygnfg.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.view]
# Source node to ATen node mapping:
# Graph fragment:
#   %view_54 : [num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%view_53, [838656, 64]), kwargs = {})
triton_poi_fused_view_13 = async_compile.triton('triton_poi_fused_view_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_view_13(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 53673984
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 64)
    x1 = xindex // 64
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (8*((x1 % 26)) + 208*(x0 // 8) + 1664*(x1 // 26) + ((x0 % 8))), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_factored_query_20261008/inductor_cache/h6/ch6cvqdvseva4u6zjrra4wvsvhxtjpsuditsdlkbxj246nup4de5.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_11 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_54, [0], True), kwargs = {})
triton_red_fused_sum_14 = async_compile.triton('triton_red_fused_sum_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 1024},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_sum_14(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 86016
    r0_numel = 624
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 64)
    x1 = xindex // 64
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 64*r0_2 + 39936*x1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_factored_query_20261008/inductor_cache/im/cimaljpgpp2hehwd4zkxklcfoyjhawjqd44hvnx3hgfzrffwdwij.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.clone, aten._unsafe_view]
# Source node to ATen node mapping:
# Graph fragment:
#   %clone_13 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%view_57,), kwargs = {memory_format: torch.contiguous_format})
#   %view_58 : [num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_13, [838656, 64]), kwargs = {})
triton_poi_fused__unsafe_view_clone_15 = async_compile.triton('triton_poi_fused__unsafe_view_clone_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1048576, 'x': 64}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_clone_15(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 838656
    xnumel = 64
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (26*x1 + 1664*(y0 // 26) + ((y0 % 26))), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x1 + 64*y0), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_factored_query_20261008/inductor_cache/yf/cyfnvazivn3ujo2xqg7zppxpapkaff2kebb6iokiwsrxt36yarnr.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_41 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_28, %unsqueeze_39), kwargs = {})
#   %sum_13 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_41, [4], True), kwargs = {})
triton_per_fused_mul_sum_16 = async_compile.triton('triton_per_fused_mul_sum_16', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 524288, 'r0_': 16},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sum_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mul_sum_16(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 516096
    r0_numel = 10
    R0_BLOCK: tl.constexpr = 16
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_4 = r0_index
    x5 = xindex
    x1 = ((xindex // 16) % 12)
    x2 = ((xindex // 192) % 21)
    x3 = xindex // 4032
    tmp0 = tl.load(in_ptr0 + (16 + r0_4 + 26*x5), r0_mask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (16 + r0_4 + 26*x5), r0_mask, other=0.0)
    tmp8 = tl.load(in_ptr2 + (x1 + 12*r0_4 + 120*x2 + 2528*x3), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp1 = 1e-08
    tmp2 = tmp0 >= tmp1
    tmp4 = triton_helpers.maximum(tmp0, tmp1)
    tmp5 = (tmp3 / tmp4)
    tmp6 = 0.0
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp9 = tmp7 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp12 = tl.where(r0_mask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x5), tmp13, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_factored_query_20261008/inductor_cache/7g/c7gdrbnwsrur27zsejfgifqhkdaxmrao67xcoqkgnfa42k2tueeh.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_40 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_28, %unsqueeze_38), kwargs = {})
#   %sum_14 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_40, [3], True), kwargs = {})
triton_per_fused_mul_sum_17 = async_compile.triton('triton_per_fused_mul_sum_17', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 524288, 'r0_': 16},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sum_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mul_sum_17(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 322560
    r0_numel = 16
    R0_BLOCK: tl.constexpr = 16
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
    r0_3 = r0_index
    x0 = (xindex % 10)
    x4 = xindex // 10
    x2 = xindex // 2520
    x5 = (xindex % 2520)
    tmp0 = tl.load(in_ptr0 + (16 + x0 + 26*r0_3 + 416*x4), xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (16 + x0 + 26*r0_3 + 416*x4), xmask, other=0.0)
    tmp8 = tl.load(in_ptr2 + (3 + 4*r0_3 + 64*x4), xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = 1e-08
    tmp2 = tmp0 >= tmp1
    tmp4 = triton_helpers.maximum(tmp0, tmp1)
    tmp5 = (tmp3 / tmp4)
    tmp6 = 0.0
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp9 = tmp7 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp12 = tl.where(xmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x5 + 2528*x2), tmp13, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_factored_query_20261008/inductor_cache/eg/cegfzqmxzidz6uhhjszdtehi3ypohn25a2xpjj7eszoeg7x7jej5.py
# Topologically Sorted Source Nodes: [visible, and__4, valid_2], Original ATen: [aten.masked_fill, aten.bitwise_and, aten.where]
# Source node to ATen node mapping:
#   and__4 => bitwise_and_4
#   valid_2 => bitwise_and_5
#   visible => full_default
# Graph fragment:
#   %full_default : [num_users=14] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %bitwise_and_4 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%primals_45, %unsqueeze_34), kwargs = {})
#   %bitwise_and_5 : [num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and_4, %unsqueeze_35), kwargs = {})
#   %where_22 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and_5, %squeeze_4, %full_default), kwargs = {})
triton_poi_fused_bitwise_and_masked_fill_where_18 = async_compile.triton('triton_poi_fused_bitwise_and_masked_fill_where_18', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32768, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*i1', 'in_ptr2': '*i1', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_and_masked_fill_where_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_bitwise_and_masked_fill_where_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 26880
    xnumel = 12
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y4 = yindex
    y5 = yindex // 10
    y0 = (yindex % 10)
    y1 = ((yindex // 10) % 21)
    y2 = yindex // 210
    tmp0 = tl.load(in_ptr0 + (x3 + 12*y4), ymask & xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x3 + 12*y5), ymask & xmask, eviction_policy='evict_last').to(tl.int1)
    tmp3 = tl.load(in_ptr2 + (3 + 4*x3 + 48*y5), ymask & xmask, eviction_policy='evict_last').to(tl.int1)
    tmp5 = tl.load(in_ptr3 + (y0 + 10*x3 + 120*y1 + 2528*y2), ymask & xmask, eviction_policy='evict_last')
    tmp2 = tmp0 & tmp1
    tmp4 = tmp2 & tmp3
    tmp6 = 0.0
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tl.store(out_ptr0 + (x3 + 12*y4), tmp7, ymask & xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_factored_query_20261008/inductor_cache/rd/crdoy4legoki6zryp5lkr7yla5rtulsdcedtzziokr4htrmxhwzw.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_15 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%slice_31, [0, 1, 2, 3], True), kwargs = {})
#   %sum_16 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%slice_31, [0, 1, 2, 3], True), kwargs = {})
#   %mul_48 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_31, %mul_42), kwargs = {})
#   %sum_19 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_48, [0, 1, 2, 3]), kwargs = {})
#   %sum_20 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%slice_31, [0, 1, 2, 3]), kwargs = {})
triton_red_fused_native_layer_norm_backward_sum_19 = async_compile.triton('triton_red_fused_native_layer_norm_backward_sum_19', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 256},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_sum_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 4, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_layer_norm_backward_sum_19(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 86016
    r0_numel = 240
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 64)
    x1 = xindex // 64
    _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    _tmp9 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (1024 + x0 + 64*((r0_2 % 10)) + 1664*(r0_2 // 10) + 39936*x1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (1024 + x0 + 64*((r0_2 % 10)) + 1664*(r0_2 // 10) + 39936*x1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x0 + 64*(((r0_2 // 10) % 12)) + 768*((r0_2 % 10)) + 7680*(r0_2 // 120) + 15360*x1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(r0_mask, tmp5, _tmp4)
        tmp7 = tmp2 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(r0_mask, tmp10, _tmp9)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, None)
    tl.store(out_ptr1 + (x3), tmp4, None)
    tl.store(out_ptr2 + (x3), tmp9, None)
    tl.store(out_ptr3 + (x3), tmp4, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_factored_query_20261008/inductor_cache/ox/coxh7e6chge5eghhelomjd7syy3v2taf2qrcesw5xobmnbq2dwlm.py
# Topologically Sorted Source Nodes: [visible], Original ATen: [aten.masked_fill, aten.clone, aten.native_layer_norm_backward, aten.where]
# Source node to ATen node mapping:
#   visible => full_default
# Graph fragment:
#   %full_default : [num_users=14] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %clone_15 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_31,), kwargs = {memory_format: torch.contiguous_format})
#   %mul_43 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_31, %primals_29), kwargs = {})
#   %mul_44 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_43, 64), kwargs = {})
#   %sum_17 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_43, [4], True), kwargs = {})
#   %mul_45 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_43, %mul_42), kwargs = {})
#   %sum_18 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_45, [4], True), kwargs = {})
#   %mul_46 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_42, %sum_18), kwargs = {})
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_44, %sum_17), kwargs = {})
#   %sub_14 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_13, %mul_46), kwargs = {})
#   %mul_47 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_8, %sub_14), kwargs = {})
#   %where_23 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_36, %mul_47, %full_default), kwargs = {})
triton_per_fused_clone_masked_fill_native_layer_norm_backward_where_20 = async_compile.triton('triton_per_fused_clone_masked_fill_native_layer_norm_backward_where_20', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*i1', 'in_ptr4': '*i1', 'in_ptr5': '*i1', 'in_ptr6': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_clone_masked_fill_native_layer_norm_backward_where_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 2, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_clone_masked_fill_native_layer_norm_backward_where_20(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 322560
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
    r0_3 = r0_index
    x0 = (xindex % 10)
    x6 = xindex // 10
    x2 = xindex // 2520
    x7 = (xindex % 2520)
    x4 = ((xindex // 10) % 12)
    x8 = xindex // 120
    x9 = xindex
    tmp0 = tl.load(in_ptr0 + (1024 + r0_3 + 64*x0 + 1664*x6), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (1024 + r0_3 + 64*x0 + 1664*x6), xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r0_3), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (r0_3 + 64*x4 + 768*x0 + 7680*x8), xmask, other=0.0)
    tmp15 = tl.load(in_ptr3 + (x4 + 12*x0 + 120*x8), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp16 = tl.load(in_ptr4 + (x6), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp18 = tl.load(in_ptr5 + (3 + 4*x6), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp20 = tl.load(in_ptr6 + (x7 + 2528*x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None]
    tmp10 = tmp4 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp13 = tl.where(xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp17 = tmp15 & tmp16
    tmp19 = tmp17 & tmp18
    tmp21 = 64.0
    tmp22 = tmp4 * tmp21
    tmp23 = tmp22 - tmp8
    tmp24 = tmp9 * tmp14
    tmp25 = tmp23 - tmp24
    tmp26 = tmp20 * tmp25
    tmp27 = 0.0
    tmp28 = tl.where(tmp19, tmp26, tmp27)
    tl.store(out_ptr2 + (r0_3 + 64*x9), tmp2, xmask)
    tl.store(in_out_ptr0 + (r0_3 + 64*x4 + 768*x0 + 7680*x8), tmp28, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_factored_query_20261008/inductor_cache/b6/cb6452mz7zlr5h57aiq625l3h2p6m77m7oozkerxz7xfilgpxxfk.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_50 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_27, %unsqueeze_33), kwargs = {})
#   %sum_21 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_50, [4], True), kwargs = {})
triton_per_fused_mul_sum_21 = async_compile.triton('triton_per_fused_mul_sum_21', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 524288, 'r0_': 16},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sum_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mul_sum_21(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 516096
    r0_numel = 12
    R0_BLOCK: tl.constexpr = 16
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_3 = r0_index
    x4 = xindex
    x1 = ((xindex // 16) % 252)
    x2 = xindex // 4032
    tmp0 = tl.load(in_ptr0 + (4 + r0_3 + 26*x4), r0_mask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (4 + r0_3 + 26*x4), r0_mask, other=0.0)
    tmp8 = tl.load(in_ptr2 + (r0_3 + 12*x1 + 3040*x2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp1 = 1e-08
    tmp2 = tmp0 >= tmp1
    tmp4 = triton_helpers.maximum(tmp0, tmp1)
    tmp5 = (tmp3 / tmp4)
    tmp6 = 0.0
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp9 = tmp7 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp12 = tl.where(r0_mask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp13, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_factored_query_20261008/inductor_cache/cw/ccwkj7gtusb6obwukiobeficjqauaaqucxmnbicdxrfg6shs3hn6.py
# Topologically Sorted Source Nodes: [visible, and__2, valid_1], Original ATen: [aten.masked_fill, aten.mul, aten.sum, aten.bitwise_and, aten.where]
# Source node to ATen node mapping:
#   and__2 => bitwise_and_2
#   valid_1 => bitwise_and_3
#   visible => full_default
# Graph fragment:
#   %full_default : [num_users=14] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_49 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_27, %unsqueeze_32), kwargs = {})
#   %sum_22 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_49, [3], True), kwargs = {})
#   %bitwise_and_2 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%primals_38, %unsqueeze_28), kwargs = {})
#   %bitwise_and_3 : [num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and_2, %unsqueeze_29), kwargs = {})
#   %where_24 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and_3, %squeeze_6, %full_default), kwargs = {})
triton_per_fused_bitwise_and_masked_fill_mul_sum_where_22 = async_compile.triton('triton_per_fused_bitwise_and_masked_fill_mul_sum_where_22', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 524288, 'r0_': 16},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*i1', 'in_ptr4': '*i1', 'in_ptr5': '*i1', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_bitwise_and_masked_fill_mul_sum_where_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 1, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_bitwise_and_masked_fill_mul_sum_where_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 387072
    r0_numel = 16
    R0_BLOCK: tl.constexpr = 16
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
    r0_3 = r0_index
    x0 = (xindex % 12)
    x4 = xindex // 12
    x2 = xindex // 3024
    x5 = (xindex % 3024)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (4 + x0 + 26*r0_3 + 416*x4), xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (4 + x0 + 26*r0_3 + 416*x4), xmask, other=0.0)
    tmp8 = tl.load(in_ptr2 + (2 + 4*r0_3 + 64*x4), xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.load(in_ptr3 + (x6), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp15 = tl.load(in_ptr4 + (x4), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp17 = tl.load(in_ptr5 + (2 + 4*x4), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = 1e-08
    tmp2 = tmp0 >= tmp1
    tmp4 = triton_helpers.maximum(tmp0, tmp1)
    tmp5 = (tmp3 / tmp4)
    tmp6 = 0.0
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp9 = tmp7 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp12 = tl.where(xmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tmp16 = tmp14 & tmp15
    tmp18 = tmp16 & tmp17
    tmp19 = tl.where(tmp18, tmp13, tmp6)
    tl.store(out_ptr1 + (x6), tmp19, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_factored_query_20261008/inductor_cache/z2/cz2ogislyvjn57so5v3mckcd6qeaau3prsstb2cxhk5vsdwsqwul.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_23 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%slice_30, [0, 1, 2, 3], True), kwargs = {})
#   %mul_57 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_30, %mul_14), kwargs = {})
#   %sum_27 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_57, [0, 1, 2, 3]), kwargs = {})
#   %sum_28 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%slice_30, [0, 1, 2, 3]), kwargs = {})
triton_red_fused_native_layer_norm_backward_sum_23 = async_compile.triton('triton_red_fused_native_layer_norm_backward_sum_23', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 512},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_sum_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_layer_norm_backward_sum_23(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 86016
    r0_numel = 288
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 64)
    x1 = xindex // 64
    _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    _tmp9 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (256 + x0 + 64*((r0_2 % 12)) + 1664*(r0_2 // 12) + 39936*x1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (256 + x0 + 64*((r0_2 % 12)) + 1664*(r0_2 // 12) + 39936*x1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x0 + 64*r0_2 + 18432*x1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(r0_mask, tmp5, _tmp4)
        tmp7 = tmp2 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(r0_mask, tmp10, _tmp9)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, None)
    tl.store(out_ptr1 + (x3), tmp9, None)
    tl.store(out_ptr2 + (x3), tmp4, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_factored_query_20261008/inductor_cache/lg/clgziv6czvye5aqppdo32vj2wmzi5qv3buc2j5cifons4keddhqp.py
# Topologically Sorted Source Nodes: [visible], Original ATen: [aten.masked_fill, aten.clone, aten.native_layer_norm_backward, aten.where]
# Source node to ATen node mapping:
#   visible => full_default
# Graph fragment:
#   %full_default : [num_users=14] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %clone_16 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_30,), kwargs = {memory_format: torch.contiguous_format})
#   %mul_52 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_30, %primals_29), kwargs = {})
#   %mul_53 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_52, 64), kwargs = {})
#   %sum_25 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_52, [4], True), kwargs = {})
#   %mul_54 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_52, %mul_14), kwargs = {})
#   %sum_26 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_54, [4], True), kwargs = {})
#   %mul_55 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_14, %sum_26), kwargs = {})
#   %sub_16 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_53, %sum_25), kwargs = {})
#   %sub_17 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_16, %mul_55), kwargs = {})
#   %mul_56 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_9, %sub_17), kwargs = {})
#   %where_26 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_30, %mul_56, %full_default), kwargs = {})
triton_per_fused_clone_masked_fill_native_layer_norm_backward_where_24 = async_compile.triton('triton_per_fused_clone_masked_fill_native_layer_norm_backward_where_24', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*i1', 'in_ptr4': '*i1', 'in_ptr5': '*i1', 'in_ptr6': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_clone_masked_fill_native_layer_norm_backward_where_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 2, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_clone_masked_fill_native_layer_norm_backward_where_24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    r0_3 = r0_index
    x0 = (xindex % 12)
    x4 = xindex // 12
    x2 = xindex // 3024
    x5 = (xindex % 3024)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (256 + r0_3 + 64*x0 + 1664*x4), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (256 + r0_3 + 64*x0 + 1664*x4), xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r0_3), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (r0_3 + 64*x6), xmask, other=0.0)
    tmp15 = tl.load(in_ptr3 + (x6), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp16 = tl.load(in_ptr4 + (x4), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp18 = tl.load(in_ptr5 + (2 + 4*x4), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp20 = tl.load(in_ptr6 + (x5 + 3040*x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None]
    tmp10 = tmp4 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp13 = tl.where(xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp17 = tmp15 & tmp16
    tmp19 = tmp17 & tmp18
    tmp21 = 64.0
    tmp22 = tmp4 * tmp21
    tmp23 = tmp22 - tmp8
    tmp24 = tmp9 * tmp14
    tmp25 = tmp23 - tmp24
    tmp26 = tmp20 * tmp25
    tmp27 = 0.0
    tmp28 = tl.where(tmp19, tmp26, tmp27)
    tl.store(out_ptr2 + (r0_3 + 64*x6), tmp2, xmask)
    tl.store(in_out_ptr0 + (r0_3 + 64*x6), tmp28, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_factored_query_20261008/inductor_cache/us/cus3qdzfdrygowfi2yjbigzsfbr2aqgbgp62lbjpcpfemzm7rj6k.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_24 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_66, [0], True), kwargs = {})
triton_red_fused_sum_25 = async_compile.triton('triton_red_fused_sum_25', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 512},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_25', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_sum_25(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 86016
    r0_numel = 288
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 64)
    x1 = xindex // 64
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 64*r0_2 + 18432*x1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_factored_query_20261008/inductor_cache/rr/crro5sx73mm5buw72kpceqorpzebqrpqhmoaqa5yoljcagd6jykp.py
# Topologically Sorted Source Nodes: [visible], Original ATen: [aten.masked_fill, aten.where]
# Source node to ATen node mapping:
#   visible => full_default
# Graph fragment:
#   %full_default : [num_users=14] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_25 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_30, %view_68, %full_default), kwargs = {})
triton_poi_fused_masked_fill_where_26 = async_compile.triton('triton_poi_fused_masked_fill_where_26', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i1', 'in_ptr1': '*i1', 'in_ptr2': '*i1', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_masked_fill_where_26', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_masked_fill_where_26(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1548288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex // 4
    x2 = xindex // 48
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last').to(tl.int1)
    tmp3 = tl.load(in_ptr2 + (2 + 4*x2), None, eviction_policy='evict_last').to(tl.int1)
    tmp5 = tl.load(in_out_ptr0 + (x5), None)
    tmp2 = tmp0 & tmp1
    tmp4 = tmp2 & tmp3
    tmp6 = 0.0
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tl.store(in_out_ptr0 + (x5), tmp7, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_factored_query_20261008/inductor_cache/zs/czs73vtltuteph444kwumwhg4bsnw6lxnnsjc45wprqu5tbzir4t.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_59 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_26, %unsqueeze_27), kwargs = {})
#   %sum_29 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_59, [4], True), kwargs = {})
triton_poi_fused_mul_sum_27 = async_compile.triton('triton_poi_fused_mul_sum_27', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sum_27', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_sum_27(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 516096
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
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_factored_query_20261008/inductor_cache/xw/cxw46h5b5y2r3u6ffgwgkv3ajqo34szjzcssvgp4ddid34jjddyz.py
# Topologically Sorted Source Nodes: [visible, and_, valid], Original ATen: [aten.masked_fill, aten.mul, aten.sum, aten.bitwise_and, aten.where]
# Source node to ATen node mapping:
#   and_ => bitwise_and
#   valid => bitwise_and_1
#   visible => full_default
# Graph fragment:
#   %full_default : [num_users=14] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_58 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_26, %unsqueeze_26), kwargs = {})
#   %sum_30 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_58, [3], True), kwargs = {})
#   %bitwise_and : [num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%primals_28, %unsqueeze_22), kwargs = {})
#   %bitwise_and_1 : [num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and, %unsqueeze_23), kwargs = {})
#   %where_27 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and_1, %squeeze_8, %full_default), kwargs = {})
triton_per_fused_bitwise_and_masked_fill_mul_sum_where_28 = async_compile.triton('triton_per_fused_bitwise_and_masked_fill_mul_sum_where_28', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 131072, 'r0_': 16},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*i1', 'in_ptr4': '*i1', 'in_ptr5': '*i1', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_bitwise_and_masked_fill_mul_sum_where_28', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 1, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_bitwise_and_masked_fill_mul_sum_where_28(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 129024
    r0_numel = 16
    R0_BLOCK: tl.constexpr = 16
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
    r0_2 = r0_index
    x0 = (xindex % 4)
    x1 = xindex // 4
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 26*r0_2 + 416*x1), xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (x0 + 26*r0_2 + 416*x1), xmask, other=0.0)
    tmp8 = tl.load(in_ptr2 + (1 + 4*r0_2 + 64*x1), xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp15 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp17 = tl.load(in_ptr5 + (1 + 4*x1), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = 1e-08
    tmp2 = tmp0 >= tmp1
    tmp4 = triton_helpers.maximum(tmp0, tmp1)
    tmp5 = (tmp3 / tmp4)
    tmp6 = 0.0
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp9 = tmp7 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp12 = tl.where(xmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tmp16 = tmp14 & tmp15
    tmp18 = tmp16 & tmp17
    tmp19 = tl.where(tmp18, tmp13, tmp6)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp19, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_factored_query_20261008/inductor_cache/zi/czicnt3ecbn37nhhovo7hpemqtlnt37mcuzlqs3bwhavjon2w2xn.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_31 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%slice_29, [0, 1, 2, 3], True), kwargs = {})
#   %mul_66 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_29, %mul_11), kwargs = {})
#   %sum_35 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_66, [0, 1, 2, 3]), kwargs = {})
#   %sum_36 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%slice_29, [0, 1, 2, 3]), kwargs = {})
triton_red_fused_native_layer_norm_backward_sum_29 = async_compile.triton('triton_red_fused_native_layer_norm_backward_sum_29', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 65536, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_sum_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_layer_norm_backward_sum_29(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 64512
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 64)
    x1 = xindex // 64
    _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    _tmp9 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 64*((r0_2 % 4)) + 1664*(r0_2 // 4) + 53248*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + 64*((r0_2 % 4)) + 1664*(r0_2 // 4) + 53248*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x0 + 64*r0_2 + 8192*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
        tmp7 = tmp2 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(r0_mask & xmask, tmp10, _tmp9)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tl.store(out_ptr1 + (x3), tmp9, xmask)
    tl.store(out_ptr2 + (x3), tmp4, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_factored_query_20261008/inductor_cache/x3/cx36j6yt7lhceyzotig5ooevflcqvovgrmqgwsyvvljpqfshaliv.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_31 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%slice_29, [0, 1, 2, 3], True), kwargs = {})
triton_red_fused_sum_30 = async_compile.triton('triton_red_fused_sum_30', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 64, 'r0_': 1024},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_sum_30(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 64
    r0_numel = 1008
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 64*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_factored_query_20261008/inductor_cache/ej/cejdggcjayb4yoz5r5oja4gouhv2ohbibcnutkjkshnhfa6bcmpi.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.select_backward, aten.add]
# Source node to ATen node mapping:
# Graph fragment:
#   %full_default_26 : [num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([3, 64], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %select_scatter_default_1 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_26, %view_62, 0, 2), kwargs = {})
#   %select_scatter_default_3 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_26, %view_65, 0, 1), kwargs = {})
#   %add_35 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_scatter_default_1, %select_scatter_default_3), kwargs = {})
#   %select_scatter_default_5 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_26, %view_69, 0, 0), kwargs = {})
#   %add_39 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_35, %select_scatter_default_5), kwargs = {})
triton_poi_fused_add_select_backward_31 = async_compile.triton('triton_poi_fused_add_select_backward_31', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_select_backward_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_select_backward_31(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 64
    x0 = (xindex % 64)
    x2 = xindex
    tmp3 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 2, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = 0.0
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = tl.full([1], 1, tl.int32)
    tmp7 = tmp0 == tmp6
    tmp9 = tl.where(tmp7, tmp8, tmp4)
    tmp10 = tmp5 + tmp9
    tmp11 = tl.full([1], 0, tl.int32)
    tmp12 = tmp0 == tmp11
    tmp14 = tl.where(tmp12, tmp13, tmp4)
    tmp15 = tmp10 + tmp14
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_factored_query_20261008/inductor_cache/qf/cqfyag57f6yvxtblrty4y2qzld2tdobvz6rplvhmgtv4fkbhj5kz.py
# Topologically Sorted Source Nodes: [visible], Original ATen: [aten.masked_fill, aten.clone, aten.native_layer_norm_backward, aten.where]
# Source node to ATen node mapping:
#   visible => full_default
# Graph fragment:
#   %full_default : [num_users=14] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %clone_17 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_29,), kwargs = {memory_format: torch.contiguous_format})
#   %mul_61 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_29, %primals_29), kwargs = {})
#   %mul_62 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_61, 64), kwargs = {})
#   %sum_33 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_61, [4], True), kwargs = {})
#   %mul_63 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_61, %mul_11), kwargs = {})
#   %sum_34 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_63, [4], True), kwargs = {})
#   %mul_64 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_11, %sum_34), kwargs = {})
#   %sub_19 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_62, %sum_33), kwargs = {})
#   %sub_20 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_19, %mul_64), kwargs = {})
#   %mul_65 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_10, %sub_20), kwargs = {})
#   %where_28 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_24, %mul_65, %full_default), kwargs = {})
triton_per_fused_clone_masked_fill_native_layer_norm_backward_where_32 = async_compile.triton('triton_per_fused_clone_masked_fill_native_layer_norm_backward_where_32', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 131072, 'r0_': 64},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*i1', 'in_ptr4': '*i1', 'in_ptr5': '*i1', 'in_ptr6': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_clone_masked_fill_native_layer_norm_backward_where_32', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 2, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_clone_masked_fill_native_layer_norm_backward_where_32(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 129024
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
    r0_2 = r0_index
    x0 = (xindex % 4)
    x1 = xindex // 4
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_2 + 64*x0 + 1664*x1), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_2 + 64*x0 + 1664*x1), xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r0_2), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (r0_2 + 64*x3), xmask, other=0.0)
    tmp15 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp16 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp18 = tl.load(in_ptr5 + (1 + 4*x1), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp20 = tl.load(in_ptr6 + (x3), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None]
    tmp10 = tmp4 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp13 = tl.where(xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp17 = tmp15 & tmp16
    tmp19 = tmp17 & tmp18
    tmp21 = 64.0
    tmp22 = tmp4 * tmp21
    tmp23 = tmp22 - tmp8
    tmp24 = tmp9 * tmp14
    tmp25 = tmp23 - tmp24
    tmp26 = tmp20 * tmp25
    tmp27 = 0.0
    tmp28 = tl.where(tmp19, tmp26, tmp27)
    tl.store(out_ptr2 + (r0_2 + 64*x3), tmp2, xmask)
    tl.store(in_out_ptr0 + (r0_2 + 64*x3), tmp28, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_factored_query_20261008/inductor_cache/7b/c7btpyswcmjc3uyt2s32crd32tsi3vn7u542ztq5bcafxq7c5jah.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_32 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_70, [0], True), kwargs = {})
triton_red_fused_sum_33 = async_compile.triton('triton_red_fused_sum_33', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 65536, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_33', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_sum_33(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 64512
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 64)
    x1 = xindex // 64
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 64*r0_2 + 8192*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_factored_query_20261008/inductor_cache/lu/clu6lwyqtm345qyrlvojc5a6zpjdfz73wvhhjy2io3b5sef3n3v6.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %add_36 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_19, %sum_27), kwargs = {})
#   %mul_66 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_29, %mul_11), kwargs = {})
#   %sum_35 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_66, [0, 1, 2, 3]), kwargs = {})
#   %add_40 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_36, %sum_35), kwargs = {})
triton_red_fused_add_native_layer_norm_backward_34 = async_compile.triton('triton_red_fused_add_native_layer_norm_backward_34', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 64, 'r0_': 1024},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_backward_34', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_backward_34(in_out_ptr0, in_ptr0, in_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 64
    r0_numel = 1008
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 64*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp4 = tl.load(in_out_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp6 = tmp4 + tmp5
    tmp7 = tmp6 + tmp2
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp7, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_factored_query_20261008/inductor_cache/ly/clyngiajtqub5lg52cntcyxq3qogpcw77xzzcvtx2hgcxj5h47ls.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.add]
# Source node to ATen node mapping:
# Graph fragment:
#   %add_32 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_22, %view_61), kwargs = {})
triton_poi_fused_add_35 = async_compile.triton('triton_poi_fused_add_35', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_35', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_35(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33030144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 64)
    x3 = xindex // 64
    x1 = ((xindex // 64) % 16)
    x2 = xindex // 1024
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*x3), None)
    tmp1 = tl.load(in_ptr1 + (8*x1 + 128*(x0 // 8) + 1024*x2 + ((x0 % 8))), None)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_factored_query_20261008/inductor_cache/ox/coxxeo3fobbtvoez7hzokupmjomjerijadqfh57djxdtertlzamb.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.gelu_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_73 : [num_users=4] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_74, %add_43), kwargs = {})
triton_poi_fused_gelu_backward_36 = async_compile.triton('triton_poi_fused_gelu_backward_36', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_backward_36', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_backward_36(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33030144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = tmp0 * tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_factored_query_20261008/inductor_cache/ug/cugbasezb5sce3wx2v5rwfuqncslmothqkkcctoi32guymgxplex.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
# Source node to ATen node mapping:
# Graph fragment:
#   %constant_pad_nd_default : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%permute_64, [0, 1, 0, 0]), kwargs = {})
triton_poi_fused_mm_37 = async_compile.triton('triton_poi_fused_mm_37', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mm_37', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mm_37(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x1 = xindex // 16
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 15, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x0 + 143*x1), xmask & tmp2, other=0.0)
    tl.store(out_ptr0 + (x2), tmp3, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_factored_query_20261008/inductor_cache/6g/c6gmlescz2q6devdwaai6o2kdeiigfhdoe4jhl5lqicbw7yphgwu.py
# Topologically Sorted Source Nodes: [visible], Original ATen: [aten.masked_fill, aten.sum, aten.where]
# Source node to ATen node mapping:
#   visible => full_default
# Graph fragment:
#   %full_default : [num_users=14] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %sum_38 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_75, [0], True), kwargs = {})
#   %where_29 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_1, %full_default, %mul_73), kwargs = {})
#   %sum_44 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%where_29, [0, 1, 2, 3], True), kwargs = {})
triton_red_fused_masked_fill_sum_where_38 = async_compile.triton('triton_red_fused_masked_fill_sum_where_38', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 512},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i1', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_masked_fill_sum_where_38', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_masked_fill_sum_where_38(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 86016
    r0_numel = 384
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 64)
    x1 = xindex // 64
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 64*r0_2 + 24576*x1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r0_2 + 384*x1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.int1)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask, tmp3, _tmp2)
        tmp5 = 0.0
        tmp6 = tl.where(tmp4, tmp5, tmp0)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask, tmp9, _tmp8)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr1 + (x3), tmp8, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_factored_query_20261008/inductor_cache/e3/ce3ogi6kfijbjc7ovqvvlkuygho2v5cykbpkjc7qk547ze3e63bd.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_39 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_73, [3], True), kwargs = {})
triton_per_fused_sum_39 = async_compile.triton('triton_per_fused_sum_39', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2097152, 'r0_': 16},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_39', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_sum_39(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 2064384
    r0_numel = 16
    R0_BLOCK: tl.constexpr = 16
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 64)
    x1 = xindex // 64
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*r0_2 + 1024*x1), None)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.sum(tmp1, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp3, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_factored_query_20261008/inductor_cache/6u/c6uljfsdylwzkgsc3cu3hrgvzjyn6gybfyptmoh4r73vpcgui7cb.py
# Topologically Sorted Source Nodes: [visible, clamp_min_1], Original ATen: [aten.masked_fill, aten.clamp_min, aten.div, aten.ge, aten.where, aten.slice_backward, aten.add]
# Source node to ATen node mapping:
#   clamp_min_1 => clamp_min_1
#   visible => full_default
# Graph fragment:
#   %full_default : [num_users=14] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%slice_18, 1e-08), kwargs = {})
#   %div_5 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%squeeze_2, %clamp_min_1), kwargs = {})
#   %ge : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%slice_18, 1e-08), kwargs = {})
#   %where_19 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge, %div_5, %full_default), kwargs = {})
#   %full_default_18 : [num_users=4] = call_function[target=torch.ops.aten.full.default](args = ([128, 21, 12, 16, 4], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_scatter_default_1 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_18, %where_19, 4, 0, 1), kwargs = {})
#   %select_scatter_default : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %squeeze_5, 4, 3), kwargs = {})
#   %add_33 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default_1, %select_scatter_default), kwargs = {})
#   %select_scatter_default_2 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %squeeze_7, 4, 2), kwargs = {})
#   %add_34 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_33, %select_scatter_default_2), kwargs = {})
#   %select_scatter_default_4 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %squeeze_9, 4, 1), kwargs = {})
#   %add_38 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_34, %select_scatter_default_4), kwargs = {})
#   %add_44 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_38, %slice_35), kwargs = {})
triton_poi_fused_add_clamp_min_div_ge_masked_fill_slice_backward_where_40 = async_compile.triton('triton_poi_fused_add_clamp_min_div_ge_masked_fill_slice_backward_where_40', '''
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
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_factored_query_20261008/inductor_cache/7t/c7txqpgtykngg7wen4opsmbocmlw6xf7gx2inifuynwwx545qm5n.py
# Topologically Sorted Source Nodes: [layer_norm_1], Original ATen: [aten.native_layer_norm_backward, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_1 => mul_6, sub_3
# Graph fragment:
#   %mul_75 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_79, %primals_19), kwargs = {})
#   %mul_76 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_75, 64), kwargs = {})
#   %sum_40 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_75, [3], True), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_7, %getitem_3), kwargs = {})
#   %mul_6 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_1), kwargs = {})
#   %mul_77 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_75, %mul_6), kwargs = {})
#   %sum_41 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_77, [3], True), kwargs = {})
#   %mul_78 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, %sum_41), kwargs = {})
#   %sub_22 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_76, %sum_40), kwargs = {})
#   %sub_23 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_22, %mul_78), kwargs = {})
#   %div_11 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_1, 64), kwargs = {})
#   %mul_79 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_11, %sub_23), kwargs = {})
triton_per_fused_native_layer_norm_native_layer_norm_backward_41 = async_compile.triton('triton_per_fused_native_layer_norm_native_layer_norm_backward_41', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_41', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_native_layer_norm_backward_41(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32256
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
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 64*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (r0_1 + 64*x0), xmask, other=0.0)
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp9 = tmp7 - tmp8
    tmp11 = tmp9 * tmp10
    tmp12 = tmp2 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp15 = tl.where(xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = 0.015625
    tmp18 = tmp10 * tmp17
    tmp19 = 64.0
    tmp20 = tmp2 * tmp19
    tmp21 = tmp20 - tmp6
    tmp22 = tmp11 * tmp16
    tmp23 = tmp21 - tmp22
    tmp24 = tmp18 * tmp23
    tl.store(out_ptr2 + (r0_1 + 64*x0), tmp24, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_factored_query_20261008/inductor_cache/eg/ceg6n4hqsvsnyiq5px3ynrr7qhshaqj2hhj6zrampzinazkblqqg.py
# Topologically Sorted Source Nodes: [layer_norm_1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   layer_norm_1 => mul_6, sub_3
# Graph fragment:
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_7, %getitem_3), kwargs = {})
#   %mul_6 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_1), kwargs = {})
#   %mul_80 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_79, %mul_6), kwargs = {})
#   %sum_42 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_80, [0, 1, 2]), kwargs = {})
#   %sum_43 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_79, [0, 1, 2]), kwargs = {})
triton_red_fused_native_layer_norm_native_layer_norm_backward_42 = async_compile.triton('triton_red_fused_native_layer_norm_native_layer_norm_backward_42', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_native_layer_norm_backward_42', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_layer_norm_native_layer_norm_backward_42(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 16128
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 64)
    x1 = xindex // 64
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    _tmp11 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 64*r0_2 + 8192*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + 64*r0_2 + 8192*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r0_2 + 128*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr3 + (r0_2 + 128*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp5 = tmp3 * tmp4
        tmp6 = tmp0 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask & xmask, tmp9, _tmp8)
        tmp10 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(r0_mask & xmask, tmp12, _tmp11)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, xmask)
    tl.store(out_ptr1 + (x3), tmp11, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_factored_query_20261008/inductor_cache/rx/crxjxkjnecyhkoamwjevuoynalytpafa45pa5nq4doc73ljhitxi.py
# Topologically Sorted Source Nodes: [layer_norm_1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   layer_norm_1 => mul_6, sub_3
# Graph fragment:
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_7, %getitem_3), kwargs = {})
#   %mul_6 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_1), kwargs = {})
#   %mul_80 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_79, %mul_6), kwargs = {})
#   %sum_42 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_80, [0, 1, 2]), kwargs = {})
triton_red_fused_native_layer_norm_native_layer_norm_backward_43 = async_compile.triton('triton_red_fused_native_layer_norm_native_layer_norm_backward_43', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 64, 'r0_': 256},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_native_layer_norm_backward_43', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_layer_norm_native_layer_norm_backward_43(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 64
    r0_numel = 252
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 64*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_factored_query_20261008/inductor_cache/57/c57r7e3jjaaxjtm5aeab6mnp2po7lanjb3rgoztzswwc6kbkp3yr.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.unfold_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %full_default_49 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([2688, 64, 104], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_unfold_backward_44 = async_compile.triton('triton_poi_fused_unfold_backward_44', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_unfold_backward_44', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_unfold_backward_44(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 17891328
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_factored_query_20261008/inductor_cache/my/cmyohxxpa5c3xr6d4pkmwwvf76bp3a3hg7syju7qqvfz2gq4oanb.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.unfold_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %full_default_49 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([2688, 64, 104], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_49, [None, None, %view_82], %view_83, True), kwargs = {})
triton_poi_fused_unfold_backward_45 = async_compile.triton('triton_poi_fused_unfold_backward_45', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 262144, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_unfold_backward_45', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_unfold_backward_45(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 172032
    xnumel = 192
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y1 = yindex // 64
    y0 = (yindex % 64)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (x2 + 192*y1), ymask & xmask, eviction_policy='evict_last').to(tl.int1)
    tmp2 = tl.load(in_ptr1 + (y0 + 64*x2 + 12288*y1), ymask & xmask, eviction_policy='evict_last')
    tmp1 = tmp0 == 0
    tmp3 = 0.0
    tmp4 = tl.where(tmp0, tmp2, tmp3)
    tmp5 = tl.where(tmp1, tmp3, tmp4)
    tl.atomic_add(out_ptr0 + (8*(x2 // 16) + 104*y3 + ((x2 % 16))), tmp5, ymask & xmask, sem='relaxed')
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_factored_query_20261008/inductor_cache/yr/cyrgwonp3rx5qtpv67gjrmcntw4ilwvm2ugh7yyziekug65dbjds.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.clone, aten._unsafe_view]
# Source node to ATen node mapping:
# Graph fragment:
#   %clone_19 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_75,), kwargs = {memory_format: torch.contiguous_format})
#   %view_84 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_19, [258048, 64]), kwargs = {})
triton_poi_fused__unsafe_view_clone_46 = async_compile.triton('triton_poi_fused__unsafe_view_clone_46', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 262144, 'x': 64}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_46', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_clone_46(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 258048
    xnumel = 64
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex
    x1 = xindex
    tmp0 = (y0 % 96)
    tmp1 = tl.full([1, 1], 104, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (104*x1 + 6656*(y0 // 96) + ((y0 % 96))), xmask & ymask & tmp2, eviction_policy='evict_last', other=0.0)
    tl.store(out_ptr0 + (x1 + 64*y0), tmp3, xmask & ymask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_factored_query_20261008/inductor_cache/zw/czwl5m4znydewzbrk5fyjgdc4xv2nov6cvox3ms5ambasiwnjdgv.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.slice_backward, aten.add]
# Source node to ATen node mapping:
# Graph fragment:
#   %full_default_39 : [num_users=4] = call_function[target=torch.ops.aten.full.default](args = ([64, 143], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_scatter_default_2 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_39, %permute_66, 1, 128, 9223372036854775807), kwargs = {})
#   %slice_scatter_default_3 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_39, %permute_69, 1, 64, 128), kwargs = {})
#   %add_45 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default_2, %slice_scatter_default_3), kwargs = {})
#   %slice_scatter_default_4 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_39, %permute_72, 1, 0, 64), kwargs = {})
#   %add_46 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_45, %slice_scatter_default_4), kwargs = {})
#   %slice_scatter_default_5 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_39, %permute_78, 1, 0, 64), kwargs = {})
#   %add_47 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_46, %slice_scatter_default_5), kwargs = {})
triton_poi_fused_add_slice_backward_47 = async_compile.triton('triton_poi_fused_add_slice_backward_47', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_slice_backward_47', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_slice_backward_47(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 143
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex
    x1 = xindex
    tmp0 = y0
    tmp1 = tl.full([1, 1], 128, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-8192) + x1 + 64*y0), xmask & ymask & tmp2, eviction_policy='evict_last', other=0.0)
    tmp4 = 0.0
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = tl.full([1, 1], 64, tl.int64)
    tmp7 = tmp0 >= tmp6
    tmp8 = tmp0 < tmp1
    tmp9 = tmp7 & tmp8
    tmp10 = tl.load(in_ptr1 + ((-4096) + x1 + 64*y0), xmask & ymask & tmp9, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.where(tmp9, tmp10, tmp4)
    tmp12 = tmp5 + tmp11
    tmp13 = tmp0 < tmp6
    tmp14 = tl.load(in_ptr2 + (x1 + 64*y0), xmask & ymask & tmp13, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.where(tmp13, tmp14, tmp4)
    tmp16 = tmp12 + tmp15
    tmp17 = tl.load(in_ptr3 + (x1 + 64*y0), xmask & ymask & tmp13, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.where(tmp13, tmp17, tmp4)
    tmp19 = tmp16 + tmp18
    tl.store(out_ptr0 + (y0 + 143*x1), tmp19, xmask & ymask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_factored_query_20261008/inductor_cache/df/cdfw7rx7dmmjafvwfoimomfv4bm2r3chdqgf7qfdp7cbn7axiygh.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %sub_24 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%permute_5, %getitem_1), kwargs = {})
#   %mul_81 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, %rsqrt), kwargs = {})
#   %mul_82 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_85, %primals_17), kwargs = {})
#   %mul_83 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_82, 64), kwargs = {})
#   %sum_45 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_82, [2], True), kwargs = {})
#   %mul_84 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_82, %mul_81), kwargs = {})
#   %sum_46 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_84, [2], True), kwargs = {})
#   %mul_85 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_81, %sum_46), kwargs = {})
#   %sub_25 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_83, %sum_45), kwargs = {})
#   %sub_26 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_25, %mul_85), kwargs = {})
#   %div_12 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt, 64), kwargs = {})
#   %mul_86 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_12, %sub_26), kwargs = {})
triton_per_fused_native_layer_norm_backward_48 = async_compile.triton('triton_per_fused_native_layer_norm_backward_48', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 262144, 'r0_': 64},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_48', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_backward_48(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 258048
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    x2 = (xindex % 96)
    x3 = xindex // 96
    tmp0 = tl.load(in_ptr0 + (r0_1 + 64*x0), None)
    tmp1 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x2 + 96*r0_1 + 6144*x3), None)
    tmp7 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.sum(tmp3, 1)[:, None]
    tmp8 = tmp6 - tmp7
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
    tmp14 = tl.sum(tmp12, 1)[:, None]
    tmp15 = 0.015625
    tmp16 = tmp9 * tmp15
    tmp17 = 64.0
    tmp18 = tmp2 * tmp17
    tmp19 = tmp18 - tmp5
    tmp20 = tmp10 * tmp14
    tmp21 = tmp19 - tmp20
    tmp22 = tmp16 * tmp21
    tl.store(out_ptr2 + (r0_1 + 64*x0), tmp22, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_factored_query_20261008/inductor_cache/u2/cu2f3ljz4rp6v7j3u6z5ju5pwmsmmzoeh65br3leerncpcnri7h5.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %sub_24 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%permute_5, %getitem_1), kwargs = {})
#   %mul_81 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, %rsqrt), kwargs = {})
#   %mul_87 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_85, %mul_81), kwargs = {})
#   %sum_47 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_87, [0, 1]), kwargs = {})
#   %sum_48 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_85, [0, 1]), kwargs = {})
triton_red_fused_native_layer_norm_backward_49 = async_compile.triton('triton_red_fused_native_layer_norm_backward_49', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 256},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_49', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_layer_norm_backward_49(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 86016
    r0_numel = 192
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 64)
    x1 = xindex // 64
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    _tmp11 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 64*r0_2 + 12288*x1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (96*x0 + 6144*(r0_2 // 96) + 12288*x1 + ((r0_2 % 96))), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r0_2 + 192*x1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr3 + (r0_2 + 192*x1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp5 = tmp3 * tmp4
        tmp6 = tmp0 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask, tmp9, _tmp8)
        tmp10 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(r0_mask, tmp12, _tmp11)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, None)
    tl.store(out_ptr1 + (x3), tmp11, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_factored_query_20261008/inductor_cache/iw/ciweazkqzjfgzhbzbg6vvug32ow4vc732yirm5ghzcm33whog2zz.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.add]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_48 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_85, [0, 1]), kwargs = {})
#   %add_48 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%squeeze_11, %sum_48), kwargs = {})
triton_red_fused_add_native_layer_norm_backward_50 = async_compile.triton('triton_red_fused_add_native_layer_norm_backward_50', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 64, 'r0_': 2048},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_backward_50', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_backward_50(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 64
    r0_numel = 1344
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 64*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp4 = tl.load(in_out_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp4 + tmp2
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_factored_query_20261008/inductor_cache/gs/cgsujqt5rmheojmdot537ydxmvmg2vdcqilzxhgdhlkvgic66rto.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.div, aten.mul]
# Source node to ATen node mapping:
# Graph fragment:
#   %div_13 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%squeeze_12, %unsqueeze_16), kwargs = {})
#   %mul_88 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_13, %unsqueeze_14), kwargs = {})
triton_poi_fused_div_mul_51 = async_compile.triton('triton_poi_fused_div_mul_51', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_mul_51', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_mul_51(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 516096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 192
    tmp0 = tl.load(in_ptr0 + (6 + 16*x2), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp1 * tmp1
    tmp3 = 0.010000000000000002
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.sqrt(tmp4)
    tmp6 = (tmp0 / tmp5)
    tmp7 = tmp6 * tmp1
    tl.store(out_ptr0 + (x2), tmp7, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_2, primals_3, primals_4, primals_5, primals_7, primals_8, primals_9, primals_17, primals_19, primals_24, primals_28, primals_29, primals_34, primals_38, primals_41, primals_45, primals_52, getitem_1, rsqrt, getitem_3, rsqrt_1, view_14, mul_11, view_16, unsqueeze_27, mul_14, view_18, unsqueeze_33, view_20, unsqueeze_39, cat_4, view_23, cat_5, div_3, cat_7, getitem_11, rsqrt_5, view_36, addmm_7, view_38, unsqueeze_44, permute_25, permute_29, permute_35, permute_36, permute_37, permute_40, permute_45, mul_42, div_8, permute_53, div_9, div_10, permute_60, add_43, permute_64, permute_65, permute_67, permute_68, permute_70, permute_71, permute_76, tangents_1, tangents_2 = args
    args.clear()
    assert_size_stride(primals_2, (128, 21, 12, 16), (4032, 192, 16, 1))
    assert_size_stride(primals_3, (128, 21, 12, 16), (4032, 192, 16, 1))
    assert_size_stride(primals_4, (128, 21), (21, 1))
    assert_size_stride(primals_5, (2688, 64, 96), (6144, 96, 1))
    assert_size_stride(primals_7, (128, 21, 12, 64), (16128, 768, 64, 1))
    assert_size_stride(primals_8, (128, 21, 12, 4), (1008, 48, 4, 1))
    assert_size_stride(primals_9, (128, 21, 12, 16, 4), (16128, 768, 64, 4, 1))
    assert_size_stride(primals_17, (64, ), (1, ))
    assert_size_stride(primals_19, (64, ), (1, ))
    assert_size_stride(primals_24, (128, 21, 12), (252, 12, 1))
    assert_size_stride(primals_28, (128, 21, 12, 4), (1008, 48, 4, 1))
    assert_size_stride(primals_29, (64, ), (1, ))
    assert_size_stride(primals_34, (128, 21, 12), (252, 12, 1))
    assert_size_stride(primals_38, (128, 21, 12, 12), (3024, 144, 12, 1))
    assert_size_stride(primals_41, (128, 21, 12), (252, 12, 1))
    assert_size_stride(primals_45, (128, 21, 12, 10), (2520, 120, 1, 12))
    assert_size_stride(primals_52, (128, ), (1, ))
    assert_size_stride(getitem_1, (2688, 96, 1), (96, 1, 1))
    assert_size_stride(rsqrt, (2688, 96, 1), (96, 1, 1))
    assert_size_stride(getitem_3, (128, 21, 12, 1), (252, 12, 1, 1))
    assert_size_stride(rsqrt_1, (128, 21, 12, 1), (252, 12, 1, 1))
    assert_size_stride(view_14, (516096, 64), (64, 1))
    assert_size_stride(mul_11, (128, 21, 12, 4, 64), (64512, 3072, 256, 64, 1))
    assert_size_stride(view_16, (129024, 5), (5, 1))
    assert_size_stride(unsqueeze_27, (128, 21, 12, 1, 4), (1008, 48, 4, 4, 1))
    assert_size_stride(mul_14, (128, 21, 12, 12, 64), (193536, 9216, 768, 64, 1))
    assert_size_stride(view_18, (387072, 4), (4, 1))
    assert_size_stride(unsqueeze_33, (128, 21, 12, 1, 12), (3040, 144, 12, 0, 1))
    assert_size_stride(view_20, (322560, 5), (5, 1))
    assert_size_stride(unsqueeze_39, (128, 21, 12, 1, 10), (2528, 120, 1, 120, 12))
    assert_size_stride(cat_4, (128, 21, 12, 26), (6552, 312, 26, 1))
    assert_size_stride(view_23, (838656, 64), (64, 1))
    assert_size_stride(cat_5, (128, 21, 12, 16, 26), (104832, 4992, 416, 26, 1))
    assert_size_stride(div_3, (128, 21, 12, 8, 16, 27), (870912, 41472, 3456, 432, 27, 1))
    assert_size_stride(cat_7, (128, 21, 12, 16, 128), (516096, 24576, 2048, 128, 1))
    assert_size_stride(getitem_11, (128, 21, 12, 16, 1), (4032, 192, 16, 1, 1))
    assert_size_stride(rsqrt_5, (128, 21, 12, 16, 1), (4032, 192, 16, 1, 1))
    assert_size_stride(view_36, (516096, 128), (128, 1))
    assert_size_stride(addmm_7, (516096, 64), (64, 1))
    assert_size_stride(view_38, (516096, 64), (64, 1))
    assert_size_stride(unsqueeze_44, (128, 21, 12, 1), (252, 12, 1, 1))
    assert_size_stride(permute_25, (1, 64), (64, 1))
    assert_size_stride(permute_29, (64, 128), (128, 1))
    assert_size_stride(permute_35, (258048, 8, 26), (208, 1, 8))
    assert_size_stride(permute_36, (258048, 8, 16), (128, 1, 8))
    assert_size_stride(permute_37, (258048, 26, 8), (208, 1, 26))
    assert_size_stride(permute_40, (64, 64), (64, 1))
    assert_size_stride(permute_45, (64, 64), (64, 1))
    assert_size_stride(mul_42, (128, 21, 12, 10, 64), (161280, 7680, 64, 768, 1))
    assert_size_stride(div_8, (128, 21, 12, 10, 1), (2528, 120, 10, 1, 1))
    assert_size_stride(permute_53, (64, 4), (4, 1))
    assert_size_stride(div_9, (128, 21, 12, 12, 1), (3040, 144, 12, 1, 1))
    assert_size_stride(div_10, (128, 21, 12, 4, 1), (1008, 48, 4, 1, 1))
    assert_size_stride(permute_60, (64, 64), (64, 1))
    assert_size_stride(add_43, (128, 21, 12, 16, 64), (258048, 12288, 1024, 64, 1))
    assert_size_stride(permute_64, (64, 15), (143, 1))
    assert_size_stride(permute_65, (15, 516096), (1, 15))
    assert_size_stride(permute_67, (64, 32256), (1, 64))
    assert_size_stride(permute_68, (64, 64), (143, 1))
    assert_size_stride(permute_70, (64, 1), (1, 64))
    assert_size_stride(permute_71, (64, 64), (143, 1))
    assert_size_stride(permute_76, (64, 258048), (1, 64))
    assert_size_stride(tangents_1, (128, 21, 12, 16), (4032, 192, 16, 1))
    assert_size_stride(tangents_2, (128, 21, 12, 8, 16, 27), (870912, 41472, 3456, 432, 27, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 21, 12, 16), (4032, 192, 16, 1), torch.float32)
        buf3 = empty_strided_cuda((1, 1, 63), (63, 63, 1), torch.float32)
        # Topologically Sorted Source Nodes: [visible, invert_6, and__6, eligible], Original ATen: [aten.masked_fill, aten.bitwise_not, aten.bitwise_and, aten.where, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_bitwise_and_bitwise_not_masked_fill_sum_where_0.run(primals_3, primals_2, unsqueeze_44, tangents_1, buf0, buf3, 63, 8192, stream=stream0)
        del primals_2
        del tangents_1
        del unsqueeze_44
        buf1 = empty_strided_cuda((516096, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf0, (516096, 1), (1, 0), 0), permute_25, out=buf1)
        del permute_25
        buf2 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf0, (1, 516096), (0, 1), 0), view_38, out=buf2)
        del view_38
        buf4 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_1.run(buf3, buf4, 1, 63, stream=stream0)
        del buf3
        buf5 = reinterpret_tensor(buf1, (128, 21, 12, 16, 64), (258048, 12288, 1024, 64, 1), 0); del buf1  # reuse
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.gelu, aten.gelu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_gelu_backward_2.run(buf5, addmm_7, 33030144, stream=stream0)
        del addmm_7
        buf6 = empty_strided_cuda((516096, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (516096, 64), (64, 1), 0), permute_29, out=buf6)
        del permute_29
        buf7 = empty_strided_cuda((64, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (64, 516096), (1, 64), 0), view_36, out=buf7)
        del view_36
        buf8 = empty_strided_cuda((1, 64, 1344), (86016, 1, 64), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_3.run(buf5, buf8, 86016, 384, stream=stream0)
        buf9 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_4.run(buf8, buf9, 64, 1344, stream=stream0)
        buf12 = empty_strided_cuda((128, 21, 12, 16, 128), (516096, 24576, 2048, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.native_layer_norm_backward, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_layer_norm_native_layer_norm_backward_5.run(buf6, primals_52, cat_7, getitem_11, rsqrt_5, buf12, 516096, 128, stream=stream0)
        del primals_52
        buf13 = empty_strided_cuda((128, 1344), (1, 128), torch.float32)
        buf15 = empty_strided_cuda((128, 1344), (1, 128), torch.float32)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_layer_norm_native_layer_norm_backward_6.run(buf6, cat_7, getitem_11, rsqrt_5, buf13, buf15, 172032, 384, stream=stream0)
        del buf6
        del cat_7
        del getitem_11
        del rsqrt_5
        buf14 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_layer_norm_native_layer_norm_backward_7.run(buf13, buf14, 128, 1344, stream=stream0)
        del buf13
        buf16 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_layer_norm_native_layer_norm_backward_7.run(buf15, buf16, 128, 1344, stream=stream0)
        del buf15
        buf17 = reinterpret_tensor(buf5, (128, 21, 12, 8, 16, 8), (258048, 12288, 1024, 128, 8, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_8.run(buf12, buf17, 33030144, stream=stream0)
        buf18 = empty_strided_cuda((258048, 26, 8), (208, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(div_3, (258048, 26, 16), (432, 1, 27), 1), reinterpret_tensor(buf17, (258048, 16, 8), (128, 8, 1), 0), out=buf18)
        buf19 = empty_strided_cuda((258048, 16, 26), (416, 26, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf17, (258048, 16, 8), (128, 8, 1), 0), permute_35, out=buf19)
        del permute_35
        buf20 = empty_strided_cuda((128, 21, 12, 8, 16, 1), (32256, 1536, 128, 16, 1, 4128768), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.slice_backward, aten.add, aten._softmax_backward_data]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_backward_data_add_slice_backward_9.run(tangents_2, buf19, div_3, buf20, 4128768, 27, stream=stream0)
        buf21 = reinterpret_tensor(buf0, (128, 21, 12, 1, 16, 1), (4032, 192, 16, 16, 1, 1), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_10.run(div_3, buf20, tangents_2, buf19, buf21, 516096, 8, stream=stream0)
        buf22 = reinterpret_tensor(buf19, (128, 21, 12, 8, 16, 26), (838656, 39936, 3328, 416, 26, 1), 0); del buf19  # reuse
        buf24 = empty_strided_cuda((128, 21, 12, 8, 16, 26), (838656, 39936, 3328, 416, 26, 1), torch.float32)
        # Topologically Sorted Source Nodes: [visible, invert_5], Original ATen: [aten.masked_fill, aten.bitwise_not, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bitwise_not_div_masked_fill_11.run(buf22, cat_4, div_3, buf20, tangents_2, buf24, 107347968, stream=stream0)
        del buf20
        del cat_4
        del div_3
        del tangents_2
        buf23 = empty_strided_cuda((128, 21, 12, 1, 16, 26), (104832, 4992, 416, 416, 26, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_12.run(buf22, buf23, 13418496, 8, stream=stream0)
        del buf22
        buf25 = empty_strided_cuda((258048, 8, 26), (208, 26, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_36, reinterpret_tensor(buf24, (258048, 16, 26), (416, 26, 1), 0), out=buf25)
        del permute_36
        buf26 = reinterpret_tensor(buf17, (258048, 16, 8), (128, 8, 1), 0); del buf17  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf24, (258048, 16, 26), (416, 26, 1), 0), permute_37, out=buf26)
        del buf24
        del permute_37
        buf27 = empty_strided_cuda((838656, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_13.run(buf18, buf27, 53673984, stream=stream0)
        buf28 = reinterpret_tensor(buf18, (838656, 64), (64, 1), 0); del buf18  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf27, permute_40, out=buf28)
        del permute_40
        buf29 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf27, (64, 838656), (1, 64), 0), view_23, out=buf29)
        buf30 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_14.run(buf27, buf30, 86016, 624, stream=stream0)
        buf31 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_4.run(buf30, buf31, 64, 1344, stream=stream0)
        buf32 = buf27; del buf27  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone, aten._unsafe_view]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_15.run(buf25, buf32, 838656, 64, stream=stream0)
        buf33 = reinterpret_tensor(buf25, (838656, 64), (64, 1), 0); del buf25  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf32, permute_45, out=buf33)
        del permute_45
        buf34 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf32, (64, 838656), (1, 64), 0), view_23, out=buf34)
        del view_23
        buf35 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_14.run(buf32, buf35, 86016, 624, stream=stream0)
        del buf32
        buf36 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_4.run(buf35, buf36, 64, 1344, stream=stream0)
        buf37 = empty_strided_cuda((128, 21, 12, 16, 1), (4032, 192, 16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_16.run(cat_5, buf23, unsqueeze_39, buf37, 516096, 10, stream=stream0)
        del unsqueeze_39
        buf38 = empty_strided_cuda((128, 21, 12, 1, 10), (2528, 120, 10, 323584, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_17.run(cat_5, buf23, primals_9, buf38, 322560, 16, stream=stream0)
        buf39 = empty_strided_cuda((128, 21, 12, 10), (2520, 120, 1, 12), torch.float32)
        # Topologically Sorted Source Nodes: [visible, and__4, valid_2], Original ATen: [aten.masked_fill, aten.bitwise_and, aten.where]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bitwise_and_masked_fill_where_18.run(primals_45, primals_41, primals_8, buf38, buf39, 26880, 12, stream=stream0)
        del buf38
        buf40 = reinterpret_tensor(buf35, (1, 1, 1, 1, 64, 1344), (86016, 86016, 86016, 86016, 1, 64), 0); del buf35  # reuse
        buf42 = empty_strided_cuda((1, 1, 1, 1, 64, 1344), (86016, 86016, 86016, 86016, 1, 64), torch.float32)
        buf48 = empty_strided_cuda((64, 1344), (1, 64), torch.float32)
        buf50 = empty_strided_cuda((64, 1344), (1, 64), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_layer_norm_backward_sum_19.run(buf28, buf33, mul_42, buf40, buf42, buf48, buf50, 86016, 240, stream=stream0)
        buf41 = empty_strided_cuda((1, 1, 1, 1, 64), (64, 64, 64, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_4.run(buf40, buf41, 64, 1344, stream=stream0)
        del buf40
        buf43 = empty_strided_cuda((1, 1, 1, 1, 64), (64, 64, 64, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_4.run(buf42, buf43, 64, 1344, stream=stream0)
        buf44 = empty_strided_cuda((128, 21, 12, 10, 64), (161280, 7680, 640, 64, 1), torch.float32)
        buf52 = mul_42; del mul_42  # reuse
        # Topologically Sorted Source Nodes: [visible], Original ATen: [aten.masked_fill, aten.clone, aten.native_layer_norm_backward, aten.where]
        stream0 = get_raw_stream(0)
        triton_per_fused_clone_masked_fill_native_layer_norm_backward_where_20.run(buf52, buf28, buf33, primals_29, primals_45, primals_41, primals_8, div_8, buf44, 322560, 64, stream=stream0)
        del div_8
        del primals_41
        del primals_45
        buf45 = empty_strided_cuda((64, 5), (5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf44, (64, 322560), (1, 64), 0), view_20, out=buf45)
        del buf44
        del view_20
        buf49 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_4.run(buf48, buf49, 64, 1344, stream=stream0)
        buf51 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_4.run(buf50, buf51, 64, 1344, stream=stream0)
        buf53 = empty_strided_cuda((128, 21, 12, 16, 1), (4032, 192, 16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_21.run(cat_5, buf23, unsqueeze_33, buf53, 516096, 12, stream=stream0)
        del unsqueeze_33
        buf55 = empty_strided_cuda((128, 21, 12, 12), (3024, 144, 12, 1), torch.float32)
        # Topologically Sorted Source Nodes: [visible, and__2, valid_1], Original ATen: [aten.masked_fill, aten.mul, aten.sum, aten.bitwise_and, aten.where]
        stream0 = get_raw_stream(0)
        triton_per_fused_bitwise_and_masked_fill_mul_sum_where_22.run(cat_5, buf23, primals_9, primals_38, primals_34, primals_8, buf55, 387072, 16, stream=stream0)
        buf56 = reinterpret_tensor(buf50, (1, 1, 1, 1, 64, 1344), (86016, 86016, 86016, 86016, 1, 64), 0); del buf50  # reuse
        buf65 = buf48; del buf48  # reuse
        buf67 = reinterpret_tensor(buf42, (64, 1344), (1, 64), 0); del buf42  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_layer_norm_backward_sum_23.run(buf28, buf33, mul_14, buf56, buf65, buf67, 86016, 288, stream=stream0)
        buf57 = empty_strided_cuda((1, 1, 1, 1, 64), (64, 64, 64, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_4.run(buf56, buf57, 64, 1344, stream=stream0)
        buf58 = empty_strided_cuda((128, 21, 12, 12, 64), (193536, 9216, 768, 64, 1), torch.float32)
        buf70 = mul_14; del mul_14  # reuse
        # Topologically Sorted Source Nodes: [visible], Original ATen: [aten.masked_fill, aten.clone, aten.native_layer_norm_backward, aten.where]
        stream0 = get_raw_stream(0)
        triton_per_fused_clone_masked_fill_native_layer_norm_backward_where_24.run(buf70, buf28, buf33, primals_29, primals_38, primals_34, primals_8, div_9, buf58, 387072, 64, stream=stream0)
        del div_9
        buf59 = empty_strided_cuda((387072, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf58, (387072, 64), (64, 1), 0), permute_53, out=buf59)
        del permute_53
        buf60 = empty_strided_cuda((64, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf58, (64, 387072), (1, 64), 0), view_18, out=buf60)
        del view_18
        buf61 = reinterpret_tensor(buf56, (1, 64, 1344), (86016, 1, 64), 0); del buf56  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_25.run(buf58, buf61, 86016, 288, stream=stream0)
        del buf58
        buf62 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_4.run(buf61, buf62, 64, 1344, stream=stream0)
        del buf61
        buf66 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_4.run(buf65, buf66, 64, 1344, stream=stream0)
        buf68 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_4.run(buf67, buf68, 64, 1344, stream=stream0)
        buf69 = reinterpret_tensor(buf59, (128, 21, 12, 12, 4), (12096, 576, 48, 4, 1), 0); del buf59  # reuse
        # Topologically Sorted Source Nodes: [visible], Original ATen: [aten.masked_fill, aten.where]
        stream0 = get_raw_stream(0)
        triton_poi_fused_masked_fill_where_26.run(buf69, primals_38, primals_34, primals_8, 1548288, stream=stream0)
        del primals_34
        del primals_38
        buf71 = empty_strided_cuda((128, 21, 12, 16, 1), (4032, 192, 16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sum_27.run(cat_5, buf23, unsqueeze_27, buf71, 516096, stream=stream0)
        del unsqueeze_27
        buf72 = empty_strided_cuda((128, 21, 12, 1, 4), (1008, 48, 4, 4, 1), torch.float32)
        buf73 = reinterpret_tensor(buf72, (128, 21, 12, 4), (1008, 48, 4, 1), 0); del buf72  # reuse
        # Topologically Sorted Source Nodes: [visible, and_, valid], Original ATen: [aten.masked_fill, aten.mul, aten.sum, aten.bitwise_and, aten.where]
        stream0 = get_raw_stream(0)
        triton_per_fused_bitwise_and_masked_fill_mul_sum_where_28.run(buf73, cat_5, buf23, primals_9, primals_28, primals_24, primals_8, 129024, 16, stream=stream0)
        del buf23
        del cat_5
        buf74 = empty_strided_cuda((1, 1, 1, 1, 64, 1008), (64512, 64512, 64512, 64512, 1, 64), torch.float32)
        buf83 = empty_strided_cuda((64, 1008), (1, 64), torch.float32)
        buf85 = empty_strided_cuda((64, 1008), (1, 64), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_layer_norm_backward_sum_29.run(buf28, buf33, mul_11, buf74, buf83, buf85, 64512, 128, stream=stream0)
        buf75 = empty_strided_cuda((1, 1, 1, 1, 64), (64, 64, 64, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_30.run(buf74, buf75, 64, 1008, stream=stream0)
        buf76 = empty_strided_cuda((3, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.select_backward, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_select_backward_31.run(buf41, buf57, buf75, buf76, 192, stream=stream0)
        buf77 = empty_strided_cuda((128, 21, 12, 4, 64), (64512, 3072, 256, 64, 1), torch.float32)
        buf89 = mul_11; del mul_11  # reuse
        # Topologically Sorted Source Nodes: [visible], Original ATen: [aten.masked_fill, aten.clone, aten.native_layer_norm_backward, aten.where]
        stream0 = get_raw_stream(0)
        triton_per_fused_clone_masked_fill_native_layer_norm_backward_where_32.run(buf89, buf28, buf33, primals_29, primals_28, primals_24, primals_8, div_10, buf77, 129024, 64, stream=stream0)
        del buf28
        del buf33
        del div_10
        del primals_24
        del primals_28
        del primals_29
        del primals_8
        buf78 = empty_strided_cuda((64, 5), (5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf77, (64, 129024), (1, 64), 0), view_16, out=buf78)
        del view_16
        buf79 = reinterpret_tensor(buf74, (1, 64, 1008), (64512, 1, 64), 0); del buf74  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_33.run(buf77, buf79, 64512, 128, stream=stream0)
        buf80 = reinterpret_tensor(buf75, (1, 64), (64, 1), 0); del buf75  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_30.run(buf79, buf80, 64, 1008, stream=stream0)
        del buf79
        buf87 = buf49; del buf49  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_native_layer_norm_backward_34.run(buf87, buf83, buf66, 64, 1008, stream=stream0)
        del buf83
        buf88 = buf51; del buf51  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_native_layer_norm_backward_34.run(buf88, buf85, buf68, 64, 1008, stream=stream0)
        del buf85
        buf90 = empty_strided_cuda((128, 21, 12, 16, 64), (258048, 12288, 1024, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_35.run(buf12, buf26, buf90, 33030144, stream=stream0)
        del buf12
        buf91 = reinterpret_tensor(buf26, (516096, 64), (64, 1), 0); del buf26  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf90, (516096, 64), (64, 1), 0), permute_60, out=buf91)
        del permute_60
        buf92 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf90, (64, 516096), (1, 64), 0), view_14, out=buf92)
        del view_14
        buf93 = reinterpret_tensor(buf67, (1, 64, 1344), (86016, 1, 64), 0); del buf67  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_3.run(buf90, buf93, 86016, 384, stream=stream0)
        del buf90
        buf94 = reinterpret_tensor(buf68, (1, 64), (64, 1), 0); del buf68  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_4.run(buf93, buf94, 64, 1344, stream=stream0)
        buf95 = reinterpret_tensor(buf91, (128, 21, 12, 16, 64), (258048, 12288, 1024, 64, 1), 0); del buf91  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.gelu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_backward_36.run(buf95, add_43, 33030144, stream=stream0)
        del add_43
        buf96 = empty_strided_cuda((64, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mm_37.run(permute_64, buf96, 1024, stream=stream0)
        del permute_64
        buf97 = reinterpret_tensor(buf77, (516096, 16), (16, 1), 0); del buf77  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf95, (516096, 64), (64, 1), 0), buf96, out=buf97)
        del buf96
        buf98 = empty_strided_cuda((15, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_65, reinterpret_tensor(buf95, (516096, 64), (64, 1), 0), out=buf98)
        del permute_65
        buf99 = buf93; del buf93  # reuse
        buf112 = reinterpret_tensor(buf65, (1, 1, 1, 1, 64, 1344), (86016, 86016, 86016, 86016, 1, 64), 0); del buf65  # reuse
        # Topologically Sorted Source Nodes: [visible], Original ATen: [aten.masked_fill, aten.sum, aten.where]
        stream0 = get_raw_stream(0)
        triton_red_fused_masked_fill_sum_where_38.run(buf95, primals_3, buf99, buf112, 86016, 384, stream=stream0)
        buf100 = reinterpret_tensor(buf66, (1, 64), (64, 1), 0); del buf66  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_4.run(buf99, buf100, 64, 1344, stream=stream0)
        buf101 = empty_strided_cuda((128, 21, 12, 1, 64), (16128, 768, 64, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_39.run(buf95, buf101, 2064384, 16, stream=stream0)
        buf102 = empty_strided_cuda((128, 21, 12, 16, 4), (16128, 768, 64, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [visible, clamp_min_1], Original ATen: [aten.masked_fill, aten.clamp_min, aten.div, aten.ge, aten.where, aten.slice_backward, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_min_div_ge_masked_fill_slice_backward_where_40.run(primals_9, buf21, buf37, buf53, buf71, buf97, buf102, 2064384, stream=stream0)
        del buf21
        del buf37
        del buf53
        del primals_9
        buf103 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_67, reinterpret_tensor(buf101, (32256, 64), (64, 1), 0), out=buf103)
        del permute_67
        buf104 = empty_strided_cuda((32256, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf101, (32256, 64), (64, 1), 0), permute_68, out=buf104)
        del permute_68
        buf107 = reinterpret_tensor(buf101, (128, 21, 12, 64), (16128, 768, 64, 1), 0); del buf101  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_1], Original ATen: [aten.native_layer_norm_backward, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_layer_norm_native_layer_norm_backward_41.run(buf104, primals_19, primals_7, getitem_3, rsqrt_1, buf107, 32256, 64, stream=stream0)
        del primals_19
        buf108 = empty_strided_cuda((64, 252), (1, 64), torch.float32)
        buf110 = empty_strided_cuda((64, 252), (1, 64), torch.float32)
        # Topologically Sorted Source Nodes: [layer_norm_1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_layer_norm_native_layer_norm_backward_42.run(buf104, primals_7, getitem_3, rsqrt_1, buf108, buf110, 16128, 128, stream=stream0)
        del buf104
        del getitem_3
        del primals_7
        del rsqrt_1
        buf109 = reinterpret_tensor(buf57, (64, ), (1, ), 0); del buf57  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_layer_norm_native_layer_norm_backward_43.run(buf108, buf109, 64, 252, stream=stream0)
        del buf108
        buf111 = reinterpret_tensor(buf41, (64, ), (1, ), 0); del buf41  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_layer_norm_native_layer_norm_backward_43.run(buf110, buf111, 64, 252, stream=stream0)
        del buf110
        buf113 = empty_strided_cuda((1, 1, 1, 1, 64), (64, 64, 64, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [visible], Original ATen: [aten.masked_fill, aten.where, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_4.run(buf112, buf113, 64, 1344, stream=stream0)
        buf114 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_70, reinterpret_tensor(buf113, (1, 64), (64, 1), 0), out=buf114)
        del permute_70
        buf115 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf113, (1, 64), (64, 1), 0), permute_71, out=buf115)
        buf116 = empty_strided_cuda((2688, 64, 104), (6656, 104, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.unfold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_unfold_backward_44.run(buf116, 17891328, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.unfold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_unfold_backward_45.run(primals_3, buf95, buf116, 172032, 192, stream=stream0)
        del buf95
        del primals_3
        buf118 = empty_strided_cuda((258048, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone, aten._unsafe_view]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_46.run(buf116, buf118, 258048, 64, stream=stream0)
        del buf116
        buf119 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_76, buf118, out=buf119)
        del permute_76
        buf120 = empty_strided_cuda((258048, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf118, permute_71, out=buf120)
        del permute_71
        buf121 = empty_strided_cuda((64, 143), (143, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.slice_backward, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_slice_backward_47.run(buf98, buf103, buf114, buf119, buf121, 143, 64, stream=stream0)
        del buf103
        del buf114
        del buf119
        del buf98
        buf129 = reinterpret_tensor(buf118, (2688, 96, 64), (6144, 64, 1), 0); del buf118  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_layer_norm_backward_48.run(buf120, primals_17, primals_5, getitem_1, rsqrt, buf129, 258048, 64, stream=stream0)
        del primals_17
        buf124 = reinterpret_tensor(buf112, (64, 1344), (1, 64), 0); del buf112  # reuse
        buf126 = reinterpret_tensor(buf99, (64, 1344), (1, 64), 0); del buf99  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_layer_norm_backward_49.run(buf120, primals_5, getitem_1, rsqrt, buf124, buf126, 86016, 192, stream=stream0)
        del buf120
        del getitem_1
        del primals_5
        del rsqrt
        buf125 = reinterpret_tensor(buf113, (64, ), (1, ), 0); del buf113  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_4.run(buf124, buf125, 64, 1344, stream=stream0)
        del buf124
        buf128 = reinterpret_tensor(buf115, (64, ), (1, ), 0); del buf115  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.add]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_native_layer_norm_backward_50.run(buf128, buf126, 64, 1344, stream=stream0)
        del buf126
        buf130 = reinterpret_tensor(buf71, (128, 21, 12, 16), (4032, 192, 16, 1), 0); del buf71  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.div, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_mul_51.run(buf97, primals_4, buf130, 516096, stream=stream0)
        del buf97
        del primals_4
    return (None, None, None, None, reinterpret_tensor(buf129, (2688, 64, 96), (6144, 1, 64), 0), None, buf107, None, buf102, None, None, None, None, None, buf130, buf121, buf125, buf128, buf109, buf111, reinterpret_tensor(buf100, (64, ), (1, ), 0), buf92, reinterpret_tensor(buf94, (64, ), (1, ), 0), None, buf89, None, buf73, None, buf87, buf88, buf78, reinterpret_tensor(buf80, (64, ), (1, ), 0), buf76, None, buf70, buf69, buf55, None, buf60, reinterpret_tensor(buf62, (64, ), (1, ), 0), None, buf52, None, buf39, None, buf45, reinterpret_tensor(buf43, (64, ), (1, ), 0), buf34, reinterpret_tensor(buf36, (64, ), (1, ), 0), buf29, reinterpret_tensor(buf31, (64, ), (1, ), 0), buf14, buf16, buf7, reinterpret_tensor(buf9, (64, ), (1, ), 0), buf2, reinterpret_tensor(buf4, (1, ), (1, ), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_2 = rand_strided((128, 21, 12, 16), (4032, 192, 16, 1), device='cuda:0', dtype=torch.bool)
    primals_3 = rand_strided((128, 21, 12, 16), (4032, 192, 16, 1), device='cuda:0', dtype=torch.bool)
    primals_4 = rand_strided((128, 21), (21, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((2688, 64, 96), (6144, 96, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((128, 21, 12, 64), (16128, 768, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((128, 21, 12, 4), (1008, 48, 4, 1), device='cuda:0', dtype=torch.bool)
    primals_9 = rand_strided((128, 21, 12, 16, 4), (16128, 768, 64, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((128, 21, 12), (252, 12, 1), device='cuda:0', dtype=torch.bool)
    primals_28 = rand_strided((128, 21, 12, 4), (1008, 48, 4, 1), device='cuda:0', dtype=torch.bool)
    primals_29 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((128, 21, 12), (252, 12, 1), device='cuda:0', dtype=torch.bool)
    primals_38 = rand_strided((128, 21, 12, 12), (3024, 144, 12, 1), device='cuda:0', dtype=torch.bool)
    primals_41 = rand_strided((128, 21, 12), (252, 12, 1), device='cuda:0', dtype=torch.bool)
    primals_45 = rand_strided((128, 21, 12, 10), (2520, 120, 1, 12), device='cuda:0', dtype=torch.bool)
    primals_52 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_1 = rand_strided((2688, 96, 1), (96, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt = rand_strided((2688, 96, 1), (96, 1, 1), device='cuda:0', dtype=torch.float32)
    getitem_3 = rand_strided((128, 21, 12, 1), (252, 12, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_1 = rand_strided((128, 21, 12, 1), (252, 12, 1, 1), device='cuda:0', dtype=torch.float32)
    view_14 = rand_strided((516096, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    mul_11 = rand_strided((128, 21, 12, 4, 64), (64512, 3072, 256, 64, 1), device='cuda:0', dtype=torch.float32)
    view_16 = rand_strided((129024, 5), (5, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_27 = rand_strided((128, 21, 12, 1, 4), (1008, 48, 4, 4, 1), device='cuda:0', dtype=torch.float32)
    mul_14 = rand_strided((128, 21, 12, 12, 64), (193536, 9216, 768, 64, 1), device='cuda:0', dtype=torch.float32)
    view_18 = rand_strided((387072, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_33 = rand_strided((128, 21, 12, 1, 12), (3040, 144, 12, 0, 1), device='cuda:0', dtype=torch.float32)
    view_20 = rand_strided((322560, 5), (5, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_39 = rand_strided((128, 21, 12, 1, 10), (2528, 120, 1, 120, 12), device='cuda:0', dtype=torch.float32)
    cat_4 = rand_strided((128, 21, 12, 26), (6552, 312, 26, 1), device='cuda:0', dtype=torch.bool)
    view_23 = rand_strided((838656, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    cat_5 = rand_strided((128, 21, 12, 16, 26), (104832, 4992, 416, 26, 1), device='cuda:0', dtype=torch.float32)
    div_3 = rand_strided((128, 21, 12, 8, 16, 27), (870912, 41472, 3456, 432, 27, 1), device='cuda:0', dtype=torch.float32)
    cat_7 = rand_strided((128, 21, 12, 16, 128), (516096, 24576, 2048, 128, 1), device='cuda:0', dtype=torch.float32)
    getitem_11 = rand_strided((128, 21, 12, 16, 1), (4032, 192, 16, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_5 = rand_strided((128, 21, 12, 16, 1), (4032, 192, 16, 1, 1), device='cuda:0', dtype=torch.float32)
    view_36 = rand_strided((516096, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_7 = rand_strided((516096, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    view_38 = rand_strided((516096, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_44 = rand_strided((128, 21, 12, 1), (252, 12, 1, 1), device='cuda:0', dtype=torch.bool)
    permute_25 = rand_strided((1, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    permute_29 = rand_strided((64, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_35 = rand_strided((258048, 8, 26), (208, 1, 8), device='cuda:0', dtype=torch.float32)
    permute_36 = rand_strided((258048, 8, 16), (128, 1, 8), device='cuda:0', dtype=torch.float32)
    permute_37 = rand_strided((258048, 26, 8), (208, 1, 26), device='cuda:0', dtype=torch.float32)
    permute_40 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    permute_45 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    mul_42 = rand_strided((128, 21, 12, 10, 64), (161280, 7680, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    div_8 = rand_strided((128, 21, 12, 10, 1), (2528, 120, 10, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_53 = rand_strided((64, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    div_9 = rand_strided((128, 21, 12, 12, 1), (3040, 144, 12, 1, 1), device='cuda:0', dtype=torch.float32)
    div_10 = rand_strided((128, 21, 12, 4, 1), (1008, 48, 4, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_60 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    add_43 = rand_strided((128, 21, 12, 16, 64), (258048, 12288, 1024, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_64 = rand_strided((64, 15), (143, 1), device='cuda:0', dtype=torch.float32)
    permute_65 = rand_strided((15, 516096), (1, 15), device='cuda:0', dtype=torch.float32)
    permute_67 = rand_strided((64, 32256), (1, 64), device='cuda:0', dtype=torch.float32)
    permute_68 = rand_strided((64, 64), (143, 1), device='cuda:0', dtype=torch.float32)
    permute_70 = rand_strided((64, 1), (1, 64), device='cuda:0', dtype=torch.float32)
    permute_71 = rand_strided((64, 64), (143, 1), device='cuda:0', dtype=torch.float32)
    permute_76 = rand_strided((64, 258048), (1, 64), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((128, 21, 12, 16), (4032, 192, 16, 1), device='cuda:0', dtype=torch.float32)
    tangents_2 = rand_strided((128, 21, 12, 8, 16, 27), (870912, 41472, 3456, 432, 27, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_2, primals_3, primals_4, primals_5, primals_7, primals_8, primals_9, primals_17, primals_19, primals_24, primals_28, primals_29, primals_34, primals_38, primals_41, primals_45, primals_52, getitem_1, rsqrt, getitem_3, rsqrt_1, view_14, mul_11, view_16, unsqueeze_27, mul_14, view_18, unsqueeze_33, view_20, unsqueeze_39, cat_4, view_23, cat_5, div_3, cat_7, getitem_11, rsqrt_5, view_36, addmm_7, view_38, unsqueeze_44, permute_25, permute_29, permute_35, permute_36, permute_37, permute_40, permute_45, mul_42, div_8, permute_53, div_9, div_10, permute_60, add_43, permute_64, permute_65, permute_67, permute_68, permute_70, permute_71, permute_76, tangents_1, tangents_2])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
