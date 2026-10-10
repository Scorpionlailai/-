# AOT ID: ['2_backward']
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


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/7l/c7lbgs2mk3oal3m4okd7ceeivqtgifh7gfzlmqerk3ks3c4mj2cb.py
# Topologically Sorted Source Nodes: [invert_5, and__6, eligible, correction_1], Original ATen: [aten._to_copy, aten.bitwise_not, aten.bitwise_and, aten.scalar_tensor, aten.where, aten.sum]
# Source node to ATen node mapping:
#   and__6 => bitwise_and_6
#   correction_1 => full_default_12
#   eligible => bitwise_and_7
#   invert_5 => bitwise_not_5
# Graph fragment:
#   %convert_element_type_75 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%tangents_1, torch.bfloat16), kwargs = {})
#   %bitwise_not_5 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_not.default](args = (%primals_2,), kwargs = {})
#   %bitwise_and_6 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%primals_3, %bitwise_not_5), kwargs = {})
#   %bitwise_and_7 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and_6, %unsqueeze_41), kwargs = {})
#   %full_default_12 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_16 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and_7, %convert_element_type_75, %full_default_12), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_35, [0], True), kwargs = {})
triton_red_fused__to_copy_bitwise_and_bitwise_not_scalar_tensor_sum_where_0 = async_compile.triton('triton_red_fused__to_copy_bitwise_and_bitwise_not_scalar_tensor_sum_where_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*i1', 'in_ptr2': '*i1', 'in_ptr3': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_bitwise_and_bitwise_not_scalar_tensor_sum_where_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_bitwise_and_bitwise_not_scalar_tensor_sum_where_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 21
    r0_numel = 8192
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp11 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index
        r0_2 = r0_index // 16
        tmp0 = tl.load(in_ptr0 + (r0_3 + 8192*x0), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (r0_3 + 8192*x0), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.int1)
        tmp4 = tl.load(in_ptr2 + (r0_2 + 512*x0), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.int1)
        tmp6 = tl.load(in_ptr3 + (r0_3 + 8192*x0), xmask & r0_mask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp1 == 0
        tmp3 = tmp0 & tmp2
        tmp5 = tmp3 & tmp4
        tmp7 = tmp6.to(tl.float32)
        tmp8 = 0.0
        tmp9 = tl.where(tmp5, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(r0_mask & xmask, tmp12, _tmp11)
        tl.store(out_ptr0 + (r0_3 + 8192*x0), tmp9, xmask & r0_mask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp11, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/yb/cybx2im4nryzr6o3jsxr6pbigfmw27txweopxmmssi5xs5fgjdjp.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum, aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_35, [0], True), kwargs = {})
#   %convert_element_type_81 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_36, torch.float32), kwargs = {})
triton_per_fused__to_copy_sum_1 = async_compile.triton('triton_per_fused__to_copy_sum_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 32},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_sum_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_sum_1(in_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 21
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
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = tmp4.to(tl.float32)
    tl.store(out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp5, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/p3/cp3labasaxvaywbah26ltb2sdw2sbwmr2cy2ww6656p7rwuuyekf.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_80 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_3, torch.float32), kwargs = {})
triton_poi_fused__to_copy_2 = async_compile.triton('triton_poi_fused__to_copy_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/zg/czgjvjuspw25cncfkktprd6xxduly2dfzgtjxz2ij5mvz3d5vzb2.py
# Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.gelu_backward, aten.gelu]
# Source node to ATen node mapping:
#   input_6 => add_25, convert_element_type_67, erf_1, mul_23
# Graph fragment:
#   %convert_element_type_82 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_37, torch.float32), kwargs = {})
#   %convert_element_type_67 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_32, torch.float32), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_67, 0.7071067811865476), kwargs = {})
#   %erf_1 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_23,), kwargs = {})
#   %add_25 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_1, 1), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_25, 0.5), kwargs = {})
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_67, %convert_element_type_67), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_27, -0.5), kwargs = {})
#   %exp_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_28,), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp_1, 0.3989422804014327), kwargs = {})
#   %mul_30 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_67, %mul_29), kwargs = {})
#   %add_27 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %mul_30), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_82, %add_27), kwargs = {})
#   %convert_element_type_84 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_31, torch.bfloat16), kwargs = {})
triton_poi_fused_gelu_gelu_backward_3 = async_compile.triton('triton_poi_fused_gelu_gelu_backward_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_gelu_backward_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 11010048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 0.7071067811865476
    tmp5 = tmp3 * tmp4
    tmp6 = libdevice.erf(tmp5)
    tmp7 = 1.0
    tmp8 = tmp6 + tmp7
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = tmp3 * tmp3
    tmp12 = -0.5
    tmp13 = tmp11 * tmp12
    tmp14 = tl_math.exp(tmp13)
    tmp15 = 0.3989422804014327
    tmp16 = tmp14 * tmp15
    tmp17 = tmp3 * tmp16
    tmp18 = tmp10 + tmp17
    tmp19 = tmp1 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tl.store(in_out_ptr0 + (x0), tmp20, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/6p/c6ppyljpq6cgvqc2of6qrikaksaxnjj356u3cog5pnogfm2uljhh.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_38, [0], True), kwargs = {})
triton_red_fused_sum_4 = async_compile.triton('triton_red_fused_sum_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_sum_4(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 86016
    r0_numel = 128
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
        tmp0 = tl.load(in_ptr0 + (x0 + 64*r0_2 + 8192*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/i7/ci7jk46trwl75s44fn3hy3vevkicydllbofuwaegognhmfmfp4sy.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum, aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_38, [0], True), kwargs = {})
#   %convert_element_type_91 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_39, torch.float32), kwargs = {})
triton_red_fused__to_copy_sum_5 = async_compile.triton('triton_red_fused__to_copy_sum_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_sum_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_sum_5(in_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
        tmp0 = tl.load(in_ptr0 + (x0 + 64*r0_1), xmask & r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp4 = tmp2.to(tl.float32)
    tl.store(out_ptr1 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/az/cazmsozrksfcrumm25lycthhadlz4glohkb26onu43acnly5b4c7.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_90 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_5, torch.float32), kwargs = {})
triton_poi_fused__to_copy_6 = async_compile.triton('triton_poi_fused__to_copy_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/7p/c7prysknfe35i3ml4yalzpr5nmtxpx4c6jjwwdvtreijjfnym6qz.py
# Topologically Sorted Source Nodes: [input_4], Original ATen: [aten._to_copy, aten.native_layer_norm_backward, aten.native_layer_norm]
# Source node to ATen node mapping:
#   input_4 => convert_element_type_60, mul_20, sub_8
# Graph fragment:
#   %convert_element_type_89 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_40, torch.float32), kwargs = {})
#   %mul_33 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_89, %primals_52), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_33, 128), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_33, [4], True), kwargs = {})
#   %convert_element_type_60 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cat_7, torch.float32), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_60, %getitem_11), kwargs = {})
#   %mul_20 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %rsqrt_5), kwargs = {})
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_33, %mul_20), kwargs = {})
#   %sum_5 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_35, [4], True), kwargs = {})
#   %mul_36 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_20, %sum_5), kwargs = {})
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_34, %sum_4), kwargs = {})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_10, %mul_36), kwargs = {})
#   %div_4 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_5, 128), kwargs = {})
#   %mul_37 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_4, %sub_11), kwargs = {})
#   %convert_element_type_92 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_37, torch.bfloat16), kwargs = {})
triton_per_fused__to_copy_native_layer_norm_native_layer_norm_backward_7 = async_compile.triton('triton_per_fused__to_copy_native_layer_norm_native_layer_norm_backward_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 262144, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_native_layer_norm_native_layer_norm_backward_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_native_layer_norm_native_layer_norm_backward_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 172032
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 128*x0), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (r0_1 + 128*x0), None).to(tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp6 = tl.sum(tmp4, 1)[:, None]
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 - tmp9
    tmp12 = tmp10 * tmp11
    tmp13 = tmp3 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp16 = tl.sum(tmp14, 1)[:, None]
    tmp17 = 0.0078125
    tmp18 = tmp11 * tmp17
    tmp19 = 128.0
    tmp20 = tmp3 * tmp19
    tmp21 = tmp20 - tmp6
    tmp22 = tmp12 * tmp16
    tmp23 = tmp21 - tmp22
    tmp24 = tmp18 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(out_ptr2 + (r0_1 + 128*x0), tmp25, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/pc/cpc4hqdgqkxoxfafhbfjxc7vmovlqo2bxhowzxtwvmhhmc43gmxr.py
# Topologically Sorted Source Nodes: [input_4], Original ATen: [aten._to_copy, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   input_4 => convert_element_type_60, mul_20, sub_8
# Graph fragment:
#   %convert_element_type_89 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_40, torch.float32), kwargs = {})
#   %convert_element_type_60 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cat_7, torch.float32), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_60, %getitem_11), kwargs = {})
#   %mul_20 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %rsqrt_5), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_89, %mul_20), kwargs = {})
#   %sum_6 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_38, [0, 1, 2, 3]), kwargs = {})
#   %sum_7 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_89, [0, 1, 2, 3]), kwargs = {})
triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_8 = async_compile.triton('triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 262144, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 172032
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 128)
    x1 = xindex // 128
    _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    _tmp13 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_2 + 16384*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (x0 + 128*r0_2 + 16384*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0_2 + 128*x1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr3 + (r0_2 + 128*x1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp3 - tmp4
        tmp7 = tmp5 * tmp6
        tmp8 = tmp1 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(r0_mask, tmp11, _tmp10)
        tmp12 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(r0_mask, tmp14, _tmp13)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, None)
    tl.store(out_ptr1 + (x3), tmp13, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/3s/c3syx6squb5p64bdcuje23zfgmf7oq7yy4dijc7zo4txflbdeuni.py
# Topologically Sorted Source Nodes: [input_4], Original ATen: [aten._to_copy, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   input_4 => convert_element_type_60, mul_20, sub_8
# Graph fragment:
#   %convert_element_type_89 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_40, torch.float32), kwargs = {})
#   %convert_element_type_60 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cat_7, torch.float32), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_60, %getitem_11), kwargs = {})
#   %mul_20 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %rsqrt_5), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_89, %mul_20), kwargs = {})
#   %sum_6 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_38, [0, 1, 2, 3]), kwargs = {})
triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_9 = async_compile.triton('triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_9', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_9(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
        tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_1), xmask & r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/kp/ckpltoggemnw5fycb3w2vcxwi4u3nlomfurecm42hs5etn4gtmto.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
# Source node to ATen node mapping:
# Graph fragment:
#   %clone_9 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_27,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_10 = async_compile.triton('triton_poi_fused_clone_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_10(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 11010048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 8)
    x1 = ((xindex // 8) % 16)
    x2 = ((xindex // 128) % 8)
    x3 = xindex // 1024
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (64 + x0 + 8*x2 + 128*x1 + 2048*x3), None).to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/5l/c5lfaqb77mxievgwddeit2m3xqeneklk27xtzsvyput42of4tyiv.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.slice_backward, aten.add, aten._softmax_backward_data]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_97 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_44, torch.float32), kwargs = {})
#   %full_default_14 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([128, 7, 12, 8, 16, 27], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_scatter_default : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_14, %convert_element_type_97, 5, 1, 9223372036854775807), kwargs = {})
#   %add_28 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tangents_2, %slice_scatter_default), kwargs = {})
#   %mul_39 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_28, %div_3), kwargs = {})
#   %sum_8 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_39, [-1], True), kwargs = {})
triton_per_fused__softmax_backward_data__to_copy_add_slice_backward_11 = async_compile.triton('triton_per_fused__softmax_backward_data__to_copy_add_slice_backward_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2097152, 'r0_': 32},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data__to_copy_add_slice_backward_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax_backward_data__to_copy_add_slice_backward_11(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1376256
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
    tmp11 = tl.load(in_ptr2 + (r0_1 + 27*x0), r0_mask, other=0.0)
    tmp1 = r0_1
    tmp2 = tl.full([1, 1], 1, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.load(in_ptr1 + ((-1) + r0_1 + 26*x0), r0_mask & tmp3, other=0.0).to(tl.float32)
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp3, tmp5, tmp6)
    tmp8 = 0.0
    tmp9 = tl.where(tmp3, tmp7, tmp8)
    tmp10 = tmp0 + tmp9
    tmp12 = tmp10 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp15 = tl.where(r0_mask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp16, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/5m/c5mkl2vtrvat4zd6sxmlutwmlikoewbkzdz2zim32isoewraanr7.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_9 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%slice_16, [3], True), kwargs = {})
triton_per_fused_sum_12 = async_compile.triton('triton_per_fused_sum_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 262144, 'r0_': 8},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_sum_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 172032
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
    tmp7 = tl.load(in_ptr3 + ((-1) + 26*x0 + 416*r0_2 + 3328*x1), tmp6, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp6, tmp8, tmp9)
    tmp11 = 0.0
    tmp12 = tl.where(tmp6, tmp10, tmp11)
    tmp13 = tmp3 + tmp12
    tmp14 = tmp13 * tmp0
    tmp15 = libdevice.fma(tmp1, tmp2, tmp14)
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, R0_BLOCK])
    tmp18 = tl.sum(tmp16, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp18, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/e4/ce4euac25pw7dczutkm3e4enxe5h3se4nbyboejvxqsmz5ome5oi.py
# Topologically Sorted Source Nodes: [context, invert_4], Original ATen: [aten.masked_fill, aten.bitwise_not, aten._to_copy, aten.div]
# Source node to ATen node mapping:
#   context => full_default
#   invert_4 => bitwise_not_4
# Graph fragment:
#   %full_default : [num_users=12] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %bitwise_not_4 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_not.default](args = (%unsqueeze_39,), kwargs = {})
#   %where_18 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_not_4, %full_default, %slice_17), kwargs = {})
#   %convert_element_type_98 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_18, torch.bfloat16), kwargs = {})
#   %div_7 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_98, 2.8284271247461903), kwargs = {})
triton_poi_fused__to_copy_bitwise_not_div_masked_fill_13 = async_compile.triton('triton_poi_fused__to_copy_bitwise_not_div_masked_fill_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_bitwise_not_div_masked_fill_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_bitwise_not_div_masked_fill_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 35782656
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
    tmp9 = tl.load(in_ptr4 + (x3), tmp8, other=0.0).to(tl.float32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp8, tmp10, tmp11)
    tmp13 = 0.0
    tmp14 = tl.where(tmp8, tmp12, tmp13)
    tmp15 = tmp5 + tmp14
    tmp16 = tmp15 * tmp2
    tmp17 = libdevice.fma(tmp3, tmp4, tmp16)
    tmp18 = tl.where(tmp1, tmp13, tmp17)
    tmp19 = tmp18.to(tl.float32)
    tmp20 = 0.35355339059327373
    tmp21 = tmp19 * tmp20
    tl.store(out_ptr0 + (x3), tmp18, None)
    tl.store(out_ptr1 + (x3), tmp21, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/rl/crldkn23bf7ceojmyut4pnqinr4v43zgydxlwynixcdc4o4teaox.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_10 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%where_18, [3], True), kwargs = {})
triton_per_fused_sum_14 = async_compile.triton('triton_per_fused_sum_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8388608, 'r0_': 8},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_sum_14(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 4472832
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


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/6b/c6bakzgkhs7ruo75zpwso42rbyddb3n4c3stw2nd32jprp7xq75b.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.view]
# Source node to ATen node mapping:
# Graph fragment:
#   %view_49 : [num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%view_48, [279552, 64]), kwargs = {})
triton_poi_fused_view_15 = async_compile.triton('triton_poi_fused_view_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_view_15(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 17891328
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 64)
    x1 = xindex // 64
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (8*((x1 % 26)) + 208*(x0 // 8) + 1664*(x1 // 26) + ((x0 % 8))), None).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp0, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/i3/ci3yvocpzz4soa4pitmps2mrmo6nxvcxgztmpmvujou3ptlthagd.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_11 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_49, [0], True), kwargs = {})
triton_red_fused_sum_16 = async_compile.triton('triton_red_fused_sum_16', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_sum_16(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 86016
    r0_numel = 208
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
        tmp0 = tl.load(in_ptr0 + (x0 + 64*r0_2 + 13312*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/5d/c5dma5fv3f2ao56bltka356qpkprtne7bgsrs576ilxfw7qs2cqw.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_108 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_7, torch.float32), kwargs = {})
triton_poi_fused__to_copy_17 = async_compile.triton('triton_poi_fused__to_copy_17', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_17(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/rv/crvez34i4ogfiwrjvbt3j2mplz5ke7snhrjwmeefwgukddecv2ej.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.clone, aten._unsafe_view]
# Source node to ATen node mapping:
# Graph fragment:
#   %clone_11 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%view_52,), kwargs = {memory_format: torch.contiguous_format})
#   %view_53 : [num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_11, [279552, 64]), kwargs = {})
triton_poi_fused__unsafe_view_clone_18 = async_compile.triton('triton_poi_fused__unsafe_view_clone_18', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 524288, 'x': 64}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_clone_18(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 279552
    xnumel = 64
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (26*x1 + 1664*(y0 // 26) + ((y0 % 26))), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (x1 + 64*y0), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/4f/c4fgc6n5gnxwbqfwnyibqs4zw2rp3b4rrbbudg4c2qjg5g43aahg.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_41 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_20, %unsqueeze_36), kwargs = {})
#   %sum_13 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_41, [4], True), kwargs = {})
triton_per_fused_mul_sum_19 = async_compile.triton('triton_per_fused_mul_sum_19', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 262144, 'r0_': 16},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sum_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mul_sum_19(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 172032
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
    r0_3 = r0_index
    x4 = xindex
    x1 = ((xindex // 16) % 12)
    x2 = xindex // 192
    tmp0 = tl.load(in_ptr0 + (16 + r0_3 + 26*x4), r0_mask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (16 + r0_3 + 26*x4), r0_mask, other=0.0)
    tmp8 = tl.load(in_ptr2 + (x1 + 12*r0_3 + 120*x2), r0_mask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/zy/czymp4connj7vl632bv43tt4gwdvcn7ohtfkx4znadz5q7bpojhz.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_40 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_20, %unsqueeze_35), kwargs = {})
#   %sum_14 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_40, [3], True), kwargs = {})
triton_per_fused_mul_sum_20 = async_compile.triton('triton_per_fused_mul_sum_20', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sum_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mul_sum_20(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 107520
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
    x0 = (xindex % 10)
    x1 = xindex // 10
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (16 + x0 + 26*r0_2 + 416*x1), xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (16 + x0 + 26*r0_2 + 416*x1), xmask, other=0.0)
    tmp8 = tl.load(in_ptr2 + (3 + 4*r0_2 + 64*x1), xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x3), tmp13, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/po/cpondtxb6qbe4r7x25wvhwa37b77d7t72sgbq5k5dngdsbb5ejqo.py
# Topologically Sorted Source Nodes: [context, and__4, valid_2], Original ATen: [aten.masked_fill, aten.bitwise_and, aten.where]
# Source node to ATen node mapping:
#   and__4 => bitwise_and_4
#   context => full_default
#   valid_2 => bitwise_and_5
# Graph fragment:
#   %full_default : [num_users=12] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %bitwise_and_4 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%primals_45, %unsqueeze_31), kwargs = {})
#   %bitwise_and_5 : [num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and_4, %unsqueeze_32), kwargs = {})
#   %where_20 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and_5, %squeeze_3, %full_default), kwargs = {})
triton_poi_fused_bitwise_and_masked_fill_where_21 = async_compile.triton('triton_poi_fused_bitwise_and_masked_fill_where_21', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*i1', 'in_ptr2': '*i1', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_and_masked_fill_where_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_bitwise_and_masked_fill_where_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8960
    xnumel = 12
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y1 = yindex // 10
    y0 = (yindex % 10)
    tmp0 = tl.load(in_ptr0 + (x2 + 12*y3), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x2 + 12*y1), ymask & xmask, eviction_policy='evict_last').to(tl.int1)
    tmp3 = tl.load(in_ptr2 + (3 + 4*x2 + 48*y1), ymask & xmask, eviction_policy='evict_last').to(tl.int1)
    tmp5 = tl.load(in_ptr3 + (y0 + 10*x2 + 120*y1), ymask & xmask, eviction_policy='evict_last')
    tmp2 = tmp0 & tmp1
    tmp4 = tmp2 & tmp3
    tmp6 = 0.0
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tl.store(out_ptr0 + (x2 + 12*y3), tmp7, xmask & ymask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/ts/ctsrb7qwgu7ysr2h5sbgouvs4vhcqfcjk4byhgdhsxiyujq63brv.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum, aten._to_copy, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_15 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%slice_23, [0, 1, 2, 3], True), kwargs = {})
#   %convert_element_type_117 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_23, torch.bfloat16), kwargs = {})
#   %sum_16 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_117, [0, 1, 2, 3], True), kwargs = {})
#   %mul_48 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_23, %mul_42), kwargs = {})
#   %sum_19 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_48, [0, 1, 2, 3]), kwargs = {})
#   %sum_20 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%slice_23, [0, 1, 2, 3]), kwargs = {})
triton_red_fused__to_copy_native_layer_norm_backward_sum_22 = async_compile.triton('triton_red_fused__to_copy_native_layer_norm_backward_sum_22', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_native_layer_norm_backward_sum_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 4, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_native_layer_norm_backward_sum_22(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 53760
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
    _tmp6 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp15 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (1024 + x0 + 64*(((r0_2 + 128*x1) % 10)) + 1664*((r0_2 + 128*x1) // 10)), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (1024 + x0 + 64*(((r0_2 + 128*x1) % 10)) + 1664*((r0_2 + 128*x1) // 10)), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp12 = tl.load(in_ptr2 + (x0 + 64*((((r0_2 + 128*x1) // 10) % 12)) + 768*(((r0_2 + 128*x1) % 10)) + 7680*((r0_2 + 128*x1) // 120)), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(r0_mask & xmask, tmp7, _tmp6)
        tmp8 = tmp4.to(tl.float32)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(r0_mask & xmask, tmp11, _tmp10)
        tmp13 = tmp4 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(r0_mask & xmask, tmp16, _tmp15)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
    tl.store(out_ptr1 + (x3), tmp10, xmask)
    tl.store(out_ptr2 + (x3), tmp15, xmask)
    tl.store(out_ptr3 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/k2/ck24wqachlyf7wlqobmzcnlrqgm6rlexz7b3sfspdtcqas6i4wmw.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_15 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%slice_23, [0, 1, 2, 3], True), kwargs = {})
triton_red_fused_sum_23 = async_compile.triton('triton_red_fused_sum_23', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_sum_23(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 64
    r0_numel = 840
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
        tmp0 = tl.load(in_ptr0 + (x0 + 64*r0_1), xmask & r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/on/con3v46dwly2nxfmdrpvhi5cfqljvg2nemggb5jzn3dfsufhbrt5.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_117 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_23, torch.bfloat16), kwargs = {})
#   %sum_16 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_117, [0, 1, 2, 3], True), kwargs = {})
#   %convert_element_type_121 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_58, torch.float32), kwargs = {})
triton_red_fused__to_copy_sum_24 = async_compile.triton('triton_red_fused__to_copy_sum_24', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_sum_24', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_sum_24(in_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 64
    r0_numel = 840
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
        tmp0 = tl.load(in_ptr0 + (x0 + 64*r0_1), xmask & r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp4 = tmp2.to(tl.float32)
    tl.store(out_ptr1 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/ug/cugkywqsyhmhcs7d2wje3xeuespgcucysycphmrbw7ivwdrkbvjj.py
# Topologically Sorted Source Nodes: [context], Original ATen: [aten.masked_fill, aten._to_copy, aten.native_layer_norm_backward, aten.where]
# Source node to ATen node mapping:
#   context => full_default
# Graph fragment:
#   %full_default : [num_users=12] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %convert_element_type_117 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_23, torch.bfloat16), kwargs = {})
#   %mul_43 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_23, %primals_29), kwargs = {})
#   %mul_44 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_43, 64), kwargs = {})
#   %sum_17 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_43, [4], True), kwargs = {})
#   %mul_45 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_43, %mul_42), kwargs = {})
#   %sum_18 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_45, [4], True), kwargs = {})
#   %mul_46 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_42, %sum_18), kwargs = {})
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_44, %sum_17), kwargs = {})
#   %sub_14 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_13, %mul_46), kwargs = {})
#   %mul_47 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_8, %sub_14), kwargs = {})
#   %where_21 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_33, %mul_47, %full_default), kwargs = {})
triton_per_fused__to_copy_masked_fill_native_layer_norm_backward_where_25 = async_compile.triton('triton_per_fused__to_copy_masked_fill_native_layer_norm_backward_where_25', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*i1', 'in_ptr4': '*i1', 'in_ptr5': '*i1', 'in_ptr6': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_masked_fill_native_layer_norm_backward_where_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 2, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_masked_fill_native_layer_norm_backward_where_25(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 107520
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
    x0 = (xindex % 10)
    x1 = xindex // 10
    x5 = xindex
    x3 = ((xindex // 10) % 12)
    x4 = xindex // 120
    tmp0 = tl.load(in_ptr0 + (1024 + r0_2 + 64*x0 + 1664*x1), xmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (1024 + r0_2 + 64*x0 + 1664*x1), xmask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (r0_2), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_out_ptr0 + (r0_2 + 64*x3 + 768*x0 + 7680*x4), xmask, other=0.0)
    tmp18 = tl.load(in_ptr3 + (x3 + 12*x0 + 120*x4), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp19 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp21 = tl.load(in_ptr5 + (3 + 4*x1), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp23 = tl.load(in_ptr6 + (x5), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 + tmp3
    tmp6 = tmp4 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp12 = tmp6 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp15 = tl.where(xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tmp4.to(tl.float32)
    tmp20 = tmp18 & tmp19
    tmp22 = tmp20 & tmp21
    tmp24 = 64.0
    tmp25 = tmp6 * tmp24
    tmp26 = tmp25 - tmp10
    tmp27 = tmp11 * tmp16
    tmp28 = tmp26 - tmp27
    tmp29 = tmp23 * tmp28
    tmp30 = 0.0
    tmp31 = tl.where(tmp22, tmp29, tmp30)
    tl.store(out_ptr2 + (r0_2 + 64*x5), tmp17, xmask)
    tl.store(in_out_ptr0 + (r0_2 + 64*x3 + 768*x0 + 7680*x4), tmp31, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/c7/cc7oxgecpiy442a47br3dcepomanuezhsrzmz2e6wxje24oycvaa.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_120 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_10, torch.float32), kwargs = {})
triton_poi_fused__to_copy_26 = async_compile.triton('triton_poi_fused__to_copy_26', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_26(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/4u/c4uprxn77avvrtpxjphvop5dl3sw5migcwd7va4qgrlba3scuq3a.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_50 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_19, %unsqueeze_30), kwargs = {})
#   %sum_21 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_50, [4], True), kwargs = {})
triton_per_fused_mul_sum_27 = async_compile.triton('triton_per_fused_mul_sum_27', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 262144, 'r0_': 16},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sum_27', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mul_sum_27(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 172032
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
    r0_2 = r0_index
    x3 = xindex
    x1 = xindex // 16
    tmp0 = tl.load(in_ptr0 + (4 + r0_2 + 26*x3), r0_mask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (4 + r0_2 + 26*x3), r0_mask, other=0.0)
    tmp8 = tl.load(in_ptr2 + (r0_2 + 12*x1), r0_mask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x3), tmp13, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/uw/cuwjzydpmnj44n4rs2vszyzberojrvt7ays4us2i4er6jnwagyro.py
# Topologically Sorted Source Nodes: [context, and__2, valid_1], Original ATen: [aten.masked_fill, aten.mul, aten.sum, aten.bitwise_and, aten.where]
# Source node to ATen node mapping:
#   and__2 => bitwise_and_2
#   context => full_default
#   valid_1 => bitwise_and_3
# Graph fragment:
#   %full_default : [num_users=12] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_49 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_19, %unsqueeze_29), kwargs = {})
#   %sum_22 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_49, [3], True), kwargs = {})
#   %bitwise_and_2 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%primals_38, %unsqueeze_25), kwargs = {})
#   %bitwise_and_3 : [num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and_2, %unsqueeze_26), kwargs = {})
#   %where_22 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and_3, %squeeze_5, %full_default), kwargs = {})
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
    x0 = (xindex % 12)
    x1 = xindex // 12
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (4 + x0 + 26*r0_2 + 416*x1), xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (4 + x0 + 26*r0_2 + 416*x1), xmask, other=0.0)
    tmp8 = tl.load(in_ptr2 + (2 + 4*r0_2 + 64*x1), xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp15 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp17 = tl.load(in_ptr5 + (2 + 4*x1), xmask, eviction_policy='evict_last').to(tl.int1)
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


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/x3/cx3xgpu6fkk4zaflpvy6of5lssgsfxqc6xd7d3mglaer7afpwcb4.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_23 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%slice_22, [0, 1, 2, 3], True), kwargs = {})
#   %mul_57 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_22, %mul_14), kwargs = {})
#   %sum_27 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_57, [0, 1, 2, 3]), kwargs = {})
#   %sum_28 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%slice_22, [0, 1, 2, 3]), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
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
    _tmp6 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    _tmp11 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (256 + x0 + 64*(((r0_2 + 128*x1) % 12)) + 1664*((r0_2 + 128*x1) // 12)), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (256 + x0 + 64*(((r0_2 + 128*x1) % 12)) + 1664*((r0_2 + 128*x1) // 12)), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp8 = tl.load(in_ptr2 + (x0 + 64*r0_2 + 8192*x1), xmask & r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(r0_mask & xmask, tmp7, _tmp6)
        tmp9 = tmp4 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(r0_mask & xmask, tmp12, _tmp11)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
    tl.store(out_ptr1 + (x3), tmp11, xmask)
    tl.store(out_ptr2 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/hz/chzr2xb4edl5b7uol2a6a76kckxv6nyusp5hua4qrqgf77jzkvbl.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_23 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%slice_22, [0, 1, 2, 3], True), kwargs = {})
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
        tmp0 = tl.load(in_ptr0 + (x0 + 64*r0_1), xmask & r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/rx/crxtzoblimpbz5ycn6lshmeyrkgj6dcztz64mzmcgm2ueshrzge3.py
# Topologically Sorted Source Nodes: [context], Original ATen: [aten.masked_fill, aten._to_copy, aten.native_layer_norm_backward, aten.where]
# Source node to ATen node mapping:
#   context => full_default
# Graph fragment:
#   %full_default : [num_users=12] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %convert_element_type_122 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_22, torch.bfloat16), kwargs = {})
#   %mul_52 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_22, %primals_29), kwargs = {})
#   %mul_53 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_52, 64), kwargs = {})
#   %sum_25 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_52, [4], True), kwargs = {})
#   %mul_54 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_52, %mul_14), kwargs = {})
#   %sum_26 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_54, [4], True), kwargs = {})
#   %mul_55 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_14, %sum_26), kwargs = {})
#   %sub_16 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_53, %sum_25), kwargs = {})
#   %sub_17 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_16, %mul_55), kwargs = {})
#   %mul_56 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_9, %sub_17), kwargs = {})
#   %where_24 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_27, %mul_56, %full_default), kwargs = {})
triton_per_fused__to_copy_masked_fill_native_layer_norm_backward_where_31 = async_compile.triton('triton_per_fused__to_copy_masked_fill_native_layer_norm_backward_where_31', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*i1', 'in_ptr4': '*i1', 'in_ptr5': '*i1', 'in_ptr6': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_masked_fill_native_layer_norm_backward_where_31', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 2, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_masked_fill_native_layer_norm_backward_where_31(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    x0 = (xindex % 12)
    x1 = xindex // 12
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (256 + r0_2 + 64*x0 + 1664*x1), xmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (256 + r0_2 + 64*x0 + 1664*x1), xmask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (r0_2), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_out_ptr0 + (r0_2 + 64*x3), xmask, other=0.0)
    tmp18 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp19 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp21 = tl.load(in_ptr5 + (2 + 4*x1), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp23 = tl.load(in_ptr6 + (x3), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 + tmp3
    tmp6 = tmp4 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp12 = tmp6 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp15 = tl.where(xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tmp4.to(tl.float32)
    tmp20 = tmp18 & tmp19
    tmp22 = tmp20 & tmp21
    tmp24 = 64.0
    tmp25 = tmp6 * tmp24
    tmp26 = tmp25 - tmp10
    tmp27 = tmp11 * tmp16
    tmp28 = tmp26 - tmp27
    tmp29 = tmp23 * tmp28
    tmp30 = 0.0
    tmp31 = tl.where(tmp22, tmp29, tmp30)
    tl.store(out_ptr2 + (r0_2 + 64*x3), tmp17, xmask)
    tl.store(in_out_ptr0 + (r0_2 + 64*x3), tmp31, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/mv/cmviug6ynkvn26fu6l4be6hnpuzqvvj54cnk6boh2pwoljncka3n.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_24 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_61, [0], True), kwargs = {})
triton_red_fused_sum_32 = async_compile.triton('triton_red_fused_sum_32', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_32', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_sum_32(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
        tmp0 = tl.load(in_ptr0 + (x0 + 64*r0_2 + 8192*x1), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/zz/czzbyfc745geqiwgwxehce2en7ca7vb7yfdy6yyf3e7y6vwklvji.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum, aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_24 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_61, [0], True), kwargs = {})
#   %convert_element_type_129 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_62, torch.float32), kwargs = {})
triton_red_fused__to_copy_sum_33 = async_compile.triton('triton_red_fused__to_copy_sum_33', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_sum_33', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_sum_33(in_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
        tmp0 = tl.load(in_ptr0 + (x0 + 64*r0_1), xmask & r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp4 = tmp2.to(tl.float32)
    tl.store(out_ptr1 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/3j/c3jbaw46yknjbfxa2izgzln75nyynhx6sxhysr6b5devollwixpi.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_128 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_12, torch.float32), kwargs = {})
triton_poi_fused__to_copy_34 = async_compile.triton('triton_poi_fused__to_copy_34', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_34', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_34(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/fd/cfdxfefmqw27rmhuvbmcmdkwetz6ux4kwxs7hateslf2h4wpxpzj.py
# Topologically Sorted Source Nodes: [context], Original ATen: [aten.masked_fill, aten._to_copy, aten.where]
# Source node to ATen node mapping:
#   context => full_default
# Graph fragment:
#   %full_default : [num_users=12] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %convert_element_type_127 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_63, torch.float32), kwargs = {})
#   %where_23 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_27, %convert_element_type_127, %full_default), kwargs = {})
triton_poi_fused__to_copy_masked_fill_where_35 = async_compile.triton('triton_poi_fused__to_copy_masked_fill_where_35', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*i1', 'in_ptr2': '*i1', 'in_ptr3': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_masked_fill_where_35', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_masked_fill_where_35(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 516096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex // 4
    x2 = xindex // 48
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last').to(tl.int1)
    tmp3 = tl.load(in_ptr2 + (2 + 4*x2), None, eviction_policy='evict_last').to(tl.int1)
    tmp5 = tl.load(in_ptr3 + (x5), None).to(tl.float32)
    tmp2 = tmp0 & tmp1
    tmp4 = tmp2 & tmp3
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 0.0
    tmp8 = tl.where(tmp4, tmp6, tmp7)
    tl.store(out_ptr0 + (x5), tmp8, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/4e/c4ehwa5bf6hjvuzdcqkgxiirtll2hmbcgtr6pq2bx3tovx5babqi.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_59 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_18, %unsqueeze_24), kwargs = {})
#   %sum_29 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_59, [4], True), kwargs = {})
triton_poi_fused_mul_sum_36 = async_compile.triton('triton_poi_fused_mul_sum_36', '''
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
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/hx/chxavsdzm3pfrn73zxrolzlybaqmkmkcdglwpip2z55poaioifda.py
# Topologically Sorted Source Nodes: [context, and_, valid], Original ATen: [aten.masked_fill, aten.mul, aten.sum, aten.bitwise_and, aten.where]
# Source node to ATen node mapping:
#   and_ => bitwise_and
#   context => full_default
#   valid => bitwise_and_1
# Graph fragment:
#   %full_default : [num_users=12] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_58 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_18, %unsqueeze_23), kwargs = {})
#   %sum_30 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_58, [3], True), kwargs = {})
#   %bitwise_and : [num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%primals_28, %unsqueeze_19), kwargs = {})
#   %bitwise_and_1 : [num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and, %unsqueeze_20), kwargs = {})
#   %where_25 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and_1, %squeeze_7, %full_default), kwargs = {})
triton_per_fused_bitwise_and_masked_fill_mul_sum_where_37 = async_compile.triton('triton_per_fused_bitwise_and_masked_fill_mul_sum_where_37', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 65536, 'r0_': 16},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*i1', 'in_ptr4': '*i1', 'in_ptr5': '*i1', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_bitwise_and_masked_fill_mul_sum_where_37', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 1, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_bitwise_and_masked_fill_mul_sum_where_37(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 43008
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


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/di/cdiil6mkagnek2d47o72mdl7o3mgumci2nq4csqr5lsvipnu6rbk.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_31 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%slice_21, [0, 1, 2, 3], True), kwargs = {})
#   %mul_66 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_21, %mul_11), kwargs = {})
#   %sum_35 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_66, [0, 1, 2, 3]), kwargs = {})
#   %sum_36 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%slice_21, [0, 1, 2, 3]), kwargs = {})
triton_red_fused_native_layer_norm_backward_sum_38 = async_compile.triton('triton_red_fused_native_layer_norm_backward_sum_38', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32768, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_sum_38', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_layer_norm_backward_sum_38(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 21504
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
    _tmp6 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    _tmp11 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 64*((r0_2 % 4)) + 1664*(r0_2 // 4) + 53248*x1), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (x0 + 64*((r0_2 % 4)) + 1664*(r0_2 // 4) + 53248*x1), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp8 = tl.load(in_ptr2 + (x0 + 64*r0_2 + 8192*x1), xmask & r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(r0_mask & xmask, tmp7, _tmp6)
        tmp9 = tmp4 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(r0_mask & xmask, tmp12, _tmp11)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
    tl.store(out_ptr1 + (x3), tmp11, xmask)
    tl.store(out_ptr2 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/dp/cdphmqiikjkdlpuhxaogamwyw4auibl5gkhgmu43jeddbdrdpao7.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_31 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%slice_21, [0, 1, 2, 3], True), kwargs = {})
triton_red_fused_sum_39 = async_compile.triton('triton_red_fused_sum_39', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 64, 'r0_': 512},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_39', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_sum_39(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 64
    r0_numel = 336
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
        tmp0 = tl.load(in_ptr0 + (x0 + 64*r0_1), xmask & r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/2t/c2ttrnpmg4cfy67dnnujcyanbowreddnciyul32o4ez6shhxizpg.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.select_backward, aten.add]
# Source node to ATen node mapping:
# Graph fragment:
#   %full_default_26 : [num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([3, 64], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %select_scatter_default_1 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_26, %view_57, 0, 2), kwargs = {})
#   %select_scatter_default_3 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_26, %view_60, 0, 1), kwargs = {})
#   %add_33 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_scatter_default_1, %select_scatter_default_3), kwargs = {})
#   %select_scatter_default_5 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_26, %view_64, 0, 0), kwargs = {})
#   %add_37 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_33, %select_scatter_default_5), kwargs = {})
triton_poi_fused_add_select_backward_40 = async_compile.triton('triton_poi_fused_add_select_backward_40', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_select_backward_40', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_select_backward_40(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/c4/cc4c75lvanlsgq6pn4k5i3n73cipctwworgqrgdnzc7weq2rbjnl.py
# Topologically Sorted Source Nodes: [context], Original ATen: [aten.masked_fill, aten._to_copy, aten.native_layer_norm_backward, aten.where]
# Source node to ATen node mapping:
#   context => full_default
# Graph fragment:
#   %full_default : [num_users=12] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %convert_element_type_130 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_21, torch.bfloat16), kwargs = {})
#   %mul_61 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_21, %primals_29), kwargs = {})
#   %mul_62 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_61, 64), kwargs = {})
#   %sum_33 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_61, [4], True), kwargs = {})
#   %mul_63 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_61, %mul_11), kwargs = {})
#   %sum_34 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_63, [4], True), kwargs = {})
#   %mul_64 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_11, %sum_34), kwargs = {})
#   %sub_19 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_62, %sum_33), kwargs = {})
#   %sub_20 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_19, %mul_64), kwargs = {})
#   %mul_65 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_10, %sub_20), kwargs = {})
#   %where_26 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_21, %mul_65, %full_default), kwargs = {})
triton_per_fused__to_copy_masked_fill_native_layer_norm_backward_where_41 = async_compile.triton('triton_per_fused__to_copy_masked_fill_native_layer_norm_backward_where_41', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 65536, 'r0_': 64},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*i1', 'in_ptr4': '*i1', 'in_ptr5': '*i1', 'in_ptr6': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_masked_fill_native_layer_norm_backward_where_41', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 2, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_masked_fill_native_layer_norm_backward_where_41(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 43008
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
    tmp0 = tl.load(in_ptr0 + (r0_2 + 64*x0 + 1664*x1), xmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r0_2 + 64*x0 + 1664*x1), xmask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (r0_2), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_out_ptr0 + (r0_2 + 64*x3), xmask, other=0.0)
    tmp18 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp19 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp21 = tl.load(in_ptr5 + (1 + 4*x1), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp23 = tl.load(in_ptr6 + (x3), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 + tmp3
    tmp6 = tmp4 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp12 = tmp6 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp15 = tl.where(xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tmp4.to(tl.float32)
    tmp20 = tmp18 & tmp19
    tmp22 = tmp20 & tmp21
    tmp24 = 64.0
    tmp25 = tmp6 * tmp24
    tmp26 = tmp25 - tmp10
    tmp27 = tmp11 * tmp16
    tmp28 = tmp26 - tmp27
    tmp29 = tmp23 * tmp28
    tmp30 = 0.0
    tmp31 = tl.where(tmp22, tmp29, tmp30)
    tl.store(out_ptr2 + (r0_2 + 64*x3), tmp17, xmask)
    tl.store(in_out_ptr0 + (r0_2 + 64*x3), tmp31, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/ad/cadvkrv3iyum2f5qkjavqshkjddtiizq7ldu3gc64ytvhfip26z7.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_32 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_65, [0], True), kwargs = {})
triton_red_fused_sum_42 = async_compile.triton('triton_red_fused_sum_42', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32768, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_42', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_sum_42(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 21504
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
        tmp0 = tl.load(in_ptr0 + (x0 + 64*r0_2 + 8192*x1), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/ts/ctse3b5tf52qpiwmjl55kdq5th4xq7nywkqvhwlictklelfssle7.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum, aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_32 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_65, [0], True), kwargs = {})
#   %convert_element_type_134 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_66, torch.float32), kwargs = {})
triton_red_fused__to_copy_sum_43 = async_compile.triton('triton_red_fused__to_copy_sum_43', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 64, 'r0_': 512},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_sum_43', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_sum_43(in_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 64
    r0_numel = 336
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
        tmp0 = tl.load(in_ptr0 + (x0 + 64*r0_1), xmask & r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp4 = tmp2.to(tl.float32)
    tl.store(out_ptr1 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/3l/c3lrw5cd3hgvm22jebzasrycjzbp57xmcmtdj4mlwl7upkz326ku.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %add_34 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_19, %sum_27), kwargs = {})
#   %mul_66 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_21, %mul_11), kwargs = {})
#   %sum_35 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_66, [0, 1, 2, 3]), kwargs = {})
#   %add_38 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_34, %sum_35), kwargs = {})
triton_red_fused_add_native_layer_norm_backward_44 = async_compile.triton('triton_red_fused_add_native_layer_norm_backward_44', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 64, 'r0_': 512},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_backward_44', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_backward_44(in_out_ptr0, in_ptr0, in_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 64
    r0_numel = 336
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
        tmp0 = tl.load(in_ptr0 + (x0 + 64*r0_1), xmask & r0_mask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/ir/cirbu7i52jyntcbofllznvhi6qckbxe66isunqpsmqyrpd3vta46.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.add]
# Source node to ATen node mapping:
# Graph fragment:
#   %add_30 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_14, %view_56), kwargs = {})
triton_poi_fused_add_45 = async_compile.triton('triton_poi_fused_add_45', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_45', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_45(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 11010048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 64)
    x3 = xindex // 64
    x1 = ((xindex // 64) % 16)
    x2 = xindex // 1024
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*x3), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (8*x1 + 128*(x0 // 8) + 1024*x2 + ((x0 % 8))), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/ce/cceffwqghepsg7dfwyulpt65olau6lr35qmrq2xntgq4mph2rigy.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
# Source node to ATen node mapping:
# Graph fragment:
#   %constant_pad_nd_default : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%permute_58, [0, 1, 0, 0]), kwargs = {})
triton_poi_fused_mm_46 = async_compile.triton('triton_poi_fused_mm_46', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mm_46', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mm_46(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 144)
    x1 = xindex // 144
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 143, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x0 + 143*x1), xmask & tmp2, other=0.0).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp3, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/dm/cdmqeftebnr3fnpgzyvmesaql7yp2ecpqajgs2zdyqkrltx3ou4n.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_149 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_17, torch.float32), kwargs = {})
triton_poi_fused__to_copy_47 = async_compile.triton('triton_poi_fused__to_copy_47', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_47', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_47(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/re/crexlawae6g37k4hbt7pjgkx27uqad2s2ipa5e4xowuvmop4oq5m.py
# Topologically Sorted Source Nodes: [clamp_min_1, context], Original ATen: [aten.clamp_min, aten.div, aten.ge, aten.masked_fill, aten.where, aten.slice_backward, aten.add]
# Source node to ATen node mapping:
#   clamp_min_1 => clamp_min_1
#   context => full_default
# Graph fragment:
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%slice_10, 1e-08), kwargs = {})
#   %div_5 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%squeeze_1, %clamp_min_1), kwargs = {})
#   %ge : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%slice_10, 1e-08), kwargs = {})
#   %full_default : [num_users=12] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_17 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge, %div_5, %full_default), kwargs = {})
#   %full_default_18 : [num_users=4] = call_function[target=torch.ops.aten.full.default](args = ([128, 7, 12, 16, 4], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_scatter_default_1 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_18, %where_17, 4, 0, 1), kwargs = {})
#   %select_scatter_default : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %squeeze_4, 4, 3), kwargs = {})
#   %add_31 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default_1, %select_scatter_default), kwargs = {})
#   %select_scatter_default_2 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %squeeze_6, 4, 2), kwargs = {})
#   %add_32 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_31, %select_scatter_default_2), kwargs = {})
#   %select_scatter_default_4 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %squeeze_8, 4, 1), kwargs = {})
#   %add_36 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_32, %select_scatter_default_4), kwargs = {})
#   %add_42 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_36, %slice_29), kwargs = {})
triton_poi_fused_add_clamp_min_div_ge_masked_fill_slice_backward_where_48 = async_compile.triton('triton_poi_fused_add_clamp_min_div_ge_masked_fill_slice_backward_where_48', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_min_div_ge_masked_fill_slice_backward_where_48', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_min_div_ge_masked_fill_slice_backward_where_48(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 688128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 4)
    x1 = xindex // 4
    x2 = xindex
    tmp17 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr5 + (135 + x0 + 144*x1), None).to(tl.float32)
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
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr0 + (x2), tmp32, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/gp/cgpegee5dlfljjw5jmabgoqn5zflvthiumwl47v3pcru4gcp44rp.py
# Topologically Sorted Source Nodes: [context, invert, layer_norm_1], Original ATen: [aten.masked_fill, aten.native_layer_norm_backward, aten.bitwise_not, aten.native_layer_norm]
# Source node to ATen node mapping:
#   context => clone, full_default, where
#   invert => bitwise_not
#   layer_norm_1 => mul_6, sub_3
# Graph fragment:
#   %full_default : [num_users=12] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_75 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_24, %primals_18), kwargs = {})
#   %sum_39 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_75, [4], True), kwargs = {})
#   %bitwise_not : [num_users=2] = call_function[target=torch.ops.aten.bitwise_not.default](args = (%unsqueeze,), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_not, %full_default, %view), kwargs = {})
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%where,), kwargs = {memory_format: torch.contiguous_format})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone, %getitem_3), kwargs = {})
#   %mul_6 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_1), kwargs = {})
#   %mul_77 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_75, %mul_6), kwargs = {})
#   %sum_40 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_77, [4], True), kwargs = {})
triton_per_fused_bitwise_not_masked_fill_native_layer_norm_native_layer_norm_backward_49 = async_compile.triton('triton_per_fused_bitwise_not_masked_fill_native_layer_norm_native_layer_norm_backward_49', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*i1', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_bitwise_not_masked_fill_native_layer_norm_native_layer_norm_backward_49', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 2, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_bitwise_not_masked_fill_native_layer_norm_native_layer_norm_backward_49(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 172032
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
    x2 = (xindex % 16)
    x3 = ((xindex // 16) % 12)
    x4 = xindex // 192
    tmp0 = tl.load(in_ptr0 + (r0_1 + 144*x0), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last').to(tl.int1)
    tmp9 = tl.load(in_ptr3 + (x2 + 8*x3 + 104*r0_1 + 6656*x4), None)
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp6 = tl.sum(tmp4, 1)[:, None]
    tmp8 = tmp7 == 0
    tmp10 = 0.0
    tmp11 = tl.where(tmp8, tmp10, tmp9)
    tmp13 = tmp11 - tmp12
    tmp15 = tmp13 * tmp14
    tmp16 = tmp3 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
    tmp19 = tl.sum(tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, None)
    tl.store(out_ptr1 + (x0), tmp19, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/by/cbyetfq3eywekfiu22nw33bmrla6x637rqusyt5bqujd6ynmpcr7.py
# Topologically Sorted Source Nodes: [context, invert, layer_norm_1], Original ATen: [aten.masked_fill, aten.bitwise_not, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   context => clone, full_default, where
#   invert => bitwise_not
#   layer_norm_1 => mul_6, sub_3
# Graph fragment:
#   %full_default : [num_users=12] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %bitwise_not : [num_users=2] = call_function[target=torch.ops.aten.bitwise_not.default](args = (%unsqueeze,), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_not, %full_default, %view), kwargs = {})
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%where,), kwargs = {memory_format: torch.contiguous_format})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone, %getitem_3), kwargs = {})
#   %mul_6 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_1), kwargs = {})
#   %mul_80 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_24, %mul_6), kwargs = {})
#   %sum_41 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_80, [0, 1, 2, 3]), kwargs = {})
triton_red_fused_bitwise_not_masked_fill_native_layer_norm_native_layer_norm_backward_50 = async_compile.triton('triton_red_fused_bitwise_not_masked_fill_native_layer_norm_native_layer_norm_backward_50', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i1', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_bitwise_not_masked_fill_native_layer_norm_native_layer_norm_backward_50', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_bitwise_not_masked_fill_native_layer_norm_native_layer_norm_backward_50(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 86016
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 1344)
    x1 = xindex // 1344
    _tmp13 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x1 + 144*r0_2 + 18432*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_2 + 128*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.int1)
        tmp4 = tl.load(in_ptr2 + (8*((((r0_2 + 128*x0) // 16) % 12)) + 104*x1 + 6656*((r0_2 + 128*x0) // 192) + ((r0_2 % 16))), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr3 + (r0_2 + 128*x0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr4 + (r0_2 + 128*x0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2 == 0
        tmp5 = 0.0
        tmp6 = tl.where(tmp3, tmp5, tmp4)
        tmp8 = tmp6 - tmp7
        tmp10 = tmp8 * tmp9
        tmp11 = tmp1 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(r0_mask, tmp14, _tmp13)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp13, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/jl/cjlvjup6pyarisriqak3b7n7kks3cegrtoj4mz3gpr7qtwr5lcnj.py
# Topologically Sorted Source Nodes: [context, invert, layer_norm_1], Original ATen: [aten.masked_fill, aten.bitwise_not, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   context => clone, full_default, where
#   invert => bitwise_not
#   layer_norm_1 => mul_6, sub_3
# Graph fragment:
#   %full_default : [num_users=12] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %bitwise_not : [num_users=2] = call_function[target=torch.ops.aten.bitwise_not.default](args = (%unsqueeze,), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_not, %full_default, %view), kwargs = {})
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%where,), kwargs = {memory_format: torch.contiguous_format})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone, %getitem_3), kwargs = {})
#   %mul_6 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_1), kwargs = {})
#   %mul_80 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_24, %mul_6), kwargs = {})
#   %sum_41 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_80, [0, 1, 2, 3]), kwargs = {})
triton_red_fused_bitwise_not_masked_fill_native_layer_norm_native_layer_norm_backward_51 = async_compile.triton('triton_red_fused_bitwise_not_masked_fill_native_layer_norm_native_layer_norm_backward_51', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 64, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_bitwise_not_masked_fill_native_layer_norm_native_layer_norm_backward_51', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_bitwise_not_masked_fill_native_layer_norm_native_layer_norm_backward_51(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
        tmp0 = tl.load(in_ptr0 + (r0_1 + 1344*x0), xmask & r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/ax/caxqmxselpvl5etgbofambdycedfk2lv5dk46haae6te4ckyziek.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_42 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%slice_24, [0, 1, 2, 3]), kwargs = {})
triton_red_fused_native_layer_norm_backward_52 = async_compile.triton('triton_red_fused_native_layer_norm_backward_52', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_52', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_layer_norm_backward_52(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 86016
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 64)
    x1 = xindex // 64
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 144*r0_2 + 18432*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(r0_mask, tmp4, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp3, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/fq/cfqflrhk5yvta4jetexrhszkcmxvmt524rlly7v7q7x6cjmltkoj.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_42 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%slice_24, [0, 1, 2, 3]), kwargs = {})
triton_red_fused_native_layer_norm_backward_53 = async_compile.triton('triton_red_fused_native_layer_norm_backward_53', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_53', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_layer_norm_backward_53(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
        tmp0 = tl.load(in_ptr0 + (x0 + 64*r0_1), xmask & r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/zm/czmjgzpzlhvo2hy2e25j2ru7ovbs5ptmnnpdnt6ithgjkrli7h3y.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.div, aten.mul]
# Source node to ATen node mapping:
# Graph fragment:
#   %div_12 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%squeeze_9, %unsqueeze_17), kwargs = {})
#   %mul_81 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_12, %unsqueeze_15), kwargs = {})
triton_poi_fused_div_mul_54 = async_compile.triton('triton_poi_fused_div_mul_54', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_mul_54', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_mul_54(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 172032
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 192
    tmp0 = tl.load(in_ptr0 + (134 + 144*x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2 * tmp2
    tmp4 = 0.010000000000000002
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = (tmp1 / tmp6)
    tmp8 = tmp7 * tmp2
    tl.store(out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/f5/cf55nvp7hbii7jvqc3czqwvwqklnfpgrynztdiawrvr7s2zxtiit.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_43 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%slice_25, [3], True), kwargs = {})
triton_per_fused_sum_55 = async_compile.triton('triton_per_fused_sum_55', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1048576, 'r0_': 16},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_55', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_sum_55(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 688128
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
    tmp0 = tl.load(in_ptr0 + (64 + x0 + 144*r0_2 + 2304*x1), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.sum(tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/yy/cyyqgu2zh4imr5eyutya3dayzixvag2qahcodb5ycn42tib36ac7.py
# Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm_backward, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm => mul_3, sub_2
# Graph fragment:
#   %mul_83 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_10, %primals_15), kwargs = {})
#   %mul_84 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_83, 64), kwargs = {})
#   %sum_44 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_83, [3], True), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_7, %getitem_1), kwargs = {})
#   %mul_3 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt), kwargs = {})
#   %mul_85 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_83, %mul_3), kwargs = {})
#   %sum_45 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_85, [3], True), kwargs = {})
#   %mul_86 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, %sum_45), kwargs = {})
#   %sub_25 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_84, %sum_44), kwargs = {})
#   %sub_26 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_25, %mul_86), kwargs = {})
#   %div_13 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt, 64), kwargs = {})
#   %mul_87 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_13, %sub_26), kwargs = {})
triton_per_fused_native_layer_norm_native_layer_norm_backward_56 = async_compile.triton('triton_per_fused_native_layer_norm_native_layer_norm_backward_56', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16384, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_56', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_native_layer_norm_backward_56(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 10752
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


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/hv/chvcupxm6buty56n2knj3gie5grwfjlda2l7nxry55wvd6ztgjk3.py
# Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   layer_norm => mul_3, sub_2
# Graph fragment:
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_7, %getitem_1), kwargs = {})
#   %mul_3 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt), kwargs = {})
#   %mul_88 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_10, %mul_3), kwargs = {})
#   %sum_46 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_88, [0, 1, 2]), kwargs = {})
#   %sum_47 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%squeeze_10, [0, 1, 2]), kwargs = {})
triton_red_fused_native_layer_norm_native_layer_norm_backward_57 = async_compile.triton('triton_red_fused_native_layer_norm_native_layer_norm_backward_57', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_native_layer_norm_backward_57', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_layer_norm_native_layer_norm_backward_57(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 5376
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
        tmp0 = tl.load(in_ptr0 + (x0 + 64*r0_2 + 8192*x1), xmask & r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + 64*r0_2 + 8192*x1), xmask & r0_mask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/ve/cvem4xrab7b46dwucesrmnbucizjqbiit3umylocq76vbptsm6i3.py
# Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   layer_norm => mul_3, sub_2
# Graph fragment:
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_7, %getitem_1), kwargs = {})
#   %mul_3 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt), kwargs = {})
#   %mul_88 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_10, %mul_3), kwargs = {})
#   %sum_46 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_88, [0, 1, 2]), kwargs = {})
triton_red_fused_native_layer_norm_native_layer_norm_backward_58 = async_compile.triton('triton_red_fused_native_layer_norm_native_layer_norm_backward_58', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 64, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_native_layer_norm_backward_58', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_layer_norm_native_layer_norm_backward_58(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 64
    r0_numel = 84
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
        tmp0 = tl.load(in_ptr0 + (x0 + 64*r0_1), xmask & r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/cj/ccjdwechad4r7vpreltr7n3dmfgw752puspeotcm4t2eatfdsymd.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.unfold_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %full_default_41 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([896, 64, 104], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_unfold_backward_59 = async_compile.triton('triton_poi_fused_unfold_backward_59', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_unfold_backward_59', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_unfold_backward_59(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5963776
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/c5/cc56fru4holz4b47iv4a6parf4tey7cj6cd44fi43ubfgldftfpv.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.unfold_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %full_default_41 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([896, 64, 104], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_41, [None, None, %view_74], %view_75, True), kwargs = {})
triton_poi_fused_unfold_backward_60 = async_compile.triton('triton_poi_fused_unfold_backward_60', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_unfold_backward_60', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_unfold_backward_60(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/f2/cf2mcbcbneejaksylk5vcjexzpeymnba3mgwhoyrsi7jndreoy5q.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
# Graph fragment:
#   %constant_pad_nd_2 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%index_put, [0, -8]), kwargs = {})
triton_poi_fused_constant_pad_nd_61 = async_compile.triton('triton_poi_fused_constant_pad_nd_61', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_61', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_61(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5505024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 96)
    x1 = xindex // 96
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 104, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x0 + 104*x1), tmp2, other=0.0)
    tl.store(out_ptr0 + (x2), tmp3, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_2, primals_3, primals_4, primals_7, primals_8, primals_9, primals_15, primals_18, primals_24, primals_28, primals_29, primals_34, primals_38, primals_41, primals_45, primals_52, unfold, getitem_1, rsqrt, getitem_3, rsqrt_1, view_7, addmm_1, view_9, mul_11, view_11, unsqueeze_24, mul_14, view_13, unsqueeze_30, view_15, unsqueeze_36, cat_4, view_18, cat_5, div_3, cat_7, getitem_11, rsqrt_5, view_31, addmm_7, view_33, unsqueeze_41, permute_19, permute_23, permute_28, permute_29, permute_30, permute_31, permute_34, permute_39, mul_42, div_8, permute_47, div_9, div_10, permute_54, permute_58, tangents_1, tangents_2 = args
    args.clear()
    assert_size_stride(primals_2, (128, 7, 12, 16), (1344, 192, 16, 1))
    assert_size_stride(primals_3, (128, 7, 12, 16), (1344, 192, 16, 1))
    assert_size_stride(primals_4, (128, 7), (7, 1))
    assert_size_stride(primals_7, (128, 7, 12, 64), (5376, 768, 64, 1))
    assert_size_stride(primals_8, (128, 7, 12, 4), (336, 48, 4, 1))
    assert_size_stride(primals_9, (128, 7, 12, 16, 4), (5376, 768, 64, 4, 1))
    assert_size_stride(primals_15, (64, ), (1, ))
    assert_size_stride(primals_18, (64, ), (1, ))
    assert_size_stride(primals_24, (128, 7, 12), (84, 12, 1))
    assert_size_stride(primals_28, (128, 7, 12, 4), (336, 48, 4, 1))
    assert_size_stride(primals_29, (64, ), (1, ))
    assert_size_stride(primals_34, (128, 7, 12), (84, 12, 1))
    assert_size_stride(primals_38, (128, 7, 12, 12), (1008, 144, 12, 1))
    assert_size_stride(primals_41, (128, 7, 12), (84, 12, 1))
    assert_size_stride(primals_45, (128, 7, 12, 10), (840, 120, 1, 12))
    assert_size_stride(primals_52, (128, ), (1, ))
    assert_size_stride(unfold, (896, 64, 12, 16), (6656, 104, 8, 1))
    assert_size_stride(getitem_1, (128, 7, 12, 1), (84, 12, 1, 1))
    assert_size_stride(rsqrt, (128, 7, 12, 1), (84, 12, 1, 1))
    assert_size_stride(getitem_3, (128, 7, 12, 16, 1), (1344, 192, 16, 1, 1))
    assert_size_stride(rsqrt_1, (128, 7, 12, 16, 1), (1344, 192, 16, 1, 1))
    assert_size_stride(view_7, (172032, 143), (143, 1))
    assert_size_stride(addmm_1, (172032, 64), (64, 1))
    assert_size_stride(view_9, (172032, 64), (64, 1))
    assert_size_stride(mul_11, (128, 7, 12, 4, 64), (21504, 3072, 256, 64, 1))
    assert_size_stride(view_11, (43008, 5), (5, 1))
    assert_size_stride(unsqueeze_24, (128, 7, 12, 1, 4), (336, 48, 4, 4, 1))
    assert_size_stride(mul_14, (128, 7, 12, 12, 64), (64512, 9216, 768, 64, 1))
    assert_size_stride(view_13, (129024, 4), (4, 1))
    assert_size_stride(unsqueeze_30, (128, 7, 12, 1, 12), (1008, 144, 12, 12, 1))
    assert_size_stride(view_15, (107520, 5), (5, 1))
    assert_size_stride(unsqueeze_36, (128, 7, 12, 1, 10), (840, 120, 1, 120, 12))
    assert_size_stride(cat_4, (128, 7, 12, 26), (2184, 312, 26, 1))
    assert_size_stride(view_18, (279552, 64), (64, 1))
    assert_size_stride(cat_5, (128, 7, 12, 16, 26), (34944, 4992, 416, 26, 1))
    assert_size_stride(div_3, (128, 7, 12, 8, 16, 27), (290304, 41472, 3456, 432, 27, 1))
    assert_size_stride(cat_7, (128, 7, 12, 16, 128), (172032, 24576, 2048, 128, 1))
    assert_size_stride(getitem_11, (128, 7, 12, 16, 1), (1344, 192, 16, 1, 1))
    assert_size_stride(rsqrt_5, (128, 7, 12, 16, 1), (1344, 192, 16, 1, 1))
    assert_size_stride(view_31, (172032, 128), (128, 1))
    assert_size_stride(addmm_7, (172032, 64), (64, 1))
    assert_size_stride(view_33, (172032, 64), (64, 1))
    assert_size_stride(unsqueeze_41, (128, 7, 12, 1), (84, 12, 1, 1))
    assert_size_stride(permute_19, (1, 64), (64, 1))
    assert_size_stride(permute_23, (64, 128), (128, 1))
    assert_size_stride(permute_28, (86016, 26, 16), (416, 1, 26))
    assert_size_stride(permute_29, (86016, 8, 26), (208, 1, 8))
    assert_size_stride(permute_30, (86016, 8, 16), (128, 1, 8))
    assert_size_stride(permute_31, (86016, 26, 8), (208, 1, 26))
    assert_size_stride(permute_34, (64, 64), (64, 1))
    assert_size_stride(permute_39, (64, 64), (64, 1))
    assert_size_stride(mul_42, (128, 7, 12, 10, 64), (53760, 7680, 64, 768, 1))
    assert_size_stride(div_8, (128, 7, 12, 10, 1), (840, 120, 10, 1, 1))
    assert_size_stride(permute_47, (64, 4), (4, 1))
    assert_size_stride(div_9, (128, 7, 12, 12, 1), (1008, 144, 12, 1, 1))
    assert_size_stride(div_10, (128, 7, 12, 4, 1), (336, 48, 4, 1, 1))
    assert_size_stride(permute_54, (64, 64), (64, 1))
    assert_size_stride(permute_58, (64, 143), (143, 1))
    assert_size_stride(tangents_1, (128, 7, 12, 16), (1344, 192, 16, 1))
    assert_size_stride(tangents_2, (128, 7, 12, 8, 16, 27), (290304, 41472, 3456, 432, 27, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 7, 12, 16), (1344, 192, 16, 1), torch.bfloat16)
        buf3 = empty_strided_cuda((1, 1, 21), (21, 21, 1), torch.float32)
        # Topologically Sorted Source Nodes: [invert_5, and__6, eligible, correction_1], Original ATen: [aten._to_copy, aten.bitwise_not, aten.bitwise_and, aten.scalar_tensor, aten.where, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_bitwise_and_bitwise_not_scalar_tensor_sum_where_0.run(primals_3, primals_2, unsqueeze_41, tangents_1, buf0, buf3, 21, 8192, stream=stream0)
        del primals_2
        del tangents_1
        del unsqueeze_41
        buf1 = empty_strided_cuda((172032, 64), (64, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf0, (172032, 1), (1, 0), 0), permute_19, out=buf1)
        del permute_19
        buf2 = empty_strided_cuda((1, 64), (64, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf0, (1, 172032), (0, 1), 0), view_33, out=buf2)
        del buf0
        del view_33
        buf6 = empty_strided_cuda((1, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_sum_1.run(buf3, buf6, 1, 21, stream=stream0)
        del buf3
        buf5 = empty_strided_cuda((1, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(buf2, buf5, 64, stream=stream0)
        del buf2
        buf7 = reinterpret_tensor(buf1, (128, 7, 12, 16, 64), (86016, 12288, 1024, 64, 1), 0); del buf1  # reuse
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.gelu_backward, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_gelu_backward_3.run(buf7, addmm_7, 11010048, stream=stream0)
        del addmm_7
        buf8 = empty_strided_cuda((172032, 128), (128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf7, (172032, 64), (64, 1), 0), permute_23, out=buf8)
        del permute_23
        buf9 = empty_strided_cuda((64, 128), (128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf7, (64, 172032), (1, 64), 0), view_31, out=buf9)
        del view_31
        buf10 = empty_strided_cuda((1, 64, 1344), (86016, 1, 64), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_4.run(buf7, buf10, 86016, 128, stream=stream0)
        buf13 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_sum_5.run(buf10, buf13, 64, 1344, stream=stream0)
        buf12 = empty_strided_cuda((64, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_6.run(buf9, buf12, 8192, stream=stream0)
        del buf9
        buf20 = empty_strided_cuda((128, 7, 12, 16, 128), (172032, 24576, 2048, 128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten._to_copy, aten.native_layer_norm_backward, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_native_layer_norm_backward_7.run(buf8, primals_52, cat_7, getitem_11, rsqrt_5, buf20, 172032, 128, stream=stream0)
        del primals_52
        buf16 = empty_strided_cuda((128, 1344), (1, 128), torch.float32)
        buf18 = empty_strided_cuda((128, 1344), (1, 128), torch.float32)
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten._to_copy, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_8.run(buf8, cat_7, getitem_11, rsqrt_5, buf16, buf18, 172032, 128, stream=stream0)
        del buf8
        del cat_7
        del getitem_11
        del rsqrt_5
        buf17 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten._to_copy, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_9.run(buf16, buf17, 128, 1344, stream=stream0)
        buf19 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_9.run(buf18, buf19, 128, 1344, stream=stream0)
        buf21 = reinterpret_tensor(buf7, (128, 7, 12, 8, 16, 8), (86016, 12288, 1024, 128, 8, 1), 0); del buf7  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_10.run(buf20, buf21, 11010048, stream=stream0)
        buf22 = empty_strided_cuda((86016, 26, 8), (208, 8, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_28, reinterpret_tensor(buf21, (86016, 16, 8), (128, 8, 1), 0), out=buf22)
        del permute_28
        buf23 = empty_strided_cuda((86016, 16, 26), (416, 26, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf21, (86016, 16, 8), (128, 8, 1), 0), permute_29, out=buf23)
        del permute_29
        buf24 = empty_strided_cuda((128, 7, 12, 8, 16, 1), (10752, 1536, 128, 16, 1, 1376256), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.slice_backward, aten.add, aten._softmax_backward_data]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_backward_data__to_copy_add_slice_backward_11.run(tangents_2, buf23, div_3, buf24, 1376256, 27, stream=stream0)
        buf25 = reinterpret_tensor(buf18, (128, 7, 12, 1, 16, 1), (1344, 192, 16, 16, 1, 1), 0); del buf18  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_12.run(div_3, buf24, tangents_2, buf23, buf25, 172032, 8, stream=stream0)
        buf26 = empty_strided_cuda((128, 7, 12, 8, 16, 26), (279552, 39936, 3328, 416, 26, 1), torch.float32)
        buf28 = empty_strided_cuda((128, 7, 12, 8, 16, 26), (279552, 39936, 3328, 416, 26, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [context, invert_4], Original ATen: [aten.masked_fill, aten.bitwise_not, aten._to_copy, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_bitwise_not_div_masked_fill_13.run(cat_4, div_3, buf24, tangents_2, buf23, buf26, buf28, 35782656, stream=stream0)
        del buf23
        del buf24
        del cat_4
        del div_3
        del tangents_2
        buf27 = empty_strided_cuda((128, 7, 12, 1, 16, 26), (34944, 4992, 416, 416, 26, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_14.run(buf26, buf27, 4472832, 8, stream=stream0)
        del buf26
        buf29 = empty_strided_cuda((86016, 8, 26), (208, 26, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_30, reinterpret_tensor(buf28, (86016, 16, 26), (416, 26, 1), 0), out=buf29)
        del permute_30
        buf30 = reinterpret_tensor(buf21, (86016, 16, 8), (128, 8, 1), 0); del buf21  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf28, (86016, 16, 26), (416, 26, 1), 0), permute_31, out=buf30)
        del buf28
        del permute_31
        buf31 = empty_strided_cuda((279552, 64), (64, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_15.run(buf22, buf31, 17891328, stream=stream0)
        buf32 = reinterpret_tensor(buf22, (279552, 64), (64, 1), 0); del buf22  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf31, permute_34, out=buf32)
        del permute_34
        buf33 = empty_strided_cuda((64, 64), (64, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf31, (64, 279552), (1, 64), 0), view_18, out=buf33)
        buf34 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_16.run(buf31, buf34, 86016, 208, stream=stream0)
        buf37 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_sum_5.run(buf34, buf37, 64, 1344, stream=stream0)
        buf36 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_17.run(buf33, buf36, 4096, stream=stream0)
        buf38 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone, aten._unsafe_view]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_18.run(buf29, buf38, 279552, 64, stream=stream0)
        buf39 = reinterpret_tensor(buf29, (279552, 64), (64, 1), 0); del buf29  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf38, permute_39, out=buf39)
        del permute_39
        buf40 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf38, (64, 279552), (1, 64), 0), view_18, out=buf40)
        del view_18
        buf41 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_16.run(buf38, buf41, 86016, 208, stream=stream0)
        del buf38
        buf44 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_sum_5.run(buf41, buf44, 64, 1344, stream=stream0)
        buf43 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_17.run(buf40, buf43, 4096, stream=stream0)
        buf45 = reinterpret_tensor(buf16, (128, 7, 12, 16, 1), (1344, 192, 16, 1, 1), 0); del buf16  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_19.run(cat_5, buf27, unsqueeze_36, buf45, 172032, 10, stream=stream0)
        del unsqueeze_36
        buf46 = empty_strided_cuda((128, 7, 12, 1, 10), (840, 120, 10, 10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_20.run(cat_5, buf27, primals_9, buf46, 107520, 16, stream=stream0)
        buf47 = empty_strided_cuda((128, 7, 12, 10), (840, 120, 1, 12), torch.float32)
        # Topologically Sorted Source Nodes: [context, and__4, valid_2], Original ATen: [aten.masked_fill, aten.bitwise_and, aten.where]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bitwise_and_masked_fill_where_21.run(primals_45, primals_41, primals_8, buf46, buf47, 8960, 12, stream=stream0)
        del buf46
        buf48 = empty_strided_cuda((1, 1, 1, 1, 64, 840), (53760, 53760, 53760, 53760, 1, 64), torch.float32)
        buf50 = empty_strided_cuda((1, 1, 1, 1, 64, 840), (53760, 53760, 53760, 53760, 1, 64), torch.float32)
        buf58 = empty_strided_cuda((64, 840), (1, 64), torch.float32)
        buf60 = empty_strided_cuda((64, 840), (1, 64), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum, aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_backward_sum_22.run(buf32, buf39, mul_42, buf48, buf50, buf58, buf60, 53760, 128, stream=stream0)
        buf49 = empty_strided_cuda((1, 1, 1, 1, 64), (64, 64, 64, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_23.run(buf48, buf49, 64, 840, stream=stream0)
        del buf48
        buf55 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_sum_24.run(buf50, buf55, 64, 840, stream=stream0)
        del buf50
        buf52 = empty_strided_cuda((128, 7, 12, 10, 64), (53760, 7680, 640, 64, 1), torch.bfloat16)
        buf62 = mul_42; del mul_42  # reuse
        # Topologically Sorted Source Nodes: [context], Original ATen: [aten.masked_fill, aten._to_copy, aten.native_layer_norm_backward, aten.where]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_masked_fill_native_layer_norm_backward_where_25.run(buf62, buf32, buf39, primals_29, primals_45, primals_41, primals_8, div_8, buf52, 107520, 64, stream=stream0)
        del div_8
        del primals_41
        del primals_45
        buf53 = empty_strided_cuda((64, 5), (5, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf52, (64, 107520), (1, 64), 0), view_15, out=buf53)
        del buf52
        del view_15
        buf54 = empty_strided_cuda((64, 5), (5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_26.run(buf53, buf54, 320, stream=stream0)
        buf59 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_23.run(buf58, buf59, 64, 840, stream=stream0)
        del buf58
        buf61 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_23.run(buf60, buf61, 64, 840, stream=stream0)
        del buf60
        buf63 = empty_strided_cuda((128, 7, 12, 16, 1), (1344, 192, 16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_sum_27.run(cat_5, buf27, unsqueeze_30, buf63, 172032, 12, stream=stream0)
        del unsqueeze_30
        buf64 = empty_strided_cuda((128, 7, 12, 1, 12), (1008, 144, 12, 12, 1), torch.float32)
        buf65 = reinterpret_tensor(buf64, (128, 7, 12, 12), (1008, 144, 12, 1), 0); del buf64  # reuse
        # Topologically Sorted Source Nodes: [context, and__2, valid_1], Original ATen: [aten.masked_fill, aten.mul, aten.sum, aten.bitwise_and, aten.where]
        stream0 = get_raw_stream(0)
        triton_per_fused_bitwise_and_masked_fill_mul_sum_where_28.run(buf65, cat_5, buf27, primals_9, primals_38, primals_34, primals_8, 129024, 16, stream=stream0)
        buf66 = empty_strided_cuda((1, 1, 1, 1, 64, 1008), (64512, 64512, 64512, 64512, 1, 64), torch.float32)
        buf77 = empty_strided_cuda((64, 1008), (1, 64), torch.float32)
        buf79 = empty_strided_cuda((64, 1008), (1, 64), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_layer_norm_backward_sum_29.run(buf32, buf39, mul_14, buf66, buf77, buf79, 64512, 128, stream=stream0)
        buf67 = empty_strided_cuda((1, 1, 1, 1, 64), (64, 64, 64, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_30.run(buf66, buf67, 64, 1008, stream=stream0)
        buf68 = empty_strided_cuda((128, 7, 12, 12, 64), (64512, 9216, 768, 64, 1), torch.bfloat16)
        buf82 = mul_14; del mul_14  # reuse
        # Topologically Sorted Source Nodes: [context], Original ATen: [aten.masked_fill, aten._to_copy, aten.native_layer_norm_backward, aten.where]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_masked_fill_native_layer_norm_backward_where_31.run(buf82, buf32, buf39, primals_29, primals_38, primals_34, primals_8, div_9, buf68, 129024, 64, stream=stream0)
        del div_9
        buf69 = empty_strided_cuda((129024, 4), (4, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf68, (129024, 64), (64, 1), 0), permute_47, out=buf69)
        del permute_47
        buf70 = empty_strided_cuda((64, 4), (4, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf68, (64, 129024), (1, 64), 0), view_13, out=buf70)
        del view_13
        buf71 = reinterpret_tensor(buf66, (1, 64, 1008), (64512, 1, 64), 0); del buf66  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_32.run(buf68, buf71, 64512, 128, stream=stream0)
        del buf68
        buf74 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_sum_33.run(buf71, buf74, 64, 1008, stream=stream0)
        del buf71
        buf73 = empty_strided_cuda((64, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_34.run(buf70, buf73, 256, stream=stream0)
        del buf70
        buf78 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_30.run(buf77, buf78, 64, 1008, stream=stream0)
        del buf77
        buf80 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_30.run(buf79, buf80, 64, 1008, stream=stream0)
        del buf79
        buf81 = empty_strided_cuda((128, 7, 12, 12, 4), (4032, 576, 48, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [context], Original ATen: [aten.masked_fill, aten._to_copy, aten.where]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_masked_fill_where_35.run(primals_38, primals_34, primals_8, buf69, buf81, 516096, stream=stream0)
        del buf69
        del primals_34
        del primals_38
        buf83 = empty_strided_cuda((128, 7, 12, 16, 1), (1344, 192, 16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sum_36.run(cat_5, buf27, unsqueeze_24, buf83, 172032, stream=stream0)
        del unsqueeze_24
        buf84 = empty_strided_cuda((128, 7, 12, 1, 4), (336, 48, 4, 4, 1), torch.float32)
        buf85 = reinterpret_tensor(buf84, (128, 7, 12, 4), (336, 48, 4, 1), 0); del buf84  # reuse
        # Topologically Sorted Source Nodes: [context, and_, valid], Original ATen: [aten.masked_fill, aten.mul, aten.sum, aten.bitwise_and, aten.where]
        stream0 = get_raw_stream(0)
        triton_per_fused_bitwise_and_masked_fill_mul_sum_where_37.run(buf85, cat_5, buf27, primals_9, primals_28, primals_24, primals_8, 43008, 16, stream=stream0)
        del buf27
        del cat_5
        buf86 = empty_strided_cuda((1, 1, 1, 1, 64, 336), (21504, 21504, 21504, 21504, 1, 64), torch.float32)
        buf97 = empty_strided_cuda((64, 336), (1, 64), torch.float32)
        buf99 = empty_strided_cuda((64, 336), (1, 64), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_layer_norm_backward_sum_38.run(buf32, buf39, mul_11, buf86, buf97, buf99, 21504, 128, stream=stream0)
        buf87 = empty_strided_cuda((1, 1, 1, 1, 64), (64, 64, 64, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_39.run(buf86, buf87, 64, 336, stream=stream0)
        buf88 = empty_strided_cuda((3, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.select_backward, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_select_backward_40.run(buf49, buf67, buf87, buf88, 192, stream=stream0)
        buf89 = empty_strided_cuda((128, 7, 12, 4, 64), (21504, 3072, 256, 64, 1), torch.bfloat16)
        buf103 = mul_11; del mul_11  # reuse
        # Topologically Sorted Source Nodes: [context], Original ATen: [aten.masked_fill, aten._to_copy, aten.native_layer_norm_backward, aten.where]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_masked_fill_native_layer_norm_backward_where_41.run(buf103, buf32, buf39, primals_29, primals_28, primals_24, primals_8, div_10, buf89, 43008, 64, stream=stream0)
        del buf32
        del buf39
        del div_10
        del primals_24
        del primals_28
        del primals_29
        del primals_8
        buf90 = buf53; del buf53  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf89, (64, 43008), (1, 64), 0), view_11, out=buf90)
        del view_11
        buf91 = reinterpret_tensor(buf86, (1, 64, 336), (21504, 1, 64), 0); del buf86  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_42.run(buf89, buf91, 21504, 128, stream=stream0)
        del buf89
        buf94 = reinterpret_tensor(buf87, (64, ), (1, ), 0); del buf87  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_sum_43.run(buf91, buf94, 64, 336, stream=stream0)
        del buf91
        buf93 = empty_strided_cuda((64, 5), (5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_26.run(buf90, buf93, 320, stream=stream0)
        del buf90
        buf101 = buf59; del buf59  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_native_layer_norm_backward_44.run(buf101, buf97, buf78, 64, 336, stream=stream0)
        del buf97
        buf102 = buf61; del buf61  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_native_layer_norm_backward_44.run(buf102, buf99, buf80, 64, 336, stream=stream0)
        del buf99
        buf104 = empty_strided_cuda((128, 7, 12, 16, 64), (86016, 12288, 1024, 64, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_45.run(buf20, buf30, buf104, 11010048, stream=stream0)
        del buf20
        buf105 = reinterpret_tensor(buf30, (172032, 64), (64, 1), 0); del buf30  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf104, (172032, 64), (64, 1), 0), permute_54, out=buf105)
        del permute_54
        buf106 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf104, (64, 172032), (1, 64), 0), view_9, out=buf106)
        del view_9
        buf107 = buf41; del buf41  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_4.run(buf104, buf107, 86016, 128, stream=stream0)
        del buf104
        buf110 = buf80; del buf80  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_sum_5.run(buf107, buf110, 64, 1344, stream=stream0)
        buf109 = empty_strided_cuda((64, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_17.run(buf106, buf109, 4096, stream=stream0)
        del buf106
        buf111 = reinterpret_tensor(buf105, (128, 7, 12, 16, 64), (86016, 12288, 1024, 64, 1), 0); del buf105  # reuse
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.gelu_backward, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_gelu_backward_3.run(buf111, addmm_1, 11010048, stream=stream0)
        del addmm_1
        buf112 = empty_strided_cuda((64, 144), (144, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mm_46.run(permute_58, buf112, 9216, stream=stream0)
        del permute_58
        buf113 = empty_strided_cuda((172032, 144), (144, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf111, (172032, 64), (64, 1), 0), buf112, out=buf113)
        del buf112
        buf114 = empty_strided_cuda((64, 143), (143, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf111, (64, 172032), (1, 64), 0), view_7, out=buf114)
        del view_7
        buf115 = buf107; del buf107  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_4.run(buf111, buf115, 86016, 128, stream=stream0)
        del buf111
        buf118 = buf78; del buf78  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_sum_5.run(buf115, buf118, 64, 1344, stream=stream0)
        buf117 = empty_strided_cuda((64, 143), (143, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_47.run(buf114, buf117, 9152, stream=stream0)
        del buf114
        buf119 = empty_strided_cuda((128, 7, 12, 16, 4), (5376, 768, 64, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [clamp_min_1, context], Original ATen: [aten.clamp_min, aten.div, aten.ge, aten.masked_fill, aten.where, aten.slice_backward, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_min_div_ge_masked_fill_slice_backward_where_48.run(primals_9, buf25, buf45, buf63, buf83, buf113, buf119, 688128, stream=stream0)
        del buf25
        del primals_9
        buf120 = reinterpret_tensor(buf83, (128, 7, 12, 16, 1), (1344, 192, 16, 1, 172032), 0); del buf83  # reuse
        buf121 = reinterpret_tensor(buf63, (128, 7, 12, 16, 1), (1344, 192, 16, 1, 172032), 0); del buf63  # reuse
        # Topologically Sorted Source Nodes: [context, invert, layer_norm_1], Original ATen: [aten.masked_fill, aten.native_layer_norm_backward, aten.bitwise_not, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_bitwise_not_masked_fill_native_layer_norm_native_layer_norm_backward_49.run(buf113, primals_18, primals_3, unfold, getitem_3, rsqrt_1, buf120, buf121, 172032, 64, stream=stream0)
        buf122 = reinterpret_tensor(buf115, (64, 1344), (1344, 1), 0); del buf115  # reuse
        # Topologically Sorted Source Nodes: [context, invert, layer_norm_1], Original ATen: [aten.masked_fill, aten.bitwise_not, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_bitwise_not_masked_fill_native_layer_norm_native_layer_norm_backward_50.run(buf113, primals_3, unfold, getitem_3, rsqrt_1, buf122, 86016, 128, stream=stream0)
        buf123 = reinterpret_tensor(buf67, (64, ), (1, ), 0); del buf67  # reuse
        # Topologically Sorted Source Nodes: [context, invert, layer_norm_1], Original ATen: [aten.masked_fill, aten.bitwise_not, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_bitwise_not_masked_fill_native_layer_norm_native_layer_norm_backward_51.run(buf122, buf123, 64, 1344, stream=stream0)
        buf124 = reinterpret_tensor(buf122, (64, 1344), (1, 64), 0); del buf122  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_layer_norm_backward_52.run(buf113, buf124, 86016, 128, stream=stream0)
        buf125 = reinterpret_tensor(buf49, (64, ), (1, ), 0); del buf49  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_layer_norm_backward_53.run(buf124, buf125, 64, 1344, stream=stream0)
        del buf124
        buf126 = reinterpret_tensor(buf45, (128, 7, 12, 16), (1344, 192, 16, 1), 0); del buf45  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.div, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_mul_54.run(buf113, primals_4, buf126, 172032, stream=stream0)
        del primals_4
        buf127 = empty_strided_cuda((128, 7, 12, 1, 64), (5376, 768, 64, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_55.run(buf113, buf127, 688128, 16, stream=stream0)
        buf130 = empty_strided_cuda((128, 7, 12, 64), (5376, 768, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm_backward, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_layer_norm_native_layer_norm_backward_56.run(buf127, primals_15, primals_7, getitem_1, rsqrt, buf130, 10752, 64, stream=stream0)
        del primals_15
        buf131 = empty_strided_cuda((64, 84), (1, 64), torch.float32)
        buf133 = empty_strided_cuda((64, 84), (1, 64), torch.float32)
        # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_layer_norm_native_layer_norm_backward_57.run(buf127, primals_7, getitem_1, rsqrt, buf131, buf133, 5376, 128, stream=stream0)
        del buf127
        del getitem_1
        del primals_7
        del rsqrt
        buf132 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_layer_norm_native_layer_norm_backward_58.run(buf131, buf132, 64, 84, stream=stream0)
        del buf131
        buf134 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_layer_norm_native_layer_norm_backward_58.run(buf133, buf134, 64, 84, stream=stream0)
        del buf133
        buf135 = empty_strided_cuda((896, 64, 104), (6656, 104, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.unfold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_unfold_backward_59.run(buf135, 5963776, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.unfold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_unfold_backward_60.run(primals_3, rsqrt_1, buf113, primals_18, buf120, unfold, getitem_3, buf121, buf135, 57344, 192, stream=stream0)
        del buf113
        del buf120
        del buf121
        del getitem_3
        del primals_18
        del primals_3
        del rsqrt_1
        del unfold
        buf137 = empty_strided_cuda((896, 64, 96), (6144, 96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_61.run(buf135, buf137, 5505024, stream=stream0)
        del buf135
    return (None, None, None, None, buf137, None, buf130, None, buf119, None, None, None, None, None, buf132, buf134, buf126, buf123, buf125, buf117, buf118, buf109, buf110, None, buf103, None, buf85, None, buf101, buf102, buf93, buf94, buf88, None, buf82, buf81, buf65, None, buf73, buf74, None, buf62, None, buf47, None, buf54, buf55, buf43, buf44, buf36, buf37, buf17, buf19, buf12, buf13, buf5, buf6, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_2 = rand_strided((128, 7, 12, 16), (1344, 192, 16, 1), device='cuda:0', dtype=torch.bool)
    primals_3 = rand_strided((128, 7, 12, 16), (1344, 192, 16, 1), device='cuda:0', dtype=torch.bool)
    primals_4 = rand_strided((128, 7), (7, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((128, 7, 12, 64), (5376, 768, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((128, 7, 12, 4), (336, 48, 4, 1), device='cuda:0', dtype=torch.bool)
    primals_9 = rand_strided((128, 7, 12, 16, 4), (5376, 768, 64, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((128, 7, 12), (84, 12, 1), device='cuda:0', dtype=torch.bool)
    primals_28 = rand_strided((128, 7, 12, 4), (336, 48, 4, 1), device='cuda:0', dtype=torch.bool)
    primals_29 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((128, 7, 12), (84, 12, 1), device='cuda:0', dtype=torch.bool)
    primals_38 = rand_strided((128, 7, 12, 12), (1008, 144, 12, 1), device='cuda:0', dtype=torch.bool)
    primals_41 = rand_strided((128, 7, 12), (84, 12, 1), device='cuda:0', dtype=torch.bool)
    primals_45 = rand_strided((128, 7, 12, 10), (840, 120, 1, 12), device='cuda:0', dtype=torch.bool)
    primals_52 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    unfold = rand_strided((896, 64, 12, 16), (6656, 104, 8, 1), device='cuda:0', dtype=torch.float32)
    getitem_1 = rand_strided((128, 7, 12, 1), (84, 12, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt = rand_strided((128, 7, 12, 1), (84, 12, 1, 1), device='cuda:0', dtype=torch.float32)
    getitem_3 = rand_strided((128, 7, 12, 16, 1), (1344, 192, 16, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_1 = rand_strided((128, 7, 12, 16, 1), (1344, 192, 16, 1, 1), device='cuda:0', dtype=torch.float32)
    view_7 = rand_strided((172032, 143), (143, 1), device='cuda:0', dtype=torch.bfloat16)
    addmm_1 = rand_strided((172032, 64), (64, 1), device='cuda:0', dtype=torch.bfloat16)
    view_9 = rand_strided((172032, 64), (64, 1), device='cuda:0', dtype=torch.bfloat16)
    mul_11 = rand_strided((128, 7, 12, 4, 64), (21504, 3072, 256, 64, 1), device='cuda:0', dtype=torch.float32)
    view_11 = rand_strided((43008, 5), (5, 1), device='cuda:0', dtype=torch.bfloat16)
    unsqueeze_24 = rand_strided((128, 7, 12, 1, 4), (336, 48, 4, 4, 1), device='cuda:0', dtype=torch.float32)
    mul_14 = rand_strided((128, 7, 12, 12, 64), (64512, 9216, 768, 64, 1), device='cuda:0', dtype=torch.float32)
    view_13 = rand_strided((129024, 4), (4, 1), device='cuda:0', dtype=torch.bfloat16)
    unsqueeze_30 = rand_strided((128, 7, 12, 1, 12), (1008, 144, 12, 12, 1), device='cuda:0', dtype=torch.float32)
    view_15 = rand_strided((107520, 5), (5, 1), device='cuda:0', dtype=torch.bfloat16)
    unsqueeze_36 = rand_strided((128, 7, 12, 1, 10), (840, 120, 1, 120, 12), device='cuda:0', dtype=torch.float32)
    cat_4 = rand_strided((128, 7, 12, 26), (2184, 312, 26, 1), device='cuda:0', dtype=torch.bool)
    view_18 = rand_strided((279552, 64), (64, 1), device='cuda:0', dtype=torch.bfloat16)
    cat_5 = rand_strided((128, 7, 12, 16, 26), (34944, 4992, 416, 26, 1), device='cuda:0', dtype=torch.float32)
    div_3 = rand_strided((128, 7, 12, 8, 16, 27), (290304, 41472, 3456, 432, 27, 1), device='cuda:0', dtype=torch.float32)
    cat_7 = rand_strided((128, 7, 12, 16, 128), (172032, 24576, 2048, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_11 = rand_strided((128, 7, 12, 16, 1), (1344, 192, 16, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_5 = rand_strided((128, 7, 12, 16, 1), (1344, 192, 16, 1, 1), device='cuda:0', dtype=torch.float32)
    view_31 = rand_strided((172032, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    addmm_7 = rand_strided((172032, 64), (64, 1), device='cuda:0', dtype=torch.bfloat16)
    view_33 = rand_strided((172032, 64), (64, 1), device='cuda:0', dtype=torch.bfloat16)
    unsqueeze_41 = rand_strided((128, 7, 12, 1), (84, 12, 1, 1), device='cuda:0', dtype=torch.bool)
    permute_19 = rand_strided((1, 64), (64, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_23 = rand_strided((64, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_28 = rand_strided((86016, 26, 16), (416, 1, 26), device='cuda:0', dtype=torch.bfloat16)
    permute_29 = rand_strided((86016, 8, 26), (208, 1, 8), device='cuda:0', dtype=torch.bfloat16)
    permute_30 = rand_strided((86016, 8, 16), (128, 1, 8), device='cuda:0', dtype=torch.bfloat16)
    permute_31 = rand_strided((86016, 26, 8), (208, 1, 26), device='cuda:0', dtype=torch.bfloat16)
    permute_34 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_39 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.bfloat16)
    mul_42 = rand_strided((128, 7, 12, 10, 64), (53760, 7680, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    div_8 = rand_strided((128, 7, 12, 10, 1), (840, 120, 10, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_47 = rand_strided((64, 4), (4, 1), device='cuda:0', dtype=torch.bfloat16)
    div_9 = rand_strided((128, 7, 12, 12, 1), (1008, 144, 12, 1, 1), device='cuda:0', dtype=torch.float32)
    div_10 = rand_strided((128, 7, 12, 4, 1), (336, 48, 4, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_54 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_58 = rand_strided((64, 143), (143, 1), device='cuda:0', dtype=torch.bfloat16)
    tangents_1 = rand_strided((128, 7, 12, 16), (1344, 192, 16, 1), device='cuda:0', dtype=torch.float32)
    tangents_2 = rand_strided((128, 7, 12, 8, 16, 27), (290304, 41472, 3456, 432, 27, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_2, primals_3, primals_4, primals_7, primals_8, primals_9, primals_15, primals_18, primals_24, primals_28, primals_29, primals_34, primals_38, primals_41, primals_45, primals_52, unfold, getitem_1, rsqrt, getitem_3, rsqrt_1, view_7, addmm_1, view_9, mul_11, view_11, unsqueeze_24, mul_14, view_13, unsqueeze_30, view_15, unsqueeze_36, cat_4, view_18, cat_5, div_3, cat_7, getitem_11, rsqrt_5, view_31, addmm_7, view_33, unsqueeze_41, permute_19, permute_23, permute_28, permute_29, permute_30, permute_31, permute_34, permute_39, mul_42, div_8, permute_47, div_9, div_10, permute_54, permute_58, tangents_1, tangents_2])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
