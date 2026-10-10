# AOT ID: ['0_forward']
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


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_variable_scaling_20261008/inductor_cache/4m/c4mrr7ptjtzmx2rcrtrcnow5uf7repbyjtiokaq5m46jikqehd7n.py
# Topologically Sorted Source Nodes: [pad], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   pad => constant_pad_nd
# Graph fragment:
#   %constant_pad_nd : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%primals_5, [0, 8], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_0 = async_compile.triton('triton_poi_fused_constant_pad_nd_0', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 17891328
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 104)
    x1 = xindex // 104
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 96, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x0 + 96*x1), tmp2, other=0.0)
    tl.store(out_ptr0 + (x2), tmp3, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_variable_scaling_20261008/inductor_cache/ze/cze3eq2op62kqkbu2paxch23x2cc7ogqtfzcwaqdosq3xxy7ytd5.py
# Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm => add_3, add_4, mul_3, mul_4, rsqrt, sub_2, var_mean
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%primals_7, [3]), kwargs = {correction: 0, keepdim: True})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_3,), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_7, %getitem_1), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, %primals_15), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %primals_16), kwargs = {})
triton_per_fused_native_layer_norm_1 = async_compile.triton('triton_per_fused_native_layer_norm_1', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 4, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp24 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp6 = tl.where(xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = (tmp7 / tmp9)
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp15 = tl.where(xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = 64.0
    tmp18 = (tmp16 / tmp17)
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.rsqrt(tmp20)
    tmp22 = tmp0 - tmp10
    tmp23 = tmp22 * tmp21
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp21, xmask)
    tl.store(out_ptr1 + (r0_1 + 64*x0), tmp27, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_variable_scaling_20261008/inductor_cache/zs/czs5aix7byk4khfjtjx6b5may6gv3nuvoeemwoawv6jcwpfpap2n.py
# Topologically Sorted Source Nodes: [invert, context, layer_norm_1], Original ATen: [aten.bitwise_not, aten.masked_fill, aten.native_layer_norm]
# Source node to ATen node mapping:
#   context => clone, full_default, where
#   invert => bitwise_not
#   layer_norm_1 => add_6, add_7, mul_6, mul_7, rsqrt_1, sub_3, var_mean_1
# Graph fragment:
#   %bitwise_not : [num_users=2] = call_function[target=torch.ops.aten.bitwise_not.default](args = (%unsqueeze,), kwargs = {})
#   %full_default : [num_users=12] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_not, %full_default, %view), kwargs = {})
#   %clone : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%where,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone, [4]), kwargs = {correction: 0, keepdim: True})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_6,), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone, %getitem_3), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_1), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, %primals_18), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, %primals_19), kwargs = {})
triton_per_fused_bitwise_not_masked_fill_native_layer_norm_2 = async_compile.triton('triton_per_fused_bitwise_not_masked_fill_native_layer_norm_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i1', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_bitwise_not_masked_fill_native_layer_norm_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 4, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_bitwise_not_masked_fill_native_layer_norm_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 516096
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
    x4 = xindex
    r0_3 = r0_index
    x0 = (xindex % 16)
    x1 = ((xindex // 16) % 12)
    x2 = xindex // 192
    x5 = xindex // 16
    tmp0 = tl.load(in_ptr0 + (x4), None, eviction_policy='evict_last').to(tl.int1)
    tmp2 = tl.load(in_ptr1 + (x0 + 8*x1 + 104*r0_3 + 6656*x2), None)
    tmp25 = tl.load(in_ptr2 + (r0_3), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr3 + (r0_3), None, eviction_policy='evict_last')
    tmp1 = tmp0 == 0
    tmp3 = 0.0
    tmp4 = tl.where(tmp1, tmp3, tmp2)
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp9 = tl.sum(tmp7, 1)[:, None]
    tmp10 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = (tmp9 / tmp11)
    tmp13 = tmp5 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp17 = tl.sum(tmp15, 1)[:, None]
    tmp18 = 64.0
    tmp19 = (tmp17 / tmp18)
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tmp23 = tmp4 - tmp12
    tmp24 = tmp23 * tmp22
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x4), tmp22, None)
    tl.store(out_ptr1 + (r0_3 + 143*x0 + 2304*x5), tmp28, None)
    tl.store(out_ptr0 + (x4), tmp12, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_variable_scaling_20261008/inductor_cache/rl/crle3eveye4vswpenawp7gaafagffu2mdgmwzpdhfshsmrjwn3fe.py
# Topologically Sorted Source Nodes: [cat_1], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_1 => cat_2
# Graph fragment:
#   %cat_2 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_7, %expand_2, %clone_1, %unsqueeze_12, %unsqueeze_18, %primals_9, %convert_element_type_2], -1), kwargs = {})
triton_poi_fused_cat_3 = async_compile.triton('triton_poi_fused_cat_3', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33030144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 64)
    x2 = xindex // 1024
    x1 = ((xindex // 64) % 16)
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x2), None, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0 + 143*x1 + 2304*x2), tmp0, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_variable_scaling_20261008/inductor_cache/de/cde65epwamla33cj572bx56b3yrltgkkualotcco3x7f62se6qwf.py
# Topologically Sorted Source Nodes: [invert, context, visible], Original ATen: [aten.bitwise_not, aten.masked_fill]
# Source node to ATen node mapping:
#   context => full_default
#   invert => bitwise_not
#   visible => clone_1, where_1
# Graph fragment:
#   %bitwise_not : [num_users=2] = call_function[target=torch.ops.aten.bitwise_not.default](args = (%unsqueeze,), kwargs = {})
#   %full_default : [num_users=12] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_not, %full_default, %view_1), kwargs = {})
#   %clone_1 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%where_1,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_bitwise_not_masked_fill_4 = async_compile.triton('triton_poi_fused_bitwise_not_masked_fill_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 524288, 'x': 8}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_not_masked_fill_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_bitwise_not_masked_fill_4(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 516096
    xnumel = 5
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y4 = yindex
    y0 = (yindex % 16)
    y1 = ((yindex // 16) % 12)
    x3 = xindex
    y2 = yindex // 192
    y5 = yindex // 16
    tmp0 = tl.load(in_ptr0 + (y4), ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tmp0 == 0
    tmp2 = y0 + 8*y1
    tmp3 = tl.full([1, 1], 96, tl.int64)
    tmp4 = tmp2 < tmp3
    tmp5 = tl.load(in_ptr1 + (y0 + 8*y1 + 96*x3 + 480*y2), ymask & xmask & tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = 0.0
    tmp7 = tl.where(tmp1, tmp6, tmp5)
    tl.store(out_ptr0 + (x3 + 143*y0 + 2304*y5), tmp7, ymask & xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_variable_scaling_20261008/inductor_cache/np/cnpfs7i36gc5uzrk7b3ler6hro4mnsjtajsndnddtvbg6xndgi6f.py
# Topologically Sorted Source Nodes: [cat_1], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_1 => cat_2
# Graph fragment:
#   %cat_2 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_7, %expand_2, %clone_1, %unsqueeze_12, %unsqueeze_18, %primals_9, %convert_element_type_2], -1), kwargs = {})
triton_poi_fused_cat_5 = async_compile.triton('triton_poi_fused_cat_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_5(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 516096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 16)
    x1 = xindex // 16
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 8.0
    tmp3 = tmp1 < tmp2
    tmp4 = 0.13333333333333333
    tmp5 = tmp1 * tmp4
    tmp6 = -1.0
    tmp7 = tmp5 + tmp6
    tmp8 = 15 + ((-1)*x0)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp9 * tmp4
    tmp11 = 1.0
    tmp12 = tmp11 - tmp10
    tmp13 = tl.where(tmp3, tmp7, tmp12)
    tl.store(out_ptr0 + (143*x0 + 2304*x1), tmp13, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_variable_scaling_20261008/inductor_cache/e3/ce3gj72hnmbgb6wo22ovebfwkf3gon6ptiwqe7olgu26nvaznpa7.py
# Topologically Sorted Source Nodes: [cat_1], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_1 => cat_2
# Graph fragment:
#   %cat_2 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_7, %expand_2, %clone_1, %unsqueeze_12, %unsqueeze_18, %primals_9, %convert_element_type_2], -1), kwargs = {})
triton_poi_fused_cat_6 = async_compile.triton('triton_poi_fused_cat_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_6(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 516096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x2 = xindex // 192
    x0 = (xindex % 16)
    x4 = xindex // 16
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tmp1 * tmp1
    tmp4 = 0.010000000000000002
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = (tmp2 / tmp6)
    tl.store(out_ptr0 + (143*x0 + 2304*x4), tmp7, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_variable_scaling_20261008/inductor_cache/la/clas25ivgefumnemp3h342d2awzyl3ey7dr3zxwzbk3zbl3rls7p.py
# Topologically Sorted Source Nodes: [cat_1], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_1 => cat_2
# Graph fragment:
#   %cat_2 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_7, %expand_2, %clone_1, %unsqueeze_12, %unsqueeze_18, %primals_9, %convert_element_type_2], -1), kwargs = {})
triton_poi_fused_cat_7 = async_compile.triton('triton_poi_fused_cat_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2064384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 4)
    x1 = ((xindex // 4) % 16)
    x2 = xindex // 64
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tl.store(out_ptr0 + (x0 + 143*x1 + 2304*x2), tmp0, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_variable_scaling_20261008/inductor_cache/tp/ctppitskpst3kuna5o2imndpqhz6zvzgvbmofyrnsdz6yszehm6k.py
# Topologically Sorted Source Nodes: [to], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   to => convert_element_type_2
# Graph fragment:
#   %convert_element_type_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%expand, torch.float32), kwargs = {})
triton_poi_fused__to_copy_8 = async_compile.triton('triton_poi_fused__to_copy_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2064384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 4)
    x2 = xindex // 64
    x1 = ((xindex // 4) % 16)
    tmp0 = tl.load(in_ptr0 + (x0 + 4*x2), None, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0 + 143*x1 + 2304*x2), tmp1, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_variable_scaling_20261008/inductor_cache/wk/cwktjw3k7ybsunaz4kbuc5hxvvgo4fr4fljl2nnrj3mflf3hxtzi.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.view]
# Source node to ATen node mapping:
#   input_1 => view_7
# Graph fragment:
#   %view_7 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%cat_2, [516096, 143]), kwargs = {})
triton_poi_fused_view_9 = async_compile.triton('triton_poi_fused_view_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_view_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 73801728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 143)
    x1 = xindex // 143
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 143*((x1 % 16)) + 2304*(x1 // 16)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_variable_scaling_20261008/inductor_cache/d3/cd3dwnhox22xfptdgct2ranexs2getcizmwand22r2dijg6pnbfh.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   input_1 => constant_pad_nd_default_2
# Graph fragment:
#   %constant_pad_nd_default_2 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view_7, [0, 1, 0, 0]), kwargs = {})
triton_poi_fused_addmm_10 = async_compile.triton('triton_poi_fused_addmm_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_10(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 74317824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 144)
    x1 = xindex // 144
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 143, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x0 + 143*x1), tmp2, other=0.0)
    tl.store(out_ptr0 + (x2), tmp3, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_variable_scaling_20261008/inductor_cache/yt/cyt2ko4ugh6nebccxlgdhahzm7r7qb6xvvs3zkbkadxa2rekmiot.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   input_1 => constant_pad_nd_default_3
# Graph fragment:
#   %constant_pad_nd_default_3 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%permute_5, [0, 0, 0, 1]), kwargs = {})
triton_poi_fused_addmm_11 = async_compile.triton('triton_poi_fused_addmm_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_11(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp3 = tl.load(in_ptr0 + (x0 + 143*x1), xmask & tmp2, other=0.0)
    tl.store(out_ptr0 + (x2), tmp3, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_variable_scaling_20261008/inductor_cache/dq/cdqdf4zcczdsrvl2fbsyuws6r3ehloeivioc6q23eyxs5ays7ing.py
# Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   input_2 => add_8, erf, mul_10, mul_8, mul_9
# Graph fragment:
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_8, 0.5), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_8, 0.7071067811865476), kwargs = {})
#   %erf : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_9,), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_8, %add_8), kwargs = {})
triton_poi_fused_gelu_12 = async_compile.triton('triton_poi_fused_gelu_12', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_12(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33030144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp0 * tmp3
    tmp5 = libdevice.erf(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tl.store(out_ptr0 + (x0), tmp8, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_variable_scaling_20261008/inductor_cache/ew/cewi5qhekeihtu5lu77hf4pe7myfylxxlle7salnsx7z4mohu3oq.py
# Topologically Sorted Source Nodes: [context, and_, valid, raw, layer_norm_2, weights], Original ATen: [aten.masked_fill, aten.bitwise_and, aten.where, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   and_ => bitwise_and
#   context => full_default
#   layer_norm_2 => add_9, mul_11, rsqrt_2, sub_4, var_mean_2
#   raw => where_5
#   valid => bitwise_and_1
#   weights => where_7
# Graph fragment:
#   %full_default : [num_users=12] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %bitwise_and : [num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%primals_28, %unsqueeze_19), kwargs = {})
#   %bitwise_and_1 : [num_users=3] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and, %unsqueeze_20), kwargs = {})
#   %where_5 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_21, %primals_25, %full_default), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%where_5, [4]), kwargs = {correction: 0, keepdim: True})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_4, 1e-05), kwargs = {})
#   %rsqrt_2 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where_5, %getitem_5), kwargs = {})
#   %mul_11 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_2), kwargs = {})
#   %where_7 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and_1, %primals_27, %full_default), kwargs = {})
#   %div_10 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_2, 64), kwargs = {})
triton_per_fused_bitwise_and_masked_fill_native_layer_norm_native_layer_norm_backward_where_13 = async_compile.triton('triton_per_fused_bitwise_and_masked_fill_native_layer_norm_native_layer_norm_backward_where_13', '''
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
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*i1', 'in_ptr2': '*i1', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_bitwise_and_masked_fill_native_layer_norm_native_layer_norm_backward_where_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_bitwise_and_masked_fill_native_layer_norm_native_layer_norm_backward_where_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    x3 = xindex
    x1 = xindex // 4
    r0_2 = r0_index
    tmp0 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp3 = tl.load(in_ptr2 + (1 + 4*x1), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp5 = tl.load(in_ptr3 + (r0_2 + 64*x3), xmask, other=0.0)
    tmp33 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 & tmp1
    tmp4 = tmp2 & tmp3
    tmp6 = 0.0
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp10 = tl.where(xmask, tmp8, 0)
    tmp11 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp13 = tl.where(xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp15 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = (tmp14 / tmp16)
    tmp18 = tmp8 - tmp17
    tmp19 = tmp18 * tmp18
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
    tmp22 = tl.where(xmask, tmp20, 0)
    tmp23 = tl.sum(tmp22, 1)[:, None]
    tmp24 = tmp7 - tmp17
    tmp25 = 64.0
    tmp26 = (tmp23 / tmp25)
    tmp27 = 1e-05
    tmp28 = tmp26 + tmp27
    tmp29 = libdevice.rsqrt(tmp28)
    tmp30 = tmp24 * tmp29
    tmp31 = 0.015625
    tmp32 = tmp29 * tmp31
    tmp34 = tl.where(tmp4, tmp33, tmp6)
    tl.store(out_ptr2 + (r0_2 + 64*x3), tmp30, xmask)
    tl.store(out_ptr3 + (x3), tmp32, xmask)
    tl.store(out_ptr4 + (x3), tmp34, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_variable_scaling_20261008/inductor_cache/jx/cjxorwtiuutqps3mplespfuhtap2a4atelerrue75afx23cpecdg.py
# Topologically Sorted Source Nodes: [context, quality], Original ATen: [aten.masked_fill, aten.where]
# Source node to ATen node mapping:
#   context => full_default
#   quality => where_6
# Graph fragment:
#   %full_default : [num_users=12] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_6 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_21, %primals_26, %full_default), kwargs = {})
triton_poi_fused_masked_fill_where_14 = async_compile.triton('triton_poi_fused_masked_fill_where_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*i1', 'in_ptr2': '*i1', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_masked_fill_where_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_masked_fill_where_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 645120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex // 5
    x6 = xindex // 20
    x7 = xindex
    x3 = xindex // 5040
    x8 = (xindex % 5040)
    tmp0 = tl.load(in_ptr0 + (x4), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x6), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp3 = tl.load(in_ptr2 + (1 + 4*x6), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp5 = tl.load(in_ptr3 + (x7), xmask)
    tmp2 = tmp0 & tmp1
    tmp4 = tmp2 & tmp3
    tmp6 = 0.0
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tl.store(out_ptr0 + (x8 + 5056*x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_variable_scaling_20261008/inductor_cache/67/c67wbudymeuizlplmvjggneadd5lqogyypox7lormh2z7ojghviu.py
# Topologically Sorted Source Nodes: [context, quality, linear_4], Original ATen: [aten.masked_fill, aten.where, aten.view]
# Source node to ATen node mapping:
#   context => full_default
#   linear_4 => view_11
#   quality => where_6
# Graph fragment:
#   %full_default : [num_users=12] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_6 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_21, %primals_26, %full_default), kwargs = {})
#   %view_11 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%where_6, [129024, 5]), kwargs = {})
triton_poi_fused_masked_fill_view_where_15 = async_compile.triton('triton_poi_fused_masked_fill_view_where_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_masked_fill_view_where_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_masked_fill_view_where_15(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 645120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 5)
    x1 = xindex // 5
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 5*((x1 % 1008)) + 5056*(x1 // 1008)), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_variable_scaling_20261008/inductor_cache/tf/ctfpg5vybcr3vipjjeekzfrnbjq3rdwmvwb2hpt77uhs2tptxacq.py
# Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   linear_4 => constant_pad_nd_default
# Graph fragment:
#   %constant_pad_nd_default : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view_11, [0, 3, 0, 0]), kwargs = {})
triton_poi_fused_addmm_16 = async_compile.triton('triton_poi_fused_addmm_16', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_16(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1032192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 8)
    x1 = xindex // 8
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 5, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x0 + 5*x1), tmp2, other=0.0)
    tl.store(out_ptr0 + (x2), tmp3, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_variable_scaling_20261008/inductor_cache/45/c45tq6nmiphd2cu4o2kf33zkya3htkbrezg4v3ftstd45t5w43az.py
# Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   linear_4 => constant_pad_nd_default_1
# Graph fragment:
#   %constant_pad_nd_default_1 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%permute_7, [0, 0, 0, 3]), kwargs = {})
triton_poi_fused_addmm_17 = async_compile.triton('triton_poi_fused_addmm_17', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_17(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 8)
    x1 = xindex // 8
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 5, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x0 + 5*x1), xmask & tmp2, other=0.0)
    tl.store(out_ptr0 + (x2), tmp3, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_variable_scaling_20261008/inductor_cache/36/c364hibjrfvdami2znmmnslbfwqvrl2t4vlevysxsb5rmx5v35xr.py
# Topologically Sorted Source Nodes: [context, and__2, valid_1, raw_1, layer_norm_3, weights_1], Original ATen: [aten.masked_fill, aten.bitwise_and, aten.where, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   and__2 => bitwise_and_2
#   context => full_default
#   layer_norm_3 => add_13, mul_14, rsqrt_3, sub_5, var_mean_3
#   raw_1 => where_8
#   valid_1 => bitwise_and_3
#   weights_1 => where_10
# Graph fragment:
#   %full_default : [num_users=12] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %bitwise_and_2 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%primals_38, %unsqueeze_25), kwargs = {})
#   %bitwise_and_3 : [num_users=3] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and_2, %unsqueeze_26), kwargs = {})
#   %where_8 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_27, %primals_35, %full_default), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%where_8, [4]), kwargs = {correction: 0, keepdim: True})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-05), kwargs = {})
#   %rsqrt_3 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_13,), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where_8, %getitem_7), kwargs = {})
#   %mul_14 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %rsqrt_3), kwargs = {})
#   %where_10 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and_3, %primals_37, %full_default), kwargs = {})
#   %div_9 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_3, 64), kwargs = {})
triton_per_fused_bitwise_and_masked_fill_native_layer_norm_native_layer_norm_backward_where_18 = async_compile.triton('triton_per_fused_bitwise_and_masked_fill_native_layer_norm_native_layer_norm_backward_where_18', '''
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
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*i1', 'in_ptr2': '*i1', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_bitwise_and_masked_fill_native_layer_norm_native_layer_norm_backward_where_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_bitwise_and_masked_fill_native_layer_norm_native_layer_norm_backward_where_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    x5 = xindex
    x6 = xindex // 12
    r0_3 = r0_index
    x2 = xindex // 3024
    x4 = (xindex % 3024)
    tmp0 = tl.load(in_ptr0 + (x5), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x6), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp3 = tl.load(in_ptr2 + (2 + 4*x6), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp5 = tl.load(in_ptr3 + (r0_3 + 64*x5), xmask, other=0.0)
    tmp33 = tl.load(in_ptr4 + (x5), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 & tmp1
    tmp4 = tmp2 & tmp3
    tmp6 = 0.0
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp10 = tl.where(xmask, tmp8, 0)
    tmp11 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp13 = tl.where(xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp15 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = (tmp14 / tmp16)
    tmp18 = tmp8 - tmp17
    tmp19 = tmp18 * tmp18
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
    tmp22 = tl.where(xmask, tmp20, 0)
    tmp23 = tl.sum(tmp22, 1)[:, None]
    tmp24 = tmp7 - tmp17
    tmp25 = 64.0
    tmp26 = (tmp23 / tmp25)
    tmp27 = 1e-05
    tmp28 = tmp26 + tmp27
    tmp29 = libdevice.rsqrt(tmp28)
    tmp30 = tmp24 * tmp29
    tmp31 = 0.015625
    tmp32 = tmp29 * tmp31
    tmp34 = tl.where(tmp4, tmp33, tmp6)
    tl.store(out_ptr2 + (r0_3 + 64*x5), tmp30, xmask)
    tl.store(out_ptr3 + (x4 + 3040*x2), tmp32, xmask)
    tl.store(out_ptr4 + (x4 + 3040*x2), tmp34, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_variable_scaling_20261008/inductor_cache/4s/c4sd7j3a5bhkd5tu2tmb3y447kxy56rctk3rotlug3bthohabuw4.py
# Topologically Sorted Source Nodes: [context, quality_1], Original ATen: [aten.masked_fill, aten.where]
# Source node to ATen node mapping:
#   context => full_default
#   quality_1 => where_9
# Graph fragment:
#   %full_default : [num_users=12] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_9 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_27, %primals_36, %full_default), kwargs = {})
triton_poi_fused_masked_fill_where_19 = async_compile.triton('triton_poi_fused_masked_fill_where_19', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*i1', 'in_ptr2': '*i1', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_masked_fill_where_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_masked_fill_where_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp5 = tl.load(in_ptr3 + (x5), None)
    tmp2 = tmp0 & tmp1
    tmp4 = tmp2 & tmp3
    tmp6 = 0.0
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tl.store(out_ptr0 + (x5), tmp7, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_variable_scaling_20261008/inductor_cache/2k/c2kqj6bpklnkfe56j4vcy53fxosmfjglb4lhtvrd67cme7wulvtr.py
# Topologically Sorted Source Nodes: [context, quality_2, linear_6], Original ATen: [aten.masked_fill, aten.where, aten.clone]
# Source node to ATen node mapping:
#   context => full_default
#   linear_6 => clone_4
#   quality_2 => where_12
# Graph fragment:
#   %full_default : [num_users=12] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_12 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_33, %primals_43, %full_default), kwargs = {})
#   %clone_4 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%where_12,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_masked_fill_where_20 = async_compile.triton('triton_poi_fused_clone_masked_fill_where_20', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*i1', 'in_ptr2': '*i1', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_masked_fill_where_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_masked_fill_where_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1612800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 5) % 10)
    x2 = ((xindex // 50) % 12)
    x5 = xindex // 600
    x7 = xindex // 50
    x8 = xindex
    x4 = xindex // 12600
    x9 = (xindex % 12600)
    tmp0 = tl.load(in_ptr0 + (x2 + 12*x1 + 120*x5), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x7), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp3 = tl.load(in_ptr2 + (3 + 4*x7), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp5 = tl.load(in_ptr3 + (x8), xmask)
    tmp2 = tmp0 & tmp1
    tmp4 = tmp2 & tmp3
    tmp6 = 0.0
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tl.store(out_ptr0 + (x9 + 12608*x4), tmp7, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_variable_scaling_20261008/inductor_cache/io/cio5dgf4uu3walkiwz7glfrnx4kiac7cjf5a6mvdfjidblazkp3p.py
# Topologically Sorted Source Nodes: [context, quality_2, linear_6], Original ATen: [aten.masked_fill, aten.where, aten.clone, aten._unsafe_view]
# Source node to ATen node mapping:
#   context => full_default
#   linear_6 => clone_4, view_15
#   quality_2 => where_12
# Graph fragment:
#   %full_default : [num_users=12] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_12 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_33, %primals_43, %full_default), kwargs = {})
#   %clone_4 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%where_12,), kwargs = {memory_format: torch.contiguous_format})
#   %view_15 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_4, [322560, 5]), kwargs = {})
triton_poi_fused__unsafe_view_clone_masked_fill_where_21 = async_compile.triton('triton_poi_fused__unsafe_view_clone_masked_fill_where_21', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_masked_fill_where_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_clone_masked_fill_where_21(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1612800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 5)
    x1 = xindex // 5
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 5*((x1 % 2520)) + 12608*(x1 // 2520)), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_variable_scaling_20261008/inductor_cache/77/c77ahelt27g677wjhexemkxfxdvinbabjimsm2hsfeafuqv4eufk.py
# Topologically Sorted Source Nodes: [context, and__4, valid_2, raw_2, layer_norm_4, linear_6, add_7, weights_2, getitem_33], Original ATen: [aten.masked_fill, aten.bitwise_and, aten.where, aten.native_layer_norm, aten.add, aten.unsqueeze, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   add_7 => add_20
#   and__4 => bitwise_and_4
#   context => full_default
#   getitem_33 => unsqueeze_36
#   layer_norm_4 => add_17, add_18, clone_3, mul_17, mul_18, rsqrt_4, sub_6, var_mean_4
#   linear_6 => add_19
#   raw_2 => where_11
#   valid_2 => bitwise_and_5
#   weights_2 => where_13
# Graph fragment:
#   %full_default : [num_users=12] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %bitwise_and_4 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%primals_45, %unsqueeze_31), kwargs = {})
#   %bitwise_and_5 : [num_users=3] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and_4, %unsqueeze_32), kwargs = {})
#   %where_11 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_33, %primals_42, %full_default), kwargs = {})
#   %clone_3 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%where_11,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_3, [4]), kwargs = {correction: 0, keepdim: True})
#   %add_17 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_8, 1e-05), kwargs = {})
#   %rsqrt_4 : [num_users=3] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_17,), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_3, %getitem_9), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %rsqrt_4), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_17, %primals_29), kwargs = {})
#   %add_18 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_18, %primals_30), kwargs = {})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_16, %primals_47), kwargs = {})
#   %add_20 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_18, %add_19), kwargs = {})
#   %where_13 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and_5, %primals_44, %full_default), kwargs = {})
#   %unsqueeze_36 : [num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%where_13, 3), kwargs = {})
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where_11, %getitem_9), kwargs = {})
#   %mul_42 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %rsqrt_4), kwargs = {})
#   %div_8 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_4, 64), kwargs = {})
triton_per_fused_add_bitwise_and_masked_fill_native_layer_norm_native_layer_norm_backward_unsqueeze_where_22 = async_compile.triton('triton_per_fused_add_bitwise_and_masked_fill_native_layer_norm_native_layer_norm_backward_unsqueeze_where_22', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i1', 'in_ptr1': '*i1', 'in_ptr2': '*i1', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr5': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_bitwise_and_masked_fill_native_layer_norm_native_layer_norm_backward_unsqueeze_where_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 4, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_bitwise_and_masked_fill_native_layer_norm_native_layer_norm_backward_unsqueeze_where_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr2, out_ptr3, out_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    x0 = (xindex % 10)
    x1 = ((xindex // 10) % 12)
    x6 = xindex // 120
    x7 = xindex // 10
    r0_4 = r0_index
    x8 = xindex
    x3 = xindex // 2520
    x5 = (xindex % 2520)
    x2 = ((xindex // 120) % 21)
    tmp0 = tl.load(in_ptr0 + (x1 + 12*x0 + 120*x6), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x7), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp3 = tl.load(in_ptr2 + (3 + 4*x7), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp5 = tl.load(in_ptr3 + (r0_4 + 64*x8), xmask, other=0.0)
    tmp31 = tl.load(in_ptr4 + (r0_4), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr5 + (r0_4), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_out_ptr0 + (r0_4 + 64*x8), xmask, other=0.0)
    tmp36 = tl.load(in_ptr6 + (r0_4), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr7 + (x8), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 & tmp1
    tmp4 = tmp2 & tmp3
    tmp6 = 0.0
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp10 = tl.where(xmask, tmp8, 0)
    tmp11 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp13 = tl.where(xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp15 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = (tmp14 / tmp16)
    tmp18 = tmp8 - tmp17
    tmp19 = tmp18 * tmp18
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
    tmp22 = tl.where(xmask, tmp20, 0)
    tmp23 = tl.sum(tmp22, 1)[:, None]
    tmp24 = tmp7 - tmp17
    tmp25 = 64.0
    tmp26 = (tmp23 / tmp25)
    tmp27 = 1e-05
    tmp28 = tmp26 + tmp27
    tmp29 = libdevice.rsqrt(tmp28)
    tmp30 = tmp24 * tmp29
    tmp32 = tmp30 * tmp31
    tmp34 = tmp32 + tmp33
    tmp37 = tmp35 + tmp36
    tmp38 = tmp34 + tmp37
    tmp39 = 0.015625
    tmp40 = tmp29 * tmp39
    tmp42 = tl.where(tmp4, tmp41, tmp6)
    tl.store(in_out_ptr0 + (r0_4 + 64*x8), tmp38, xmask)
    tl.store(out_ptr2 + (r0_4 + 64*x1 + 768*x0 + 7680*x6), tmp30, xmask)
    tl.store(out_ptr3 + (x5 + 2528*x3), tmp40, xmask)
    tl.store(out_ptr5 + (x1 + 12*x0 + 120*x2 + 2528*x3), tmp42, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_variable_scaling_20261008/inductor_cache/w3/cw33xxmdpyyaos7giswyaf2irfcl56rv4eqqnk4dv42chdyape5p.py
# Topologically Sorted Source Nodes: [encoded_3], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   encoded_3 => cat_3
# Graph fragment:
#   %cat_3 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_12, %add_16, %add_21], -2), kwargs = {})
triton_poi_fused_cat_23 = async_compile.triton('triton_poi_fused_cat_23', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_23', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 14, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 53673984
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 64) % 26)
    x0 = (xindex % 64)
    x2 = xindex // 1664
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64*(x1) + 256*x2), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 * tmp6
    tmp8 = tl.load(in_ptr2 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.load(in_ptr3 + (x0 + 64*(x1) + 256*x2), tmp4, other=0.0)
    tmp11 = tl.load(in_ptr4 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 + tmp11
    tmp13 = tmp9 + tmp12
    tmp14 = tl.load(in_ptr5 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp4, tmp15, tmp16)
    tmp18 = tmp0 >= tmp3
    tmp19 = tl.full([1], 16, tl.int64)
    tmp20 = tmp0 < tmp19
    tmp21 = tmp18 & tmp20
    tmp22 = tl.load(in_ptr6 + (x0 + 64*((-4) + x1) + 768*x2), tmp21, other=0.0)
    tmp23 = tl.load(in_ptr1 + (x0), tmp21, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 * tmp23
    tmp25 = tl.load(in_ptr2 + (x0), tmp21, eviction_policy='evict_last', other=0.0)
    tmp26 = tmp24 + tmp25
    tmp27 = tl.load(in_ptr7 + (x0 + 64*((-4) + x1) + 768*x2), tmp21, other=0.0)
    tmp28 = tl.load(in_ptr8 + (x0), tmp21, eviction_policy='evict_last', other=0.0)
    tmp29 = tmp27 + tmp28
    tmp30 = tmp26 + tmp29
    tmp31 = tl.load(in_ptr5 + (64 + x0), tmp21, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp30 + tmp31
    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
    tmp34 = tl.where(tmp21, tmp32, tmp33)
    tmp35 = tmp0 >= tmp19
    tmp36 = tl.full([1], 26, tl.int64)
    tmp37 = tmp0 < tmp36
    tmp38 = tl.load(in_ptr9 + (x0 + 64*((-16) + x1) + 640*x2), tmp35, other=0.0)
    tmp39 = tl.load(in_ptr5 + (128 + x0), tmp35, eviction_policy='evict_last', other=0.0)
    tmp40 = tmp38 + tmp39
    tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
    tmp42 = tl.where(tmp35, tmp40, tmp41)
    tmp43 = tl.where(tmp21, tmp34, tmp42)
    tmp44 = tl.where(tmp4, tmp17, tmp43)
    tl.store(out_ptr0 + (x4), tmp44, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_variable_scaling_20261008/inductor_cache/fz/cfzwjpg6xut6at7sl2nhc5w57jdzntug6rkjyt7eyiteowbu4zsm.py
# Topologically Sorted Source Nodes: [valid_3, any_1], Original ATen: [aten.cat, aten.any]
# Source node to ATen node mapping:
#   any_1 => any_1
#   valid_3 => cat_4
# Graph fragment:
#   %cat_4 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%bitwise_and_1, %bitwise_and_3, %bitwise_and_5], -1), kwargs = {})
#   %any_1 : [num_users=1] = call_function[target=torch.ops.aten.any.dim](args = (%cat_4, -1), kwargs = {})
triton_per_fused_any_cat_24 = async_compile.triton('triton_per_fused_any_cat_24', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 32},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*i1', 'in_ptr2': '*i1', 'in_ptr3': '*i1', 'in_ptr4': '*i1', 'in_ptr5': '*i1', 'in_ptr6': '*i1', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_any_cat_24', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 1, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_any_cat_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32256
    r0_numel = 26
    R0_BLOCK: tl.constexpr = 32
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x3 = xindex
    x0 = (xindex % 12)
    x1 = xindex // 12
    tmp0 = r0_2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (4*x3 + (r0_2)), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(x3, [XBLOCK, R0_BLOCK])), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp7 = tmp5 & tmp6
    tmp8 = tl.load(in_ptr2 + (tl.broadcast_to(1 + 4*x3, [XBLOCK, R0_BLOCK])), xmask & r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp9 = tmp7 & tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1, 1], 16, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tmp12 & tmp14
    tmp16 = tl.load(in_ptr3 + (12*x3 + ((-4) + r0_2)), xmask & r0_mask & tmp15, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp17 = tl.load(in_ptr4 + (tl.broadcast_to(x3, [XBLOCK, R0_BLOCK])), xmask & r0_mask & tmp15, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp18 = tmp16 & tmp17
    tmp19 = tl.load(in_ptr2 + (tl.broadcast_to(2 + 4*x3, [XBLOCK, R0_BLOCK])), xmask & r0_mask & tmp15, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp20 = tmp18 & tmp19
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp15, tmp20, tmp21)
    tmp23 = tmp0 >= tmp13
    tmp24 = tl.full([1, 1], 26, tl.int64)
    tmp25 = tmp0 < tmp24
    tmp26 = tl.load(in_ptr5 + (x0 + 12*((-16) + r0_2) + 120*x1), xmask & r0_mask & tmp23, other=0.0).to(tl.int1)
    tmp27 = tl.load(in_ptr6 + (tl.broadcast_to(x3, [XBLOCK, R0_BLOCK])), xmask & r0_mask & tmp23, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp28 = tmp26 & tmp27
    tmp29 = tl.load(in_ptr2 + (tl.broadcast_to(3 + 4*x3, [XBLOCK, R0_BLOCK])), xmask & r0_mask & tmp23, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp30 = tmp28 & tmp29
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp23, tmp30, tmp31)
    tmp33 = tl.where(tmp15, tmp22, tmp32)
    tmp34 = tl.where(tmp4, tmp11, tmp33)
    tmp35 = tmp34.to(tl.int64)
    tmp36 = (tmp35 != 0)
    tmp37 = tl.broadcast_to(tmp36, [XBLOCK, R0_BLOCK])
    tmp39 = tl.where(r0_mask & xmask, tmp37, False)
    tmp40 = triton_helpers.any(tmp39, 1)[:, None]
    tl.store(out_ptr0 + (r0_2 + 26*x3), tmp34, xmask & r0_mask)
    tl.store(out_ptr1 + (x3), tmp40, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_variable_scaling_20261008/inductor_cache/xm/cxmj73u7n23xbqza6aw3u434a5o2a7k3mql6kxocbwlb6cxcizol.py
# Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul => clone_6
# Graph fragment:
#   %clone_6 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_4,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_25 = async_compile.triton('triton_poi_fused_clone_25', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2097152, 'x': 32}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_25', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_25(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2064384
    xnumel = 26
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 64)
    y1 = yindex // 64
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 64*x2 + 1664*y1), ymask & xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + 26*y3), tmp2, xmask & ymask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_variable_scaling_20261008/inductor_cache/3m/c3mfyapqmmposorgtadhru5jdlrzkhv57zkhcbxh7htslrbjkdm7.py
# Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul => clone_5
# Graph fragment:
#   %clone_5 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_3,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_26 = async_compile.triton('triton_poi_fused_clone_26', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_26', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_26(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33030144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 8)
    x1 = ((xindex // 8) % 16)
    x2 = ((xindex // 128) % 8)
    x3 = xindex // 1024
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 8*x2 + 64*x1 + 1024*x3), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_variable_scaling_20261008/inductor_cache/jn/cjn543gqosd3mljal63xqvoaedkamqlcihpeqkrrbhvssvyxef4o.py
# Topologically Sorted Source Nodes: [mass], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   mass => cat_5
# Graph fragment:
#   %cat_5 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%mul_13, %mul_16, %mul_19], -1), kwargs = {})
triton_poi_fused_cat_27 = async_compile.triton('triton_poi_fused_cat_27', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_27', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_27(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 13418496
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 26)
    x5 = xindex // 26
    x6 = xindex // 416
    x4 = xindex // 104832
    x7 = ((xindex // 416) % 252)
    x2 = ((xindex // 416) % 12)
    x3 = ((xindex // 4992) % 21)
    x8 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (1 + 4*x5), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (4*x6 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 * tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 16, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tmp10 & tmp12
    tmp14 = tl.load(in_ptr0 + (2 + 4*x5), tmp13, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr2 + (12*x7 + 3040*x4 + ((-4) + x0)), tmp13, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp14 * tmp15
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp13, tmp16, tmp17)
    tmp19 = tmp0 >= tmp11
    tmp20 = tl.full([1], 26, tl.int64)
    tmp21 = tmp0 < tmp20
    tmp22 = tl.load(in_ptr0 + (3 + 4*x5), tmp19, eviction_policy='evict_last', other=0.0)
    tmp23 = tl.load(in_ptr3 + (x2 + 12*((-16) + x0) + 120*x3 + 2528*x4), tmp19, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 * tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp19, tmp24, tmp25)
    tmp27 = tl.where(tmp13, tmp18, tmp26)
    tmp28 = tl.where(tmp4, tmp9, tmp27)
    tl.store(out_ptr0 + (x8), tmp28, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_variable_scaling_20261008/inductor_cache/rv/crvwgns6usu5jq2hm3uzz4qln65wkqjeeioq4sf5ho2bebtqpz7r.py
# Topologically Sorted Source Nodes: [cat_5, attention], Original ATen: [aten.cat, aten._softmax]
# Source node to ATen node mapping:
#   attention => div_3
#   cat_5 => cat_6
# Graph fragment:
#   %cat_6 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%expand_5, %where_14], -1), kwargs = {})
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%cat_6, -1), kwargs = {})
#   %sub_tensor : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_6, %getitem_12), kwargs = {})
#   %exp_default : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor,), kwargs = {})
#   %div_3 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default, %getitem_13), kwargs = {})
triton_per_fused__softmax_cat_28 = async_compile.triton('triton_per_fused__softmax_cat_28', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i1', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_cat_28', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 4, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax_cat_28(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    r0_3 = r0_index
    x0 = (xindex % 16)
    x2 = xindex // 128
    x5 = xindex
    tmp0 = r0_3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (tl.broadcast_to(4*x0 + 64*x2, [XBLOCK, R0_BLOCK])), r0_mask & tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = 1e-08
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tl_math.log(tmp7)
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp4, tmp8, tmp9)
    tmp11 = tmp0 >= tmp3
    tmp12 = tl.full([1, 1], 27, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr1 + (26*x2 + ((-1) + r0_3)), r0_mask & tmp11, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp15 = tmp14 == 0
    tmp16 = tl.load(in_ptr2 + (26*x5 + ((-1) + r0_3)), r0_mask & tmp11, eviction_policy='evict_last', other=0.0)
    tmp17 = 0.35355339059327373
    tmp18 = tmp16 * tmp17
    tmp19 = tl.load(in_ptr3 + (26*x0 + 416*x2 + ((-1) + r0_3)), r0_mask & tmp11, eviction_policy='evict_last', other=0.0)
    tmp20 = 1e-08
    tmp21 = triton_helpers.maximum(tmp19, tmp20)
    tmp22 = tl_math.log(tmp21)
    tmp23 = tmp18 + tmp22
    tmp24 = float("-inf")
    tmp25 = tl.where(tmp15, tmp24, tmp23)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp11, tmp25, tmp26)
    tmp28 = tl.where(tmp4, tmp10, tmp27)
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, R0_BLOCK])
    tmp31 = tl.broadcast_to(tmp29, [XBLOCK, R0_BLOCK])
    tmp33 = tl.where(r0_mask, tmp31, float("-inf"))
    tmp34 = triton_helpers.max2(tmp33, 1)[:, None]
    tmp35 = tmp29 - tmp34
    tmp36 = tl_math.exp(tmp35)
    tmp37 = tl.broadcast_to(tmp36, [XBLOCK, R0_BLOCK])
    tmp39 = tl.where(r0_mask, tmp37, 0)
    tmp40 = tl.sum(tmp39, 1)[:, None]
    tmp41 = tmp28 - tmp34
    tmp42 = tl_math.exp(tmp41)
    tmp43 = (tmp42 / tmp40)
    tl.store(out_ptr2 + (r0_3 + 27*x5), tmp43, r0_mask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_variable_scaling_20261008/inductor_cache/5k/c5km7t4gr7cbe7l726yh5shyee5c7jbir4hwgydout3cxhr7ai7z.py
# Topologically Sorted Source Nodes: [matmul_1], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul_1 => clone_7
# Graph fragment:
#   %clone_7 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_7,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_29 = async_compile.triton('triton_poi_fused_clone_29', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_29', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_29(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 53673984
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 8)
    x1 = ((xindex // 8) % 26)
    x2 = ((xindex // 208) % 8)
    x3 = xindex // 1664
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 8*x2 + 64*x1 + 1664*x3), None)
    tmp1 = tl.load(in_ptr1 + (x0 + 8*x2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_variable_scaling_20261008/inductor_cache/kf/ckfe4oaz2yhv3lblzyoelbf6hccpoh6fc3ccql72th5e3zr33z3e.py
# Topologically Sorted Source Nodes: [features, input_4], Original ATen: [aten.cat, aten.native_layer_norm]
# Source node to ATen node mapping:
#   features => cat_7
#   input_4 => add_23, add_24, mul_20, mul_21, rsqrt_5, sub_8, var_mean_5
# Graph fragment:
#   %cat_7 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%view_10, %view_30], -1), kwargs = {})
#   %var_mean_5 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%cat_7, [4]), kwargs = {correction: 0, keepdim: True})
#   %add_23 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_10, 1e-05), kwargs = {})
#   %rsqrt_5 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_23,), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_7, %getitem_11), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %rsqrt_5), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_20, %primals_52), kwargs = {})
#   %add_24 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_21, %primals_53), kwargs = {})
triton_per_fused_cat_native_layer_norm_30 = async_compile.triton('triton_per_fused_cat_native_layer_norm_30', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_30', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 4, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_cat_native_layer_norm_30(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    r0_2 = r0_index
    x3 = xindex
    x0 = (xindex % 16)
    x1 = xindex // 16
    tmp31 = tl.load(in_ptr2 + (r0_2), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr3 + (r0_2), None, eviction_policy='evict_last')
    tmp0 = r0_2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (64*x3 + (r0_2)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1, 1], 128, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (8*x0 + 128*(((((-64) + r0_2) // 8) % 8)) + 1024*x1 + ((((-64) + r0_2) % 8))), tmp6, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp13 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
    tmp15 = tl.sum(tmp13, 1)[:, None]
    tmp16 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = (tmp15 / tmp17)
    tmp19 = tmp11 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, R0_BLOCK])
    tmp23 = tl.sum(tmp21, 1)[:, None]
    tmp24 = 128.0
    tmp25 = (tmp23 / tmp24)
    tmp26 = 1e-05
    tmp27 = tmp25 + tmp26
    tmp28 = libdevice.rsqrt(tmp27)
    tmp29 = tmp10 - tmp18
    tmp30 = tmp29 * tmp28
    tmp32 = tmp30 * tmp31
    tmp34 = tmp32 + tmp33
    tl.store(out_ptr0 + (r0_2 + 128*x3), tmp10, None)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp28, None)
    tl.store(out_ptr2 + (r0_2 + 128*x3), tmp34, None)
    tl.store(out_ptr1 + (x3), tmp18, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/performance/e23_variable_scaling_20261008/inductor_cache/43/c43ukmpx577uzfj67scquncv6gfduxvk66v3ahpr72odv6nx2maq.py
# Topologically Sorted Source Nodes: [context, invert_5, and__6, eligible, correction_1], Original ATen: [aten.masked_fill, aten.bitwise_not, aten.bitwise_and, aten.where]
# Source node to ATen node mapping:
#   and__6 => bitwise_and_6
#   context => full_default
#   correction_1 => where_15
#   eligible => bitwise_and_7
#   invert_5 => bitwise_not_5
# Graph fragment:
#   %full_default : [num_users=12] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %bitwise_not_5 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_not.default](args = (%primals_2,), kwargs = {})
#   %bitwise_and_6 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%primals_3, %bitwise_not_5), kwargs = {})
#   %bitwise_and_7 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and_6, %unsqueeze_41), kwargs = {})
#   %where_15 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and_7, %squeeze, %full_default), kwargs = {})
triton_poi_fused_bitwise_and_bitwise_not_masked_fill_where_31 = async_compile.triton('triton_poi_fused_bitwise_and_bitwise_not_masked_fill_where_31', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i1', 'in_ptr1': '*i1', 'in_ptr2': '*i1', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_and_bitwise_not_masked_fill_where_31', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_bitwise_and_bitwise_not_masked_fill_where_31(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 516096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 16
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x2), None).to(tl.int1)
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last').to(tl.int1)
    tmp6 = tl.load(in_out_ptr0 + (x2), None)
    tmp7 = tl.load(in_ptr3 + (0))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp2 = tmp1 == 0
    tmp3 = tmp0 & tmp2
    tmp5 = tmp3 & tmp4
    tmp9 = tmp6 + tmp8
    tmp10 = 0.0
    tmp11 = tl.where(tmp5, tmp9, tmp10)
    tl.store(in_out_ptr0 + (x2), tmp11, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57 = args
    args.clear()
    assert_size_stride(primals_1, (128, 21, 12, 64), (16128, 768, 64, 1))
    assert_size_stride(primals_2, (128, 21, 12, 16), (4032, 192, 16, 1))
    assert_size_stride(primals_3, (128, 21, 12, 16), (4032, 192, 16, 1))
    assert_size_stride(primals_4, (128, 21), (21, 1))
    assert_size_stride(primals_5, (2688, 64, 96), (6144, 96, 1))
    assert_size_stride(primals_6, (2688, 5, 96), (480, 96, 1))
    assert_size_stride(primals_7, (128, 21, 12, 64), (16128, 768, 64, 1))
    assert_size_stride(primals_8, (128, 21, 12, 4), (1008, 48, 4, 1))
    assert_size_stride(primals_9, (128, 21, 12, 16, 4), (16128, 768, 64, 4, 1))
    assert_size_stride(primals_10, (128, 21, 12, 64), (16128, 768, 64, 1))
    assert_size_stride(primals_11, (128, 21, 12, 64), (16128, 768, 64, 1))
    assert_size_stride(primals_12, (128, 21, 12, 64), (16128, 768, 64, 1))
    assert_size_stride(primals_13, (16, 64), (64, 1))
    assert_size_stride(primals_14, (16, ), (1, ))
    assert_size_stride(primals_15, (64, ), (1, ))
    assert_size_stride(primals_16, (64, ), (1, ))
    assert_size_stride(primals_17, (128, 21, 12, 16), (4032, 192, 16, 1))
    assert_size_stride(primals_18, (64, ), (1, ))
    assert_size_stride(primals_19, (64, ), (1, ))
    assert_size_stride(primals_20, (64, 143), (143, 1))
    assert_size_stride(primals_21, (64, ), (1, ))
    assert_size_stride(primals_22, (64, 64), (64, 1))
    assert_size_stride(primals_23, (64, ), (1, ))
    assert_size_stride(primals_24, (128, 21, 12), (252, 12, 1))
    assert_size_stride(primals_25, (128, 21, 12, 4, 64), (64512, 3072, 256, 64, 1))
    assert_size_stride(primals_26, (128, 21, 12, 4, 5), (5040, 240, 20, 5, 1))
    assert_size_stride(primals_27, (128, 21, 12, 4), (1008, 48, 4, 1))
    assert_size_stride(primals_28, (128, 21, 12, 4), (1008, 48, 4, 1))
    assert_size_stride(primals_29, (64, ), (1, ))
    assert_size_stride(primals_30, (64, ), (1, ))
    assert_size_stride(primals_31, (64, 5), (5, 1))
    assert_size_stride(primals_32, (64, ), (1, ))
    assert_size_stride(primals_33, (3, 64), (64, 1))
    assert_size_stride(primals_34, (128, 21, 12), (252, 12, 1))
    assert_size_stride(primals_35, (128, 21, 12, 12, 64), (193536, 9216, 768, 64, 1))
    assert_size_stride(primals_36, (128, 21, 12, 12, 4), (12096, 576, 48, 4, 1))
    assert_size_stride(primals_37, (128, 21, 12, 12), (3024, 144, 12, 1))
    assert_size_stride(primals_38, (128, 21, 12, 12), (3024, 144, 12, 1))
    assert_size_stride(primals_39, (64, 4), (4, 1))
    assert_size_stride(primals_40, (64, ), (1, ))
    assert_size_stride(primals_41, (128, 21, 12), (252, 12, 1))
    assert_size_stride(primals_42, (128, 21, 12, 10, 64), (161280, 7680, 640, 64, 1))
    assert_size_stride(primals_43, (128, 21, 12, 10, 5), (12600, 600, 50, 5, 1))
    assert_size_stride(primals_44, (128, 21, 12, 10), (2520, 120, 10, 1))
    assert_size_stride(primals_45, (128, 21, 12, 10), (2520, 120, 1, 12))
    assert_size_stride(primals_46, (64, 5), (5, 1))
    assert_size_stride(primals_47, (64, ), (1, ))
    assert_size_stride(primals_48, (64, 64), (64, 1))
    assert_size_stride(primals_49, (64, ), (1, ))
    assert_size_stride(primals_50, (64, 64), (64, 1))
    assert_size_stride(primals_51, (64, ), (1, ))
    assert_size_stride(primals_52, (128, ), (1, ))
    assert_size_stride(primals_53, (128, ), (1, ))
    assert_size_stride(primals_54, (64, 128), (128, 1))
    assert_size_stride(primals_55, (64, ), (1, ))
    assert_size_stride(primals_56, (1, 64), (64, 1))
    assert_size_stride(primals_57, (1, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((2688, 64, 104), (6656, 104, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pad], Original ATen: [aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_0.run(primals_5, buf0, 17891328, stream=stream0)
        del primals_5
        buf1 = empty_strided_cuda((128, 21, 12, 1), (252, 12, 1, 1), torch.float32)
        buf2 = empty_strided_cuda((128, 21, 12, 1), (252, 12, 1, 32256), torch.float32)
        buf4 = reinterpret_tensor(buf2, (128, 21, 12, 1), (252, 12, 1, 1), 0); del buf2  # reuse
        buf5 = empty_strided_cuda((128, 21, 12, 64), (16128, 768, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_layer_norm_1.run(buf4, primals_7, primals_15, primals_16, buf1, buf5, 32256, 64, stream=stream0)
        del primals_16
        buf6 = empty_strided_cuda((128, 21, 12, 16, 1), (4032, 192, 16, 1, 1), torch.float32)
        buf7 = empty_strided_cuda((128, 21, 12, 16, 1), (4032, 192, 16, 1, 516096), torch.float32)
        buf9 = reinterpret_tensor(buf7, (128, 21, 12, 16, 1), (4032, 192, 16, 1, 1), 0); del buf7  # reuse
        buf17 = empty_strided_cuda((128, 21, 12, 16, 143), (580608, 27648, 2304, 143, 1), torch.float32)
        buf10 = reinterpret_tensor(buf17, (128, 21, 12, 16, 64), (580608, 27648, 2304, 143, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [invert, context, layer_norm_1], Original ATen: [aten.bitwise_not, aten.masked_fill, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_bitwise_not_masked_fill_native_layer_norm_2.run(buf9, primals_3, buf0, primals_18, primals_19, buf6, buf10, 516096, 64, stream=stream0)
        del primals_19
        buf11 = reinterpret_tensor(buf17, (128, 21, 12, 16, 64), (580608, 27648, 2304, 143, 1), 64)  # alias
        # Topologically Sorted Source Nodes: [cat_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf5, buf11, 33030144, stream=stream0)
        del buf5
        buf12 = reinterpret_tensor(buf17, (128, 21, 12, 16, 5), (580608, 27648, 2304, 143, 1), 128)  # alias
        # Topologically Sorted Source Nodes: [invert, context, visible], Original ATen: [aten.bitwise_not, aten.masked_fill]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bitwise_not_masked_fill_4.run(primals_3, primals_6, buf12, 516096, 5, stream=stream0)
        del primals_6
        buf13 = reinterpret_tensor(buf17, (128, 21, 12, 16, 1), (580608, 27648, 2304, 143, 1), 133)  # alias
        # Topologically Sorted Source Nodes: [cat_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf13, 516096, stream=stream0)
        buf14 = reinterpret_tensor(buf17, (128, 21, 12, 16, 1), (580608, 27648, 2304, 143, 1), 134)  # alias
        # Topologically Sorted Source Nodes: [cat_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(primals_17, primals_4, buf14, 516096, stream=stream0)
        del primals_17
        buf15 = reinterpret_tensor(buf17, (128, 21, 12, 16, 4), (580608, 27648, 2304, 143, 1), 135)  # alias
        # Topologically Sorted Source Nodes: [cat_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(primals_9, buf15, 2064384, stream=stream0)
        buf16 = reinterpret_tensor(buf17, (128, 21, 12, 16, 4), (580608, 27648, 2304, 143, 1), 139)  # alias
        # Topologically Sorted Source Nodes: [to], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_8.run(primals_8, buf16, 2064384, stream=stream0)
        buf18 = empty_strided_cuda((516096, 143), (143, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_9.run(buf17, buf18, 73801728, stream=stream0)
        del buf10
        del buf11
        del buf12
        del buf13
        del buf14
        del buf15
        del buf16
        del buf17
        buf19 = empty_strided_cuda((516096, 144), (144, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_10.run(buf18, buf19, 74317824, stream=stream0)
        buf20 = empty_strided_cuda((144, 64), (1, 144), torch.float32)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_11.run(primals_20, buf20, 9216, stream=stream0)
        buf21 = empty_strided_cuda((516096, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_21, buf19, buf20, alpha=1, beta=1, out=buf21)
        del buf19
        del buf20
        del primals_21
        buf22 = empty_strided_cuda((128, 21, 12, 16, 64), (258048, 12288, 1024, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_12.run(buf21, buf22, 33030144, stream=stream0)
        buf23 = empty_strided_cuda((516096, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_23, reinterpret_tensor(buf22, (516096, 64), (64, 1), 0), reinterpret_tensor(primals_22, (64, 64), (1, 64), 0), alpha=1, beta=1, out=buf23)
        del primals_23
        buf27 = empty_strided_cuda((128, 21, 12, 4, 64), (64512, 3072, 256, 64, 1), torch.float32)
        buf77 = empty_strided_cuda((128, 21, 12, 4, 1), (1008, 48, 4, 1, 1), torch.float32)
        buf33 = empty_strided_cuda((128, 21, 12, 4), (1008, 48, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [context, and_, valid, raw, layer_norm_2, weights], Original ATen: [aten.masked_fill, aten.bitwise_and, aten.where, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_bitwise_and_masked_fill_native_layer_norm_native_layer_norm_backward_where_13.run(primals_28, primals_24, primals_8, primals_25, primals_27, buf27, buf77, buf33, 129024, 64, stream=stream0)
        del primals_25
        del primals_27
        buf28 = empty_strided_cuda((128, 21, 12, 4, 5), (5056, 240, 20, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [context, quality], Original ATen: [aten.masked_fill, aten.where]
        stream0 = get_raw_stream(0)
        triton_poi_fused_masked_fill_where_14.run(primals_28, primals_24, primals_8, primals_26, buf28, 645120, stream=stream0)
        del primals_26
        buf29 = empty_strided_cuda((129024, 5), (5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [context, quality, linear_4], Original ATen: [aten.masked_fill, aten.where, aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_masked_fill_view_where_15.run(buf28, buf29, 645120, stream=stream0)
        del buf28
        buf30 = empty_strided_cuda((129024, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_16.run(buf29, buf30, 1032192, stream=stream0)
        buf31 = empty_strided_cuda((8, 64), (1, 8), torch.float32)
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_17.run(primals_31, buf31, 512, stream=stream0)
        del primals_31
        buf32 = empty_strided_cuda((129024, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.addmm]
        extern_kernels.mm(buf30, buf31, out=buf32)
        del buf30
        del buf31
        buf37 = empty_strided_cuda((128, 21, 12, 12, 64), (193536, 9216, 768, 64, 1), torch.float32)
        buf76 = empty_strided_cuda((128, 21, 12, 12, 1), (3040, 144, 12, 1, 1), torch.float32)
        buf40 = empty_strided_cuda((128, 21, 12, 12), (3040, 144, 12, 1), torch.float32)
        # Topologically Sorted Source Nodes: [context, and__2, valid_1, raw_1, layer_norm_3, weights_1], Original ATen: [aten.masked_fill, aten.bitwise_and, aten.where, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_bitwise_and_masked_fill_native_layer_norm_native_layer_norm_backward_where_18.run(primals_38, primals_34, primals_8, primals_35, primals_37, buf37, buf76, buf40, 387072, 64, stream=stream0)
        del primals_35
        del primals_37
        buf38 = empty_strided_cuda((128, 21, 12, 12, 4), (12096, 576, 48, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [context, quality_1], Original ATen: [aten.masked_fill, aten.where]
        stream0 = get_raw_stream(0)
        triton_poi_fused_masked_fill_where_19.run(primals_38, primals_34, primals_8, primals_36, buf38, 1548288, stream=stream0)
        del primals_36
        buf39 = empty_strided_cuda((387072, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf38, (387072, 4), (4, 1), 0), reinterpret_tensor(primals_39, (4, 64), (1, 4), 0), out=buf39)
        buf44 = empty_strided_cuda((128, 21, 12, 10, 5), (12608, 600, 50, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [context, quality_2, linear_6], Original ATen: [aten.masked_fill, aten.where, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_masked_fill_where_20.run(primals_45, primals_41, primals_8, primals_43, buf44, 1612800, stream=stream0)
        del primals_43
        buf45 = empty_strided_cuda((322560, 5), (5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [context, quality_2, linear_6], Original ATen: [aten.masked_fill, aten.where, aten.clone, aten._unsafe_view]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_masked_fill_where_21.run(buf44, buf45, 1612800, stream=stream0)
        del buf44
        buf46 = empty_strided_cuda((322560, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten.mm]
        extern_kernels.mm(buf45, reinterpret_tensor(primals_46, (5, 64), (1, 5), 0), out=buf46)
        del primals_46
        buf47 = reinterpret_tensor(buf46, (128, 21, 12, 10, 64), (161280, 7680, 640, 64, 1), 0); del buf46  # reuse
        buf74 = empty_strided_cuda((128, 21, 12, 10, 64), (161280, 7680, 64, 768, 1), torch.float32)
        buf75 = empty_strided_cuda((128, 21, 12, 10, 1), (2528, 120, 10, 1, 1), torch.float32)
        buf49 = empty_strided_cuda((128, 21, 12, 1, 10), (2528, 120, 1, 120, 12), torch.float32)
        # Topologically Sorted Source Nodes: [context, and__4, valid_2, raw_2, layer_norm_4, linear_6, add_7, weights_2, getitem_33], Original ATen: [aten.masked_fill, aten.bitwise_and, aten.where, aten.native_layer_norm, aten.add, aten.unsqueeze, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_bitwise_and_masked_fill_native_layer_norm_native_layer_norm_backward_unsqueeze_where_22.run(buf47, primals_45, primals_41, primals_8, primals_42, primals_29, primals_30, primals_47, primals_44, buf74, buf75, buf49, 322560, 64, stream=stream0)
        del primals_42
        del primals_44
        del primals_47
        buf50 = empty_strided_cuda((128, 21, 12, 26, 64), (419328, 19968, 1664, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [encoded_3], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_23.run(buf27, primals_29, primals_30, buf32, primals_32, primals_33, buf37, buf39, primals_40, buf47, buf50, 53673984, stream=stream0)
        del buf32
        del buf39
        del buf47
        del primals_30
        del primals_32
        del primals_33
        del primals_40
        buf51 = empty_strided_cuda((128, 21, 12, 26), (6552, 312, 26, 1), torch.bool)
        buf72 = empty_strided_cuda((128, 21, 12), (252, 12, 1), torch.bool)
        # Topologically Sorted Source Nodes: [valid_3, any_1], Original ATen: [aten.cat, aten.any]
        stream0 = get_raw_stream(0)
        triton_per_fused_any_cat_24.run(primals_28, primals_24, primals_8, primals_38, primals_34, primals_45, primals_41, buf51, buf72, 32256, 26, stream=stream0)
        buf52 = empty_strided_cuda((838656, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf50, (838656, 64), (64, 1), 0), reinterpret_tensor(primals_48, (64, 64), (1, 64), 0), out=buf52)
        buf53 = empty_strided_cuda((838656, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf50, (838656, 64), (64, 1), 0), reinterpret_tensor(primals_50, (64, 64), (1, 64), 0), out=buf53)
        buf54 = empty_strided_cuda((128, 21, 12, 8, 8, 26), (419328, 19968, 1664, 208, 26, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_25.run(buf52, primals_49, buf54, 2064384, 26, stream=stream0)
        del primals_49
        buf55 = empty_strided_cuda((128, 21, 12, 8, 16, 8), (258048, 12288, 1024, 128, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_26.run(buf23, buf55, 33030144, stream=stream0)
        buf56 = empty_strided_cuda((258048, 16, 26), (416, 26, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf55, (258048, 16, 8), (128, 8, 1), 0), reinterpret_tensor(buf54, (258048, 8, 26), (208, 26, 1), 0), out=buf56)
        buf57 = empty_strided_cuda((128, 21, 12, 16, 26), (104832, 4992, 416, 26, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mass], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_27.run(primals_9, buf33, buf40, buf49, buf57, 13418496, stream=stream0)
        buf60 = empty_strided_cuda((128, 21, 12, 8, 16, 27), (870912, 41472, 3456, 432, 27, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_5, attention], Original ATen: [aten.cat, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_cat_28.run(primals_9, buf51, buf56, buf57, buf60, 4128768, 27, stream=stream0)
        del buf56
        buf61 = reinterpret_tensor(buf52, (128, 21, 12, 8, 26, 8), (419328, 19968, 1664, 208, 8, 1), 0); del buf52  # reuse
        # Topologically Sorted Source Nodes: [matmul_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_29.run(buf53, primals_51, buf61, 53673984, stream=stream0)
        del buf53
        del primals_51
        buf62 = empty_strided_cuda((258048, 16, 8), (128, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf60, (258048, 16, 26), (432, 27, 1), 1), reinterpret_tensor(buf61, (258048, 26, 8), (208, 8, 1), 0), out=buf62)
        buf63 = empty_strided_cuda((128, 21, 12, 16, 128), (516096, 24576, 2048, 128, 1), torch.float32)
        buf64 = empty_strided_cuda((128, 21, 12, 16, 1), (4032, 192, 16, 1, 1), torch.float32)
        buf65 = empty_strided_cuda((128, 21, 12, 16, 1), (4032, 192, 16, 1, 516096), torch.float32)
        buf67 = reinterpret_tensor(buf65, (128, 21, 12, 16, 1), (4032, 192, 16, 1, 1), 0); del buf65  # reuse
        buf68 = empty_strided_cuda((128, 21, 12, 16, 128), (516096, 24576, 2048, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features, input_4], Original ATen: [aten.cat, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_cat_native_layer_norm_30.run(buf67, buf23, buf62, primals_52, primals_53, buf63, buf64, buf68, 516096, 128, stream=stream0)
        del primals_53
        buf69 = reinterpret_tensor(buf62, (516096, 64), (64, 1), 0); del buf62  # reuse
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_55, reinterpret_tensor(buf68, (516096, 128), (128, 1), 0), reinterpret_tensor(primals_54, (128, 64), (1, 128), 0), alpha=1, beta=1, out=buf69)
        del primals_55
        buf70 = reinterpret_tensor(buf23, (128, 21, 12, 16, 64), (258048, 12288, 1024, 64, 1), 0); del buf23  # reuse
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_12.run(buf69, buf70, 33030144, stream=stream0)
        buf71 = empty_strided_cuda((516096, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf70, (516096, 64), (64, 1), 0), reinterpret_tensor(primals_56, (64, 1), (1, 64), 0), out=buf71)
        buf73 = reinterpret_tensor(buf71, (128, 21, 12, 16), (4032, 192, 16, 1), 0); del buf71  # reuse
        # Topologically Sorted Source Nodes: [context, invert_5, and__6, eligible, correction_1], Original ATen: [aten.masked_fill, aten.bitwise_not, aten.bitwise_and, aten.where]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bitwise_and_bitwise_not_masked_fill_where_31.run(buf73, primals_3, primals_2, buf72, primals_57, 516096, stream=stream0)
        del primals_57
    return (buf73, buf60, buf51, primals_2, primals_3, primals_4, primals_7, primals_8, primals_9, primals_15, primals_18, primals_24, primals_28, primals_29, primals_34, primals_38, primals_41, primals_45, primals_52, reinterpret_tensor(buf0, (2688, 64, 12, 16), (6656, 104, 8, 1), 0), buf1, buf4, buf6, buf9, buf18, buf21, reinterpret_tensor(buf22, (516096, 64), (64, 1), 0), buf27, buf29, reinterpret_tensor(buf33, (128, 21, 12, 1, 4), (1008, 48, 4, 4, 1), 0), buf37, reinterpret_tensor(buf38, (387072, 4), (4, 1), 0), reinterpret_tensor(buf40, (128, 21, 12, 1, 12), (3040, 144, 12, 0, 1), 0), buf45, buf49, buf51, reinterpret_tensor(buf50, (838656, 64), (64, 1), 0), buf57, buf60, buf63, buf64, buf67, reinterpret_tensor(buf68, (516096, 128), (128, 1), 0), buf69, reinterpret_tensor(buf70, (516096, 64), (64, 1), 0), reinterpret_tensor(buf72, (128, 21, 12, 1), (252, 12, 1, 1), 0), primals_56, primals_54, reinterpret_tensor(buf61, (258048, 8, 26), (208, 1, 8), 0), reinterpret_tensor(buf55, (258048, 8, 16), (128, 1, 8), 0), reinterpret_tensor(buf54, (258048, 26, 8), (208, 1, 26), 0), primals_50, primals_48, buf74, buf75, primals_39, buf76, buf77, primals_22, primals_20, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((128, 21, 12, 64), (16128, 768, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((128, 21, 12, 16), (4032, 192, 16, 1), device='cuda:0', dtype=torch.bool)
    primals_3 = rand_strided((128, 21, 12, 16), (4032, 192, 16, 1), device='cuda:0', dtype=torch.bool)
    primals_4 = rand_strided((128, 21), (21, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((2688, 64, 96), (6144, 96, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((2688, 5, 96), (480, 96, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((128, 21, 12, 64), (16128, 768, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((128, 21, 12, 4), (1008, 48, 4, 1), device='cuda:0', dtype=torch.bool)
    primals_9 = rand_strided((128, 21, 12, 16, 4), (16128, 768, 64, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((128, 21, 12, 64), (16128, 768, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((128, 21, 12, 64), (16128, 768, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((128, 21, 12, 64), (16128, 768, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((128, 21, 12, 16), (4032, 192, 16, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((64, 143), (143, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((128, 21, 12), (252, 12, 1), device='cuda:0', dtype=torch.bool)
    primals_25 = rand_strided((128, 21, 12, 4, 64), (64512, 3072, 256, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((128, 21, 12, 4, 5), (5040, 240, 20, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((128, 21, 12, 4), (1008, 48, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((128, 21, 12, 4), (1008, 48, 4, 1), device='cuda:0', dtype=torch.bool)
    primals_29 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((64, 5), (5, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((3, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((128, 21, 12), (252, 12, 1), device='cuda:0', dtype=torch.bool)
    primals_35 = rand_strided((128, 21, 12, 12, 64), (193536, 9216, 768, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((128, 21, 12, 12, 4), (12096, 576, 48, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((128, 21, 12, 12), (3024, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((128, 21, 12, 12), (3024, 144, 12, 1), device='cuda:0', dtype=torch.bool)
    primals_39 = rand_strided((64, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((128, 21, 12), (252, 12, 1), device='cuda:0', dtype=torch.bool)
    primals_42 = rand_strided((128, 21, 12, 10, 64), (161280, 7680, 640, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((128, 21, 12, 10, 5), (12600, 600, 50, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((128, 21, 12, 10), (2520, 120, 10, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((128, 21, 12, 10), (2520, 120, 1, 12), device='cuda:0', dtype=torch.bool)
    primals_46 = rand_strided((64, 5), (5, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((64, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((1, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
