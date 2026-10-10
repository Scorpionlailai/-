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


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/5j/c5jz3d2xohwatqmom465ffm24c5prkj55vfjsqweqn2vekhar3db.py
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
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5963776
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


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/no/cnov5bxnp5dnjby25apcja5njfa5odig57p7umttbe3idqnnfljs.py
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
    size_hints={'x': 16384, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 4, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/x4/cx45b3e6mph55dxofmxhkhd3qwsuylkgdehmzjwdod27was7cfk7.py
# Topologically Sorted Source Nodes: [invert, context, layer_norm_1], Original ATen: [aten.bitwise_not, aten.masked_fill, aten.native_layer_norm]
# Source node to ATen node mapping:
#   context => clone, full_default, where
#   invert => bitwise_not
#   layer_norm_1 => add_6, rsqrt_1, var_mean_1
# Graph fragment:
#   %bitwise_not : [num_users=2] = call_function[target=torch.ops.aten.bitwise_not.default](args = (%unsqueeze,), kwargs = {})
#   %full_default : [num_users=11] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_not, %full_default, %view), kwargs = {})
#   %clone : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%where,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone, [4]), kwargs = {correction: 0, keepdim: True})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_6,), kwargs = {})
triton_per_fused_bitwise_not_masked_fill_native_layer_norm_2 = async_compile.triton('triton_per_fused_bitwise_not_masked_fill_native_layer_norm_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i1', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_bitwise_not_masked_fill_native_layer_norm_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 4, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_bitwise_not_masked_fill_native_layer_norm_2(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    x4 = xindex
    r0_3 = r0_index
    x0 = (xindex % 16)
    x1 = ((xindex // 16) % 12)
    x2 = xindex // 192
    tmp0 = tl.load(in_ptr0 + (x4), None, eviction_policy='evict_last').to(tl.int1)
    tmp2 = tl.load(in_ptr1 + (x0 + 8*x1 + 104*r0_3 + 6656*x2), None)
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
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x4), tmp22, None)
    tl.store(out_ptr0 + (x4), tmp12, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/s2/cs2dzjkcnr3hl26vltg5mlkuoyp45qigbkqccdxzofi5u4uhr7ul.py
# Topologically Sorted Source Nodes: [cat_1, input_1], Original ATen: [aten.cat, aten._to_copy]
# Source node to ATen node mapping:
#   cat_1 => cat_2
#   input_1 => convert_element_type_15
# Graph fragment:
#   %cat_2 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_7, %expand_2, %clone_1, %unsqueeze_12, %unsqueeze_18, %primals_9, %convert_element_type_12], -1), kwargs = {})
#   %convert_element_type_15 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cat_2, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_cat_3 = async_compile.triton('triton_poi_fused__to_copy_cat_3', '''
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
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/45/c45lrvoxq3ihfhob2tsfsdw2oro7jr2umawteiihmllizxjtasrr.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy, aten.view]
# Source node to ATen node mapping:
#   input_1 => convert_element_type_15, view_7
# Graph fragment:
#   %convert_element_type_15 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cat_2, torch.bfloat16), kwargs = {})
#   %view_7 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_15, [172032, 143]), kwargs = {})
triton_poi_fused__to_copy_view_4 = async_compile.triton('triton_poi_fused__to_copy_view_4', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_view_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_view_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24600576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 143)
    x1 = xindex // 143
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 143*((x1 % 16)) + 2304*(x1 // 16)), None).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp0, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/7y/c7yykxx7rcxq2iugiluv5yjvk4s65x3k26oefpj7x54cmht23bc2.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_1 => convert_element_type_14
# Graph fragment:
#   %convert_element_type_14 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_20, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_5 = async_compile.triton('triton_poi_fused__to_copy_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/ee/ceehx2nb2np5q7paz5qsc4s57t7xc5z5mydkngvxjsoo2tgl5gke.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_1 => convert_element_type_13
# Graph fragment:
#   %convert_element_type_13 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_21, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_6 = async_compile.triton('triton_poi_fused__to_copy_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/sz/cszv5t6kvp27ntqoyekvklvbfc22tuv3kgdxulr5dx3z64olp43p.py
# Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   input_2 => add_8, convert_element_type_19, convert_element_type_20, erf, mul_10, mul_8, mul_9
# Graph fragment:
#   %convert_element_type_19 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_8, torch.float32), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_19, 0.5), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_19, 0.7071067811865476), kwargs = {})
#   %erf : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_9,), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_8, %add_8), kwargs = {})
#   %convert_element_type_20 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_10, torch.bfloat16), kwargs = {})
triton_poi_fused_gelu_7 = async_compile.triton('triton_poi_fused_gelu_7', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 11010048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = 0.7071067811865476
    tmp5 = tmp1 * tmp4
    tmp6 = libdevice.erf(tmp5)
    tmp7 = 1.0
    tmp8 = tmp6 + tmp7
    tmp9 = tmp3 * tmp8
    tmp10 = tmp9.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp10, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/fh/cfhqugpeksvv32ocwdhjdaansvet45zmgyphrt7cbywuuqikhgnu.py
# Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_3 => convert_element_type_22
# Graph fragment:
#   %convert_element_type_22 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_22, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_8 = async_compile.triton('triton_poi_fused__to_copy_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/md/cmd6rirsaa52k4xdgrwpmujxnwtoiwvr4xiz72xog6ixzjdugntx.py
# Topologically Sorted Source Nodes: [context, and_, valid, raw, layer_norm_2, weights], Original ATen: [aten.masked_fill, aten.bitwise_and, aten.where, aten.native_layer_norm]
# Source node to ATen node mapping:
#   and_ => bitwise_and
#   context => full_default
#   layer_norm_2 => add_9, mul_11, rsqrt_2, sub_4, var_mean_2
#   raw => where_5
#   valid => bitwise_and_1
#   weights => where_7
# Graph fragment:
#   %full_default : [num_users=11] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %bitwise_and : [num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%primals_28, %unsqueeze_19), kwargs = {})
#   %bitwise_and_1 : [num_users=3] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and, %unsqueeze_20), kwargs = {})
#   %where_5 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_21, %primals_25, %full_default), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%where_5, [4]), kwargs = {correction: 0, keepdim: True})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_4, 1e-05), kwargs = {})
#   %rsqrt_2 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where_5, %getitem_5), kwargs = {})
#   %mul_11 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_2), kwargs = {})
#   %where_7 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and_1, %primals_27, %full_default), kwargs = {})
triton_per_fused_bitwise_and_masked_fill_native_layer_norm_where_9 = async_compile.triton('triton_per_fused_bitwise_and_masked_fill_native_layer_norm_where_9', '''
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
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*i1', 'in_ptr2': '*i1', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_bitwise_and_masked_fill_native_layer_norm_where_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_bitwise_and_masked_fill_native_layer_norm_where_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    x3 = xindex
    x1 = xindex // 4
    r0_2 = r0_index
    tmp0 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp3 = tl.load(in_ptr2 + (1 + 4*x1), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp5 = tl.load(in_ptr3 + (r0_2 + 64*x3), xmask, other=0.0)
    tmp31 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
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
    tmp32 = tl.where(tmp4, tmp31, tmp6)
    tl.store(out_ptr2 + (r0_2 + 64*x3), tmp30, xmask)
    tl.store(out_ptr3 + (x3), tmp32, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/ea/ceaglfkzvp26ab2b5faeixpiax2tpsbjglw3q4wtebkccekq6rzh.py
# Topologically Sorted Source Nodes: [context, quality, linear_4], Original ATen: [aten.masked_fill, aten.where, aten._to_copy]
# Source node to ATen node mapping:
#   context => full_default
#   linear_4 => convert_element_type_28
#   quality => where_6
# Graph fragment:
#   %full_default : [num_users=11] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_6 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_21, %primals_26, %full_default), kwargs = {})
#   %convert_element_type_28 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_6, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_masked_fill_where_10 = async_compile.triton('triton_poi_fused__to_copy_masked_fill_where_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*i1', 'in_ptr2': '*i1', 'in_ptr3': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_masked_fill_where_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_masked_fill_where_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 215040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex // 5
    x6 = xindex // 20
    x7 = xindex
    x3 = xindex // 1680
    x8 = (xindex % 1680)
    tmp0 = tl.load(in_ptr0 + (x4), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x6), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp3 = tl.load(in_ptr2 + (1 + 4*x6), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp5 = tl.load(in_ptr3 + (x7), xmask)
    tmp2 = tmp0 & tmp1
    tmp4 = tmp2 & tmp3
    tmp6 = 0.0
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp7.to(tl.float32)
    tl.store(out_ptr0 + (x8 + 1728*x3), tmp8, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/2l/c2lltxxj4ulb5lgjptswtx77gfaks5caharvmvahyf5a3die3eo6.py
# Topologically Sorted Source Nodes: [context, quality, linear_4], Original ATen: [aten.masked_fill, aten.where, aten._to_copy, aten.view]
# Source node to ATen node mapping:
#   context => full_default
#   linear_4 => convert_element_type_28, view_11
#   quality => where_6
# Graph fragment:
#   %full_default : [num_users=11] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_6 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_21, %primals_26, %full_default), kwargs = {})
#   %convert_element_type_28 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_6, torch.bfloat16), kwargs = {})
#   %view_11 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_28, [43008, 5]), kwargs = {})
triton_poi_fused__to_copy_masked_fill_view_where_11 = async_compile.triton('triton_poi_fused__to_copy_masked_fill_view_where_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_masked_fill_view_where_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_masked_fill_view_where_11(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 215040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 5)
    x1 = xindex // 5
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 5*((x1 % 336)) + 1728*(x1 // 336)), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/ix/cixnisejssvco5ztfiaodw5okk5tx54oflbb6xkr3xfpvz7stwf7.py
# Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   linear_4 => convert_element_type_27
# Graph fragment:
#   %convert_element_type_27 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_31, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_12 = async_compile.triton('triton_poi_fused__to_copy_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_12(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/fg/cfgvucfalrmhn633vfsnwow5e5573zhx6d6csjgd6tgtwfcswn53.py
# Topologically Sorted Source Nodes: [context, and__2, valid_1, raw_1, layer_norm_3, weights_1], Original ATen: [aten.masked_fill, aten.bitwise_and, aten.where, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   and__2 => bitwise_and_2
#   context => full_default
#   layer_norm_3 => add_13, mul_14, rsqrt_3, sub_5, var_mean_3
#   raw_1 => where_8
#   valid_1 => bitwise_and_3
#   weights_1 => where_10
# Graph fragment:
#   %full_default : [num_users=11] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
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
    x1 = xindex // 12
    r0_2 = r0_index
    tmp0 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp3 = tl.load(in_ptr2 + (2 + 4*x1), xmask, eviction_policy='evict_last').to(tl.int1)
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


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/sv/csve7v3sfpadbvkr3elfxmm7j7almja42wvdj7gdpsakucco5mny.py
# Topologically Sorted Source Nodes: [context, quality_1, linear_5], Original ATen: [aten.masked_fill, aten.where, aten._to_copy]
# Source node to ATen node mapping:
#   context => full_default
#   linear_5 => convert_element_type_34
#   quality_1 => where_9
# Graph fragment:
#   %full_default : [num_users=11] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_9 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_27, %primals_36, %full_default), kwargs = {})
#   %convert_element_type_34 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_9, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_masked_fill_where_14 = async_compile.triton('triton_poi_fused__to_copy_masked_fill_where_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*i1', 'in_ptr2': '*i1', 'in_ptr3': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_masked_fill_where_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_masked_fill_where_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp5 = tl.load(in_ptr3 + (x5), None)
    tmp2 = tmp0 & tmp1
    tmp4 = tmp2 & tmp3
    tmp6 = 0.0
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp7.to(tl.float32)
    tl.store(out_ptr0 + (x5), tmp8, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/25/c2574vpbfagzwvdwfxd4kw65uamxp7a5n76va7wy2giyxd5ak57e.py
# Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten._to_copy, aten.t]
# Source node to ATen node mapping:
#   linear_5 => convert_element_type_33, permute_8
# Graph fragment:
#   %convert_element_type_33 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_39, torch.bfloat16), kwargs = {})
#   %permute_8 : [num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_33, [1, 0]), kwargs = {})
triton_poi_fused__to_copy_t_15 = async_compile.triton('triton_poi_fused__to_copy_t_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_t_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_t_15(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/og/cogvqbmm6f4kuep6m423kewadmtm7c3jlcc54ehucb7zcublzd2c.py
# Topologically Sorted Source Nodes: [context, quality_2, linear_6], Original ATen: [aten.masked_fill, aten.where, aten._to_copy, aten.clone]
# Source node to ATen node mapping:
#   context => full_default
#   linear_6 => clone_4, convert_element_type_40
#   quality_2 => where_12
# Graph fragment:
#   %full_default : [num_users=11] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_12 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_33, %primals_43, %full_default), kwargs = {})
#   %convert_element_type_40 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_12, torch.bfloat16), kwargs = {})
#   %clone_4 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convert_element_type_40,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused__to_copy_clone_masked_fill_where_16 = async_compile.triton('triton_poi_fused__to_copy_clone_masked_fill_where_16', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*i1', 'in_ptr2': '*i1', 'in_ptr3': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_clone_masked_fill_where_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_clone_masked_fill_where_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 537600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 5) % 10)
    x2 = ((xindex // 50) % 12)
    x5 = xindex // 600
    x7 = xindex // 50
    x8 = xindex
    x4 = xindex // 4200
    x9 = (xindex % 4200)
    tmp0 = tl.load(in_ptr0 + (x2 + 12*x1 + 120*x5), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x7), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp3 = tl.load(in_ptr2 + (3 + 4*x7), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp5 = tl.load(in_ptr3 + (x8), xmask)
    tmp2 = tmp0 & tmp1
    tmp4 = tmp2 & tmp3
    tmp6 = 0.0
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp7.to(tl.float32)
    tl.store(out_ptr0 + (x9 + 4224*x4), tmp8, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/43/c436wq5nj6cms3n6pdib4vhwnb4txts6siqlspx733q7z626fxqi.py
# Topologically Sorted Source Nodes: [context, quality_2, linear_6], Original ATen: [aten.masked_fill, aten.where, aten._to_copy, aten.clone, aten._unsafe_view]
# Source node to ATen node mapping:
#   context => full_default
#   linear_6 => clone_4, convert_element_type_40, view_15
#   quality_2 => where_12
# Graph fragment:
#   %full_default : [num_users=11] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_12 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_33, %primals_43, %full_default), kwargs = {})
#   %convert_element_type_40 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_12, torch.bfloat16), kwargs = {})
#   %clone_4 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convert_element_type_40,), kwargs = {memory_format: torch.contiguous_format})
#   %view_15 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_4, [107520, 5]), kwargs = {})
triton_poi_fused__to_copy__unsafe_view_clone_masked_fill_where_17 = async_compile.triton('triton_poi_fused__to_copy__unsafe_view_clone_masked_fill_where_17', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_view_clone_masked_fill_where_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_view_clone_masked_fill_where_17(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 537600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 5)
    x1 = xindex // 5
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 5*((x1 % 840)) + 4224*(x1 // 840)), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/uy/cuylxfitrruqycdbiogfp2pipqofzudd5swfsd2sjjtqmjbn6d53.py
# Topologically Sorted Source Nodes: [context, and__4, valid_2, raw_2, layer_norm_4, linear_6, add_7, weights_2, getitem_33], Original ATen: [aten.masked_fill, aten.bitwise_and, aten.where, aten.native_layer_norm, aten._to_copy, aten.add, aten.unsqueeze, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   add_7 => add_20
#   and__4 => bitwise_and_4
#   context => full_default
#   getitem_33 => unsqueeze_36
#   layer_norm_4 => add_17, add_18, clone_3, mul_17, mul_18, rsqrt_4, sub_6, var_mean_4
#   linear_6 => add_19, convert_element_type_38
#   raw_2 => where_11
#   valid_2 => bitwise_and_5
#   weights_2 => where_13
# Graph fragment:
#   %full_default : [num_users=11] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
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
#   %convert_element_type_38 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_47, torch.bfloat16), kwargs = {})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_16, %convert_element_type_38), kwargs = {})
#   %add_20 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_18, %add_19), kwargs = {})
#   %where_13 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and_5, %primals_44, %full_default), kwargs = {})
#   %unsqueeze_36 : [num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%where_13, 3), kwargs = {})
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where_11, %getitem_9), kwargs = {})
#   %mul_42 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %rsqrt_4), kwargs = {})
#   %div_8 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_4, 64), kwargs = {})
triton_per_fused__to_copy_add_bitwise_and_masked_fill_native_layer_norm_native_layer_norm_backward_unsqueeze_where_18 = async_compile.triton('triton_per_fused__to_copy_add_bitwise_and_masked_fill_native_layer_norm_native_layer_norm_backward_unsqueeze_where_18', '''
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
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*i1', 'in_ptr2': '*i1', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*bf16', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr6': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_bitwise_and_masked_fill_native_layer_norm_native_layer_norm_backward_unsqueeze_where_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 4, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_bitwise_and_masked_fill_native_layer_norm_native_layer_norm_backward_unsqueeze_where_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr2, out_ptr3, out_ptr4, out_ptr6, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    x0 = (xindex % 10)
    x1 = ((xindex // 10) % 12)
    x2 = xindex // 120
    x5 = xindex // 10
    r0_3 = r0_index
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x1 + 12*x0 + 120*x2), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x5), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp3 = tl.load(in_ptr2 + (3 + 4*x5), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp5 = tl.load(in_ptr3 + (r0_3 + 64*x4), xmask, other=0.0)
    tmp31 = tl.load(in_ptr4 + (r0_3), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr5 + (r0_3), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr6 + (r0_3 + 64*x4), xmask, other=0.0).to(tl.float32)
    tmp36 = tl.load(in_ptr7 + (r0_3), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr8 + (x4), xmask, eviction_policy='evict_last')
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
    tmp37 = tmp36.to(tl.float32)
    tmp38 = tmp35 + tmp37
    tmp39 = tmp38.to(tl.float32)
    tmp40 = tmp34 + tmp39
    tmp41 = 0.015625
    tmp42 = tmp29 * tmp41
    tmp44 = tl.where(tmp4, tmp43, tmp6)
    tl.store(out_ptr2 + (r0_3 + 64*x4), tmp40, xmask)
    tl.store(out_ptr3 + (r0_3 + 64*x1 + 768*x0 + 7680*x2), tmp30, xmask)
    tl.store(out_ptr4 + (x4), tmp42, xmask)
    tl.store(out_ptr6 + (x1 + 12*x0 + 120*x2), tmp44, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/7n/c7nzerggh23x2aeounykk3amicz3ak2boxubuu3cj2urhctlxuy7.py
# Topologically Sorted Source Nodes: [encoded_3, linear_7], Original ATen: [aten.cat, aten._to_copy]
# Source node to ATen node mapping:
#   encoded_3 => cat_3
#   linear_7 => convert_element_type_45
# Graph fragment:
#   %cat_3 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_12, %add_16, %add_21], -2), kwargs = {})
#   %convert_element_type_45 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cat_3, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_cat_19 = async_compile.triton('triton_poi_fused__to_copy_cat_19', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*bf16', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_cat_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 14, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_cat_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 17891328
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 64) % 26)
    x0 = (xindex % 64)
    x2 = xindex // 1664
    x3 = xindex
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
    tmp10 = tl.load(in_ptr3 + (x0 + 64*(x1) + 256*x2), tmp4, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_ptr4 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 + tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp9 + tmp14
    tmp16 = tl.load(in_ptr5 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp4, tmp17, tmp18)
    tmp20 = tmp0 >= tmp3
    tmp21 = tl.full([1], 16, tl.int64)
    tmp22 = tmp0 < tmp21
    tmp23 = tmp20 & tmp22
    tmp24 = tl.load(in_ptr6 + (x0 + 64*((-4) + x1) + 768*x2), tmp23, other=0.0)
    tmp25 = tl.load(in_ptr1 + (x0), tmp23, eviction_policy='evict_last', other=0.0)
    tmp26 = tmp24 * tmp25
    tmp27 = tl.load(in_ptr2 + (x0), tmp23, eviction_policy='evict_last', other=0.0)
    tmp28 = tmp26 + tmp27
    tmp29 = tl.load(in_ptr7 + (x0 + 64*((-4) + x1) + 768*x2), tmp23, other=0.0).to(tl.float32)
    tmp30 = tl.load(in_ptr8 + (x0), tmp23, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp29 + tmp31
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp28 + tmp33
    tmp35 = tl.load(in_ptr5 + (64 + x0), tmp23, eviction_policy='evict_last', other=0.0)
    tmp36 = tmp34 + tmp35
    tmp37 = tl.full(tmp36.shape, 0.0, tmp36.dtype)
    tmp38 = tl.where(tmp23, tmp36, tmp37)
    tmp39 = tmp0 >= tmp21
    tmp40 = tl.full([1], 26, tl.int64)
    tmp41 = tmp0 < tmp40
    tmp42 = tl.load(in_ptr9 + (x0 + 64*((-16) + x1) + 640*x2), tmp39, other=0.0)
    tmp43 = tl.load(in_ptr5 + (128 + x0), tmp39, eviction_policy='evict_last', other=0.0)
    tmp44 = tmp42 + tmp43
    tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
    tmp46 = tl.where(tmp39, tmp44, tmp45)
    tmp47 = tl.where(tmp23, tmp38, tmp46)
    tmp48 = tl.where(tmp4, tmp19, tmp47)
    tmp49 = tmp48.to(tl.float32)
    tl.store(out_ptr1 + (x3), tmp49, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/pn/cpn7ku25srdnsvik3ao4ulhcvbx3xitighwwtfx7gnqih2sxdtz6.py
# Topologically Sorted Source Nodes: [valid_3, any_1], Original ATen: [aten.cat, aten.any]
# Source node to ATen node mapping:
#   any_1 => any_1
#   valid_3 => cat_4
# Graph fragment:
#   %cat_4 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%bitwise_and_1, %bitwise_and_3, %bitwise_and_5], -1), kwargs = {})
#   %any_1 : [num_users=1] = call_function[target=torch.ops.aten.any.dim](args = (%cat_4, -1), kwargs = {})
triton_per_fused_any_cat_20 = async_compile.triton('triton_per_fused_any_cat_20', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16384, 'r0_': 32},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*i1', 'in_ptr2': '*i1', 'in_ptr3': '*i1', 'in_ptr4': '*i1', 'in_ptr5': '*i1', 'in_ptr6': '*i1', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_any_cat_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 1, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_any_cat_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 10752
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


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/nl/cnlzmrakozzk6iwpvdtum7qqlpgoyd3qpqgyvuipkzk247nc45e4.py
# Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul => clone_6
# Graph fragment:
#   %clone_6 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_4,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_21 = async_compile.triton('triton_poi_fused_clone_21', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1048576, 'x': 32}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_21(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 688128
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
    tmp0 = tl.load(in_ptr0 + (y0 + 64*x2 + 1664*y1), ymask & xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tl.store(out_ptr0 + (x2 + 26*y3), tmp3, xmask & ymask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/gl/cglgaszjjakg5ih7mxe76kagwkbnljjhmgnqfm7l7snv2efjhg2k.py
# Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul => clone_5
# Graph fragment:
#   %clone_5 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_3,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_22 = async_compile.triton('triton_poi_fused_clone_22', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_22', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_22(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 11010048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 8)
    x1 = ((xindex // 8) % 16)
    x2 = ((xindex // 128) % 8)
    x3 = xindex // 1024
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 8*x2 + 64*x1 + 1024*x3), None).to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/ek/cekx5haajsi624vizaxpletomtiwr2u7x2mefcrsr54c4qvemg54.py
# Topologically Sorted Source Nodes: [mass], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   mass => cat_5
# Graph fragment:
#   %cat_5 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%mul_13, %mul_16, %mul_19], -1), kwargs = {})
triton_poi_fused_cat_23 = async_compile.triton('triton_poi_fused_cat_23', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_23', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4472832
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 26)
    x4 = xindex // 26
    x5 = xindex // 416
    x2 = ((xindex // 416) % 12)
    x3 = xindex // 4992
    x6 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (1 + 4*x4), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (4*x5 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 * tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 16, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tmp10 & tmp12
    tmp14 = tl.load(in_ptr0 + (2 + 4*x4), tmp13, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr2 + (12*x5 + ((-4) + x0)), tmp13, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp14 * tmp15
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp13, tmp16, tmp17)
    tmp19 = tmp0 >= tmp11
    tmp20 = tl.full([1], 26, tl.int64)
    tmp21 = tmp0 < tmp20
    tmp22 = tl.load(in_ptr0 + (3 + 4*x4), tmp19, eviction_policy='evict_last', other=0.0)
    tmp23 = tl.load(in_ptr3 + (x2 + 12*((-16) + x0) + 120*x3), tmp19, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 * tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp19, tmp24, tmp25)
    tmp27 = tl.where(tmp13, tmp18, tmp26)
    tmp28 = tl.where(tmp4, tmp9, tmp27)
    tl.store(out_ptr0 + (x6), tmp28, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/o4/co4kgb44j3z5kymbjdljfyery3usv46yo7uxc5yrr3qfwojhfbhx.py
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
triton_per_fused__softmax_cat_24 = async_compile.triton('triton_per_fused__softmax_cat_24', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i1', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_cat_24', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 4, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax_cat_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp16 = tl.load(in_ptr2 + (26*x5 + ((-1) + r0_3)), r0_mask & tmp11, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp17 = 0.35355339059327373
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tl.load(in_ptr3 + (26*x0 + 416*x2 + ((-1) + r0_3)), r0_mask & tmp11, eviction_policy='evict_last', other=0.0)
    tmp21 = 1e-08
    tmp22 = triton_helpers.maximum(tmp20, tmp21)
    tmp23 = tl_math.log(tmp22)
    tmp24 = tmp19 + tmp23
    tmp25 = float("-inf")
    tmp26 = tl.where(tmp15, tmp25, tmp24)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp11, tmp26, tmp27)
    tmp29 = tl.where(tmp4, tmp10, tmp28)
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK, R0_BLOCK])
    tmp32 = tl.broadcast_to(tmp30, [XBLOCK, R0_BLOCK])
    tmp34 = tl.where(r0_mask, tmp32, float("-inf"))
    tmp35 = triton_helpers.max2(tmp34, 1)[:, None]
    tmp36 = tmp30 - tmp35
    tmp37 = tl_math.exp(tmp36)
    tmp38 = tl.broadcast_to(tmp37, [XBLOCK, R0_BLOCK])
    tmp40 = tl.where(r0_mask, tmp38, 0)
    tmp41 = tl.sum(tmp40, 1)[:, None]
    tmp42 = tmp29 - tmp35
    tmp43 = tl_math.exp(tmp42)
    tmp44 = (tmp43 / tmp41)
    tl.store(out_ptr2 + (r0_3 + 27*x5), tmp44, r0_mask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/27/c27qti3t6yozi4aalh2yykexx46t3nkacqtp2oqhtr4hh3wd2vw2.py
# Topologically Sorted Source Nodes: [matmul_1], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul_1 => clone_7
# Graph fragment:
#   %clone_7 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_7,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_25 = async_compile.triton('triton_poi_fused_clone_25', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_25', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_25(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 17891328
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 8)
    x1 = ((xindex // 8) % 26)
    x2 = ((xindex // 208) % 8)
    x3 = xindex // 1664
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 8*x2 + 64*x1 + 1664*x3), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0 + 8*x2), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tl.store(out_ptr0 + (x4), tmp3, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/y5/cy5ku4u7ewsi3wi32vha2xbwsun3xjnuu7bazkcf6ezgibznhauk.py
# Topologically Sorted Source Nodes: [matmul_1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   matmul_1 => convert_element_type_57
# Graph fragment:
#   %convert_element_type_57 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_13, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_26 = async_compile.triton('triton_poi_fused__to_copy_26', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_26', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_26(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 35782656
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 26)
    x1 = xindex // 26
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (1 + x0 + 27*x1), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp1, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/uc/cuc2drk7jq3h6szr5ujtksqnpuyqss5kh3q5fk2av4pzsxcegcrg.py
# Topologically Sorted Source Nodes: [features, input_4, input_5], Original ATen: [aten.cat, aten._to_copy, aten.native_layer_norm]
# Source node to ATen node mapping:
#   features => cat_7
#   input_4 => add_23, add_24, convert_element_type_60, mul_20, mul_21, rsqrt_5, sub_8, var_mean_5
#   input_5 => convert_element_type_63
# Graph fragment:
#   %cat_7 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%view_10, %view_30], -1), kwargs = {})
#   %convert_element_type_60 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cat_7, torch.float32), kwargs = {})
#   %var_mean_5 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_60, [4]), kwargs = {correction: 0, keepdim: True})
#   %add_23 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_10, 1e-05), kwargs = {})
#   %rsqrt_5 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_23,), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_60, %getitem_11), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %rsqrt_5), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_20, %primals_52), kwargs = {})
#   %add_24 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_21, %primals_53), kwargs = {})
#   %convert_element_type_63 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_24, torch.bfloat16), kwargs = {})
triton_per_fused__to_copy_cat_native_layer_norm_27 = async_compile.triton('triton_per_fused__to_copy_cat_native_layer_norm_27', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_cat_native_layer_norm_27', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 4, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_cat_native_layer_norm_27(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    r0_2 = r0_index
    x3 = xindex
    x0 = (xindex % 16)
    x1 = xindex // 16
    tmp32 = tl.load(in_ptr2 + (r0_2), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr3 + (r0_2), None, eviction_policy='evict_last')
    tmp0 = r0_2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (64*x3 + (r0_2)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1, 1], 128, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (8*x0 + 128*(((((-64) + r0_2) // 8) % 8)) + 1024*x1 + ((((-64) + r0_2) % 8))), tmp6, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
    tmp14 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp16 = tl.sum(tmp14, 1)[:, None]
    tmp17 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = (tmp16 / tmp18)
    tmp20 = tmp12 - tmp19
    tmp21 = tmp20 * tmp20
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, R0_BLOCK])
    tmp24 = tl.sum(tmp22, 1)[:, None]
    tmp25 = 128.0
    tmp26 = (tmp24 / tmp25)
    tmp27 = 1e-05
    tmp28 = tmp26 + tmp27
    tmp29 = libdevice.rsqrt(tmp28)
    tmp30 = tmp11 - tmp19
    tmp31 = tmp30 * tmp29
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp35.to(tl.float32)
    tl.store(out_ptr0 + (r0_2 + 128*x3), tmp10, None)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp29, None)
    tl.store(out_ptr2 + (r0_2 + 128*x3), tmp36, None)
    tl.store(out_ptr1 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/s5/cs5yez6syweoaqwhbp2jrf7casl4xpdu3mzxhzeeqbbzfq6bplgh.py
# Topologically Sorted Source Nodes: [input_5], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_5 => convert_element_type_62
# Graph fragment:
#   %convert_element_type_62 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_54, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_28 = async_compile.triton('triton_poi_fused__to_copy_28', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_28', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_28(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/3y/c3yy7yzk3ex56ckn5gsqzdpnyfw4wr3qwwm42owdo2h7l2xkjgqb.py
# Topologically Sorted Source Nodes: [invert_5, and__6, eligible, correction_1, float_1], Original ATen: [aten.bitwise_not, aten.bitwise_and, aten.scalar_tensor, aten.where, aten._to_copy]
# Source node to ATen node mapping:
#   and__6 => bitwise_and_6
#   correction_1 => full_default_12, where_15
#   eligible => bitwise_and_7
#   float_1 => convert_element_type_74
#   invert_5 => bitwise_not_5
# Graph fragment:
#   %bitwise_not_5 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_not.default](args = (%primals_2,), kwargs = {})
#   %bitwise_and_6 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%primals_3, %bitwise_not_5), kwargs = {})
#   %bitwise_and_7 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and_6, %unsqueeze_41), kwargs = {})
#   %full_default_12 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_15 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and_7, %squeeze, %full_default_12), kwargs = {})
#   %convert_element_type_74 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where_15, torch.float32), kwargs = {})
triton_poi_fused__to_copy_bitwise_and_bitwise_not_scalar_tensor_where_29 = async_compile.triton('triton_poi_fused__to_copy_bitwise_and_bitwise_not_scalar_tensor_where_29', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*i1', 'in_ptr2': '*i1', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_bitwise_and_bitwise_not_scalar_tensor_where_29', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_bitwise_and_bitwise_not_scalar_tensor_where_29(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 172032
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 16
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x2), None).to(tl.int1)
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last').to(tl.int1)
    tmp6 = tl.load(in_ptr3 + (x2), None).to(tl.float32)
    tmp7 = tl.load(in_ptr4 + (0))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp2 = tmp1 == 0
    tmp3 = tmp0 & tmp2
    tmp5 = tmp3 & tmp4
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp6 + tmp9
    tmp11 = 0.0
    tmp12 = tl.where(tmp5, tmp10, tmp11)
    tmp13 = tmp12.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp13, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57 = args
    args.clear()
    assert_size_stride(primals_1, (128, 7, 12, 64), (5376, 768, 64, 1))
    assert_size_stride(primals_2, (128, 7, 12, 16), (1344, 192, 16, 1))
    assert_size_stride(primals_3, (128, 7, 12, 16), (1344, 192, 16, 1))
    assert_size_stride(primals_4, (128, 7), (7, 1))
    assert_size_stride(primals_5, (896, 64, 96), (6144, 96, 1))
    assert_size_stride(primals_6, (896, 5, 96), (480, 96, 1))
    assert_size_stride(primals_7, (128, 7, 12, 64), (5376, 768, 64, 1))
    assert_size_stride(primals_8, (128, 7, 12, 4), (336, 48, 4, 1))
    assert_size_stride(primals_9, (128, 7, 12, 16, 4), (5376, 768, 64, 4, 1))
    assert_size_stride(primals_10, (128, 7, 12, 64), (5376, 768, 64, 1))
    assert_size_stride(primals_11, (128, 7, 12, 64), (5376, 768, 64, 1))
    assert_size_stride(primals_12, (128, 7, 12, 64), (5376, 768, 64, 1))
    assert_size_stride(primals_13, (16, 64), (64, 1))
    assert_size_stride(primals_14, (16, ), (1, ))
    assert_size_stride(primals_15, (64, ), (1, ))
    assert_size_stride(primals_16, (64, ), (1, ))
    assert_size_stride(primals_17, (128, 7, 12, 16), (1344, 192, 16, 1))
    assert_size_stride(primals_18, (64, ), (1, ))
    assert_size_stride(primals_19, (64, ), (1, ))
    assert_size_stride(primals_20, (64, 143), (143, 1))
    assert_size_stride(primals_21, (64, ), (1, ))
    assert_size_stride(primals_22, (64, 64), (64, 1))
    assert_size_stride(primals_23, (64, ), (1, ))
    assert_size_stride(primals_24, (128, 7, 12), (84, 12, 1))
    assert_size_stride(primals_25, (128, 7, 12, 4, 64), (21504, 3072, 256, 64, 1))
    assert_size_stride(primals_26, (128, 7, 12, 4, 5), (1680, 240, 20, 5, 1))
    assert_size_stride(primals_27, (128, 7, 12, 4), (336, 48, 4, 1))
    assert_size_stride(primals_28, (128, 7, 12, 4), (336, 48, 4, 1))
    assert_size_stride(primals_29, (64, ), (1, ))
    assert_size_stride(primals_30, (64, ), (1, ))
    assert_size_stride(primals_31, (64, 5), (5, 1))
    assert_size_stride(primals_32, (64, ), (1, ))
    assert_size_stride(primals_33, (3, 64), (64, 1))
    assert_size_stride(primals_34, (128, 7, 12), (84, 12, 1))
    assert_size_stride(primals_35, (128, 7, 12, 12, 64), (64512, 9216, 768, 64, 1))
    assert_size_stride(primals_36, (128, 7, 12, 12, 4), (4032, 576, 48, 4, 1))
    assert_size_stride(primals_37, (128, 7, 12, 12), (1008, 144, 12, 1))
    assert_size_stride(primals_38, (128, 7, 12, 12), (1008, 144, 12, 1))
    assert_size_stride(primals_39, (64, 4), (4, 1))
    assert_size_stride(primals_40, (64, ), (1, ))
    assert_size_stride(primals_41, (128, 7, 12), (84, 12, 1))
    assert_size_stride(primals_42, (128, 7, 12, 10, 64), (53760, 7680, 640, 64, 1))
    assert_size_stride(primals_43, (128, 7, 12, 10, 5), (4200, 600, 50, 5, 1))
    assert_size_stride(primals_44, (128, 7, 12, 10), (840, 120, 10, 1))
    assert_size_stride(primals_45, (128, 7, 12, 10), (840, 120, 1, 12))
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
        buf0 = empty_strided_cuda((896, 64, 104), (6656, 104, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pad], Original ATen: [aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_0.run(primals_5, buf0, 5963776, stream=stream0)
        del primals_5
        buf1 = empty_strided_cuda((128, 7, 12, 1), (84, 12, 1, 1), torch.float32)
        buf2 = empty_strided_cuda((128, 7, 12, 1), (84, 12, 1, 10752), torch.float32)
        buf4 = reinterpret_tensor(buf2, (128, 7, 12, 1), (84, 12, 1, 1), 0); del buf2  # reuse
        buf5 = empty_strided_cuda((128, 7, 12, 64), (5376, 768, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_layer_norm_1.run(buf4, primals_7, primals_15, primals_16, buf1, buf5, 10752, 64, stream=stream0)
        del primals_16
        buf6 = empty_strided_cuda((128, 7, 12, 16, 1), (1344, 192, 16, 1, 1), torch.float32)
        buf7 = empty_strided_cuda((128, 7, 12, 16, 1), (1344, 192, 16, 1, 172032), torch.float32)
        buf9 = reinterpret_tensor(buf7, (128, 7, 12, 16, 1), (1344, 192, 16, 1, 1), 0); del buf7  # reuse
        # Topologically Sorted Source Nodes: [invert, context, layer_norm_1], Original ATen: [aten.bitwise_not, aten.masked_fill, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_bitwise_not_masked_fill_native_layer_norm_2.run(buf9, primals_3, buf0, buf6, 172032, 64, stream=stream0)
        buf11 = empty_strided_cuda((128, 7, 12, 16, 143), (193536, 27648, 2304, 143, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [cat_1, input_1], Original ATen: [aten.cat, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_cat_3.run(primals_3, buf0, buf6, buf9, primals_18, primals_19, buf5, primals_6, primals_17, primals_4, primals_9, primals_8, buf11, 24600576, stream=stream0)
        del buf5
        del primals_17
        del primals_19
        del primals_6
        buf12 = empty_strided_cuda((172032, 143), (143, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy, aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_view_4.run(buf11, buf12, 24600576, stream=stream0)
        del buf11
        buf13 = empty_strided_cuda((64, 143), (143, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_5.run(primals_20, buf13, 9152, stream=stream0)
        del primals_20
        buf14 = empty_strided_cuda((64, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_6.run(primals_21, buf14, 64, stream=stream0)
        del primals_21
        buf15 = empty_strided_cuda((172032, 64), (64, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy, aten.addmm]
        extern_kernels.addmm(buf14, buf12, reinterpret_tensor(buf13, (143, 64), (1, 143), 0), alpha=1, beta=1, out=buf15)
        buf16 = empty_strided_cuda((128, 7, 12, 16, 64), (86016, 12288, 1024, 64, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_7.run(buf15, buf16, 11010048, stream=stream0)
        buf17 = empty_strided_cuda((64, 64), (64, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_8.run(primals_22, buf17, 4096, stream=stream0)
        del primals_22
        buf18 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_6.run(primals_23, buf18, 64, stream=stream0)
        del primals_23
        buf19 = empty_strided_cuda((172032, 64), (64, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._to_copy, aten.addmm]
        extern_kernels.addmm(buf18, reinterpret_tensor(buf16, (172032, 64), (64, 1), 0), reinterpret_tensor(buf17, (64, 64), (1, 64), 0), alpha=1, beta=1, out=buf19)
        buf23 = empty_strided_cuda((128, 7, 12, 4, 64), (21504, 3072, 256, 64, 1), torch.float32)
        buf28 = empty_strided_cuda((128, 7, 12, 4), (336, 48, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [context, and_, valid, raw, layer_norm_2, weights], Original ATen: [aten.masked_fill, aten.bitwise_and, aten.where, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_bitwise_and_masked_fill_native_layer_norm_where_9.run(primals_28, primals_24, primals_8, primals_25, primals_27, buf23, buf28, 43008, 64, stream=stream0)
        del primals_25
        del primals_27
        buf24 = empty_strided_cuda((128, 7, 12, 4, 5), (1728, 240, 20, 5, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [context, quality, linear_4], Original ATen: [aten.masked_fill, aten.where, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_masked_fill_where_10.run(primals_28, primals_24, primals_8, primals_26, buf24, 215040, stream=stream0)
        del primals_26
        buf25 = empty_strided_cuda((43008, 5), (5, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [context, quality, linear_4], Original ATen: [aten.masked_fill, aten.where, aten._to_copy, aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_masked_fill_view_where_11.run(buf24, buf25, 215040, stream=stream0)
        del buf24
        buf26 = empty_strided_cuda((64, 5), (5, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_12.run(primals_31, buf26, 320, stream=stream0)
        del primals_31
        buf27 = empty_strided_cuda((43008, 64), (64, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.addmm]
        extern_kernels.mm(buf25, reinterpret_tensor(buf26, (5, 64), (1, 5), 0), out=buf27)
        buf32 = empty_strided_cuda((128, 7, 12, 12, 64), (64512, 9216, 768, 64, 1), torch.float32)
        buf80 = empty_strided_cuda((128, 7, 12, 12, 1), (1008, 144, 12, 1, 1), torch.float32)
        buf36 = empty_strided_cuda((128, 7, 12, 12), (1008, 144, 12, 1), torch.float32)
        # Topologically Sorted Source Nodes: [context, and__2, valid_1, raw_1, layer_norm_3, weights_1], Original ATen: [aten.masked_fill, aten.bitwise_and, aten.where, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_bitwise_and_masked_fill_native_layer_norm_native_layer_norm_backward_where_13.run(primals_38, primals_34, primals_8, primals_35, primals_37, buf32, buf80, buf36, 129024, 64, stream=stream0)
        del primals_35
        del primals_37
        buf33 = empty_strided_cuda((128, 7, 12, 12, 4), (4032, 576, 48, 4, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [context, quality_1, linear_5], Original ATen: [aten.masked_fill, aten.where, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_masked_fill_where_14.run(primals_38, primals_34, primals_8, primals_36, buf33, 516096, stream=stream0)
        del primals_36
        buf34 = empty_strided_cuda((4, 64), (1, 4), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_15.run(primals_39, buf34, 256, stream=stream0)
        del primals_39
        buf35 = empty_strided_cuda((129024, 64), (64, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf33, (129024, 4), (4, 1), 0), buf34, out=buf35)
        buf40 = empty_strided_cuda((128, 7, 12, 10, 5), (4224, 600, 50, 5, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [context, quality_2, linear_6], Original ATen: [aten.masked_fill, aten.where, aten._to_copy, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_clone_masked_fill_where_16.run(primals_45, primals_41, primals_8, primals_43, buf40, 537600, stream=stream0)
        del primals_43
        buf41 = empty_strided_cuda((107520, 5), (5, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [context, quality_2, linear_6], Original ATen: [aten.masked_fill, aten.where, aten._to_copy, aten.clone, aten._unsafe_view]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_view_clone_masked_fill_where_17.run(buf40, buf41, 537600, stream=stream0)
        del buf40
        buf42 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_12.run(primals_46, buf42, 320, stream=stream0)
        del primals_46
        buf43 = empty_strided_cuda((107520, 64), (64, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten.mm]
        extern_kernels.mm(buf41, reinterpret_tensor(buf42, (5, 64), (1, 5), 0), out=buf43)
        del buf42
        buf44 = empty_strided_cuda((128, 7, 12, 10, 64), (53760, 7680, 640, 64, 1), torch.float32)
        buf78 = empty_strided_cuda((128, 7, 12, 10, 64), (53760, 7680, 64, 768, 1), torch.float32)
        buf79 = empty_strided_cuda((128, 7, 12, 10, 1), (840, 120, 10, 1, 1), torch.float32)
        buf46 = empty_strided_cuda((128, 7, 12, 1, 10), (840, 120, 1, 120, 12), torch.float32)
        # Topologically Sorted Source Nodes: [context, and__4, valid_2, raw_2, layer_norm_4, linear_6, add_7, weights_2, getitem_33], Original ATen: [aten.masked_fill, aten.bitwise_and, aten.where, aten.native_layer_norm, aten._to_copy, aten.add, aten.unsqueeze, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_bitwise_and_masked_fill_native_layer_norm_native_layer_norm_backward_unsqueeze_where_18.run(primals_45, primals_41, primals_8, primals_42, primals_29, primals_30, buf43, primals_47, primals_44, buf44, buf78, buf79, buf46, 107520, 64, stream=stream0)
        del buf43
        del primals_42
        del primals_44
        del primals_47
        buf49 = empty_strided_cuda((128, 7, 12, 26, 64), (139776, 19968, 1664, 64, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [encoded_3, linear_7], Original ATen: [aten.cat, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_cat_19.run(buf23, primals_29, primals_30, buf27, primals_32, primals_33, buf32, buf35, primals_40, buf44, buf49, 17891328, stream=stream0)
        del buf27
        del buf35
        del buf44
        del primals_30
        del primals_32
        del primals_33
        del primals_40
        buf48 = empty_strided_cuda((128, 7, 12, 26), (2184, 312, 26, 1), torch.bool)
        buf76 = empty_strided_cuda((128, 7, 12), (84, 12, 1), torch.bool)
        # Topologically Sorted Source Nodes: [valid_3, any_1], Original ATen: [aten.cat, aten.any]
        stream0 = get_raw_stream(0)
        triton_per_fused_any_cat_20.run(primals_28, primals_24, primals_8, primals_38, primals_34, primals_45, primals_41, buf48, buf76, 10752, 26, stream=stream0)
        del primals_24
        del primals_28
        buf50 = empty_strided_cuda((64, 64), (1, 64), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_8.run(primals_48, buf50, 4096, stream=stream0)
        del primals_48
        buf51 = empty_strided_cuda((279552, 64), (64, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf49, (279552, 64), (64, 1), 0), buf50, out=buf51)
        buf52 = empty_strided_cuda((64, 64), (1, 64), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_8.run(primals_50, buf52, 4096, stream=stream0)
        del primals_50
        buf53 = empty_strided_cuda((279552, 64), (64, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf49, (279552, 64), (64, 1), 0), buf52, out=buf53)
        buf54 = empty_strided_cuda((128, 7, 12, 8, 8, 26), (139776, 19968, 1664, 208, 26, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_21.run(buf51, primals_49, buf54, 688128, 26, stream=stream0)
        del primals_49
        buf55 = empty_strided_cuda((128, 7, 12, 8, 16, 8), (86016, 12288, 1024, 128, 8, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_22.run(buf19, buf55, 11010048, stream=stream0)
        buf56 = empty_strided_cuda((86016, 16, 26), (416, 26, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf55, (86016, 16, 8), (128, 8, 1), 0), reinterpret_tensor(buf54, (86016, 8, 26), (208, 26, 1), 0), out=buf56)
        buf57 = empty_strided_cuda((128, 7, 12, 16, 26), (34944, 4992, 416, 26, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mass], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_23.run(primals_9, buf28, buf36, buf46, buf57, 4472832, stream=stream0)
        buf60 = empty_strided_cuda((128, 7, 12, 8, 16, 27), (290304, 41472, 3456, 432, 27, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_5, attention], Original ATen: [aten.cat, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_cat_24.run(primals_9, buf48, buf56, buf57, buf60, 1376256, 27, stream=stream0)
        buf61 = reinterpret_tensor(buf51, (128, 7, 12, 8, 26, 8), (139776, 19968, 1664, 208, 8, 1), 0); del buf51  # reuse
        # Topologically Sorted Source Nodes: [matmul_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_25.run(buf53, primals_51, buf61, 17891328, stream=stream0)
        del buf53
        del primals_51
        buf62 = reinterpret_tensor(buf56, (128, 7, 12, 8, 16, 26), (279552, 39936, 3328, 416, 26, 1), 0); del buf56  # reuse
        # Topologically Sorted Source Nodes: [matmul_1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_26.run(buf60, buf62, 35782656, stream=stream0)
        buf63 = empty_strided_cuda((86016, 16, 8), (128, 8, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf62, (86016, 16, 26), (416, 26, 1), 0), reinterpret_tensor(buf61, (86016, 26, 8), (208, 8, 1), 0), out=buf63)
        buf64 = empty_strided_cuda((128, 7, 12, 16, 128), (172032, 24576, 2048, 128, 1), torch.bfloat16)
        buf65 = empty_strided_cuda((128, 7, 12, 16, 1), (1344, 192, 16, 1, 1), torch.float32)
        buf66 = empty_strided_cuda((128, 7, 12, 16, 1), (1344, 192, 16, 1, 172032), torch.float32)
        buf68 = reinterpret_tensor(buf66, (128, 7, 12, 16, 1), (1344, 192, 16, 1, 1), 0); del buf66  # reuse
        buf69 = empty_strided_cuda((128, 7, 12, 16, 128), (172032, 24576, 2048, 128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [features, input_4, input_5], Original ATen: [aten.cat, aten._to_copy, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_cat_native_layer_norm_27.run(buf68, buf19, buf63, primals_52, primals_53, buf64, buf65, buf69, 172032, 128, stream=stream0)
        del primals_53
        buf70 = empty_strided_cuda((64, 128), (128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_28.run(primals_54, buf70, 8192, stream=stream0)
        del primals_54
        buf71 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_6.run(primals_55, buf71, 64, stream=stream0)
        del primals_55
        buf72 = reinterpret_tensor(buf63, (172032, 64), (64, 1), 0); del buf63  # reuse
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten._to_copy, aten.addmm]
        extern_kernels.addmm(buf71, reinterpret_tensor(buf69, (172032, 128), (128, 1), 0), reinterpret_tensor(buf70, (128, 64), (1, 128), 0), alpha=1, beta=1, out=buf72)
        buf73 = reinterpret_tensor(buf19, (128, 7, 12, 16, 64), (86016, 12288, 1024, 64, 1), 0); del buf19  # reuse
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_7.run(buf72, buf73, 11010048, stream=stream0)
        buf74 = reinterpret_tensor(buf71, (64, 1), (1, 64), 0); del buf71  # reuse
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_6.run(primals_56, buf74, 64, stream=stream0)
        del primals_56
        buf75 = empty_strided_cuda((172032, 1), (1, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf73, (172032, 64), (64, 1), 0), buf74, out=buf75)
        buf77 = empty_strided_cuda((128, 7, 12, 16), (1344, 192, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [invert_5, and__6, eligible, correction_1, float_1], Original ATen: [aten.bitwise_not, aten.bitwise_and, aten.scalar_tensor, aten.where, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_bitwise_and_bitwise_not_scalar_tensor_where_29.run(primals_3, primals_2, buf76, buf75, primals_57, buf77, 172032, stream=stream0)
        del buf75
        del primals_57
    return (buf77, buf60, buf48, primals_2, primals_3, primals_4, primals_7, primals_9, primals_15, primals_18, primals_29, primals_34, primals_38, primals_41, primals_45, primals_52, reinterpret_tensor(buf0, (896, 64, 12, 16), (6656, 104, 8, 1), 0), buf1, buf4, buf6, buf9, buf12, buf15, reinterpret_tensor(buf16, (172032, 64), (64, 1), 0), buf23, buf25, reinterpret_tensor(buf28, (128, 7, 12, 1, 4), (336, 48, 4, 4, 1), 0), reinterpret_tensor(primals_8, (128, 7, 12, 1), (336, 48, 4, 1), 2), buf32, reinterpret_tensor(buf33, (129024, 4), (4, 1), 0), reinterpret_tensor(buf36, (128, 7, 12, 1, 12), (1008, 144, 12, 12, 1), 0), reinterpret_tensor(primals_8, (128, 7, 12, 1), (336, 48, 4, 1), 3), buf41, buf46, buf48, reinterpret_tensor(buf49, (279552, 64), (64, 1), 0), buf57, buf60, buf64, buf65, buf68, reinterpret_tensor(buf69, (172032, 128), (128, 1), 0), buf72, reinterpret_tensor(buf73, (172032, 64), (64, 1), 0), reinterpret_tensor(buf76, (128, 7, 12, 1), (84, 12, 1, 1), 0), reinterpret_tensor(buf74, (1, 64), (64, 1), 0), buf70, reinterpret_tensor(buf62, (86016, 26, 16), (416, 1, 26), 0), reinterpret_tensor(buf61, (86016, 8, 26), (208, 1, 8), 0), reinterpret_tensor(buf55, (86016, 8, 16), (128, 1, 8), 0), reinterpret_tensor(buf54, (86016, 26, 8), (208, 1, 26), 0), reinterpret_tensor(buf52, (64, 64), (64, 1), 0), reinterpret_tensor(buf50, (64, 64), (64, 1), 0), buf78, buf79, reinterpret_tensor(buf34, (64, 4), (4, 1), 0), buf80, buf17, buf13, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((128, 7, 12, 64), (5376, 768, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((128, 7, 12, 16), (1344, 192, 16, 1), device='cuda:0', dtype=torch.bool)
    primals_3 = rand_strided((128, 7, 12, 16), (1344, 192, 16, 1), device='cuda:0', dtype=torch.bool)
    primals_4 = rand_strided((128, 7), (7, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((896, 64, 96), (6144, 96, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((896, 5, 96), (480, 96, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((128, 7, 12, 64), (5376, 768, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((128, 7, 12, 4), (336, 48, 4, 1), device='cuda:0', dtype=torch.bool)
    primals_9 = rand_strided((128, 7, 12, 16, 4), (5376, 768, 64, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((128, 7, 12, 64), (5376, 768, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((128, 7, 12, 64), (5376, 768, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((128, 7, 12, 64), (5376, 768, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((128, 7, 12, 16), (1344, 192, 16, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((64, 143), (143, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((128, 7, 12), (84, 12, 1), device='cuda:0', dtype=torch.bool)
    primals_25 = rand_strided((128, 7, 12, 4, 64), (21504, 3072, 256, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((128, 7, 12, 4, 5), (1680, 240, 20, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((128, 7, 12, 4), (336, 48, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((128, 7, 12, 4), (336, 48, 4, 1), device='cuda:0', dtype=torch.bool)
    primals_29 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((64, 5), (5, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((3, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((128, 7, 12), (84, 12, 1), device='cuda:0', dtype=torch.bool)
    primals_35 = rand_strided((128, 7, 12, 12, 64), (64512, 9216, 768, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((128, 7, 12, 12, 4), (4032, 576, 48, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((128, 7, 12, 12), (1008, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((128, 7, 12, 12), (1008, 144, 12, 1), device='cuda:0', dtype=torch.bool)
    primals_39 = rand_strided((64, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((128, 7, 12), (84, 12, 1), device='cuda:0', dtype=torch.bool)
    primals_42 = rand_strided((128, 7, 12, 10, 64), (53760, 7680, 640, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((128, 7, 12, 10, 5), (4200, 600, 50, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((128, 7, 12, 10), (840, 120, 10, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((128, 7, 12, 10), (840, 120, 1, 12), device='cuda:0', dtype=torch.bool)
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
