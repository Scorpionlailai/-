# AOT ID: ['0_inference']
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


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/2q/c2qafp5zfcsinjfurdy52fthxm7fnb3ccsbbiirjmi3ff3qkg2li.py
# Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm => add_3, add_4, mul_3, mul_4, rsqrt, sub_2, var_mean
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%arg6_1, [3]), kwargs = {correction: 0, keepdim: True})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg6_1, %getitem_1), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_3,), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, %arg14_1), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %arg15_1), kwargs = {})
triton_per_fused_native_layer_norm_0 = async_compile.triton('triton_per_fused_native_layer_norm_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 4, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_0(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp17 = tmp0 - tmp10
    tmp18 = 64.0
    tmp19 = (tmp16 / tmp18)
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tl.store(out_ptr2 + (r0_1 + 64*x0), tmp27, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/6m/c6mxgpgoa6ddotq74pifnnochgifbrbozq6u3olmkuaq57tpqlnc.py
# Topologically Sorted Source Nodes: [invert, context, layer_norm_1], Original ATen: [aten.bitwise_not, aten.masked_fill, aten.native_layer_norm]
# Source node to ATen node mapping:
#   context => clone, full_default, where
#   invert => bitwise_not
#   layer_norm_1 => var_mean_1
# Graph fragment:
#   %bitwise_not : [num_users=1] = call_function[target=torch.ops.aten.bitwise_not.default](args = (%unsqueeze,), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_not, %full_default, %view), kwargs = {})
#   %clone : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%where,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone, [4]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_bitwise_not_masked_fill_native_layer_norm_1 = async_compile.triton('triton_per_fused_bitwise_not_masked_fill_native_layer_norm_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_bitwise_not_masked_fill_native_layer_norm_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 4, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_bitwise_not_masked_fill_native_layer_norm_1(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    x0 = (xindex % 16)
    x1 = ((xindex // 16) % 12)
    r0_3 = r0_index
    x2 = xindex // 192
    tmp0 = tl.load(in_ptr0 + (x4), None, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tmp0 == 0
    tmp2 = x0 + 8*x1
    tmp3 = tl.full([1, 1], 96, tl.int64)
    tmp4 = tmp2 < tmp3
    tmp5 = tl.load(in_ptr1 + (x0 + 8*x1 + 96*r0_3 + 6144*x2), tmp4, other=0.0)
    tmp6 = 0.0
    tmp7 = tl.where(tmp1, tmp6, tmp5)
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp10 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp12 = tl.sum(tmp10, 1)[:, None]
    tmp13 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = (tmp12 / tmp14)
    tmp16 = tmp8 - tmp15
    tmp17 = tmp16 * tmp16
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
    tmp20 = tl.sum(tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp15, None)
    tl.store(out_ptr1 + (x4), tmp20, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/hc/chcrcmgd34mibgbs7fszpshpuhgpgvnlwdnn52zv4j36eninxw3m.py
# Topologically Sorted Source Nodes: [quality], Original ATen: [aten.scalar_tensor, aten.where]
# Source node to ATen node mapping:
#   quality => full_default_3, where_6
# Graph fragment:
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_6 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_22, %arg25_1, %full_default_3), kwargs = {})
triton_poi_fused_scalar_tensor_where_2 = async_compile.triton('triton_poi_fused_scalar_tensor_where_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*i1', 'in_ptr2': '*i1', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_scalar_tensor_where_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_scalar_tensor_where_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x8 + 1696*x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/4q/c4q2vyedk3d54hagfij7hmlupvoo7vht5ab5p6r3urlpahls6gra.py
# Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   linear_4 => constant_pad_nd_default_2
# Graph fragment:
#   %constant_pad_nd_default_2 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view_11, [0, 3, 0, 0]), kwargs = {})
triton_poi_fused_addmm_3 = async_compile.triton('triton_poi_fused_addmm_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 344064
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 8)
    x1 = xindex // 8
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 5, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x0 + 5*((x1 % 336)) + 1696*(x1 // 336)), tmp2, other=0.0)
    tl.store(out_ptr0 + (x2), tmp3, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/kv/ckvb2paepdfck7oxohdz7jrazaao543hubwbrdobv7c6vlyeqie3.py
# Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   linear_4 => constant_pad_nd_default_3
# Graph fragment:
#   %constant_pad_nd_default_3 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%permute_7, [0, 0, 0, 3]), kwargs = {})
triton_poi_fused_addmm_4 = async_compile.triton('triton_poi_fused_addmm_4', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/dn/cdnxsdmq24tv52abjhbgco3ndsxtdnf5r6zbvpuq3u4lok24xmac.py
# Topologically Sorted Source Nodes: [raw, layer_norm_2, add_3], Original ATen: [aten.scalar_tensor, aten.where, aten.native_layer_norm, aten.add]
# Source node to ATen node mapping:
#   add_3 => add_11
#   layer_norm_2 => add_10, add_9, mul_11, mul_12, rsqrt_2, sub_4, var_mean_2
#   raw => full_default_2, where_5
# Graph fragment:
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_5 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_21, %arg24_1, %full_default_2), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%where_5, [4]), kwargs = {correction: 0, keepdim: True})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where_5, %getitem_5), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_4, 1e-05), kwargs = {})
#   %rsqrt_2 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_2), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_11, %arg28_1), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_12, %arg29_1), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_10, %view_12), kwargs = {})
triton_per_fused_add_native_layer_norm_scalar_tensor_where_5 = async_compile.triton('triton_per_fused_add_native_layer_norm_scalar_tensor_where_5', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i1', 'in_ptr1': '*i1', 'in_ptr2': '*i1', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_scalar_tensor_where_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 4, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_scalar_tensor_where_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp31 = tl.load(in_ptr4 + (r0_2), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr5 + (r0_2), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_out_ptr0 + (r0_2 + 64*x3), xmask, other=0.0)
    tmp36 = tl.load(in_ptr6 + (r0_2), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (r0_2 + 64*x3), tmp38, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/a7/ca76hzulj6qy56mnt4iybayunsb6uqza27e2ovlskyb7bup2z4jm.py
# Topologically Sorted Source Nodes: [quality_1], Original ATen: [aten.scalar_tensor, aten.where]
# Source node to ATen node mapping:
#   quality_1 => full_default_6, where_9
# Graph fragment:
#   %full_default_6 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_9 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_28, %arg35_1, %full_default_6), kwargs = {})
triton_poi_fused_scalar_tensor_where_6 = async_compile.triton('triton_poi_fused_scalar_tensor_where_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*i1', 'in_ptr2': '*i1', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_scalar_tensor_where_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_scalar_tensor_where_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x5), tmp7, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/6x/c6xh2gprj6ajygco63xn7ablwyjtpuq7bopc2h2v6wwolhnang6s.py
# Topologically Sorted Source Nodes: [raw_1, layer_norm_3, add_5], Original ATen: [aten.scalar_tensor, aten.where, aten.native_layer_norm, aten.add]
# Source node to ATen node mapping:
#   add_5 => add_15
#   layer_norm_3 => add_13, add_14, mul_14, mul_15, rsqrt_3, sub_5, var_mean_3
#   raw_1 => full_default_5, where_8
# Graph fragment:
#   %full_default_5 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_8 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_27, %arg34_1, %full_default_5), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%where_8, [4]), kwargs = {correction: 0, keepdim: True})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where_8, %getitem_7), kwargs = {})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-05), kwargs = {})
#   %rsqrt_3 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_13,), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %rsqrt_3), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_14, %arg28_1), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_15, %arg29_1), kwargs = {})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_14, %view_14), kwargs = {})
triton_per_fused_add_native_layer_norm_scalar_tensor_where_7 = async_compile.triton('triton_per_fused_add_native_layer_norm_scalar_tensor_where_7', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i1', 'in_ptr1': '*i1', 'in_ptr2': '*i1', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_scalar_tensor_where_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 4, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_scalar_tensor_where_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp31 = tl.load(in_ptr4 + (r0_2), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr5 + (r0_2), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_out_ptr0 + (r0_2 + 64*x3), xmask, other=0.0)
    tmp36 = tl.load(in_ptr6 + (r0_2), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (r0_2 + 64*x3), tmp38, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/zg/czgpir766lbpyltlw5wyv77zctvnbuvusf32o4h6dmobbl4cpbqr.py
# Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten.bmm]
# Source node to ATen node mapping:
#   linear_6 => constant_pad_nd_default
# Graph fragment:
#   %constant_pad_nd_default : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view_15, [0, 3, 0, 2, 0, 0]), kwargs = {})
triton_poi_fused_bmm_8 = async_compile.triton('triton_poi_fused_bmm_8', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bmm_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_bmm_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1032192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 8) % 12)
    x0 = (xindex % 8)
    x2 = xindex // 96
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 10, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x0
    tmp4 = tl.full([1], 5, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp2 & tmp5
    tmp7 = tl.load(in_ptr0 + (12*x1 + 120*(x2 // 12) + ((x2 % 12))), tmp6, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp8 = tl.load(in_ptr1 + (x2), tmp6, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp9 = tmp7 & tmp8
    tmp10 = tl.load(in_ptr2 + (3 + 4*x2), tmp6, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp11 = tmp9 & tmp10
    tmp12 = tl.load(in_ptr3 + (x0 + 5*x1 + 50*x2), tmp6, other=0.0)
    tmp13 = 0.0
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp6, tmp14, tmp15)
    tl.store(out_ptr0 + (x4), tmp16, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/vz/cvztr5ex7lvosme53ntmqa5wvvoui3ylkek5mikd2j4bw253jbpp.py
# Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten.bmm]
# Source node to ATen node mapping:
#   linear_6 => constant_pad_nd_default_1
# Graph fragment:
#   %constant_pad_nd_default_1 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view_16, [0, 0, 0, 3, 0, 0]), kwargs = {})
triton_poi_fused_bmm_9 = async_compile.triton('triton_poi_fused_bmm_9', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bmm_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_bmm_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5505024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 8)
    x1 = ((xindex // 8) % 64)
    x3 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 5, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x0 + 5*x1), tmp2, eviction_policy='evict_last', other=0.0)
    tl.store(out_ptr0 + (x3), tmp3, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/oy/coyrxhmhoxyi4boarrq2fc4ysv4b6h4q3jlvekj5t36lh5gykukp.py
# Topologically Sorted Source Nodes: [raw_2, layer_norm_4, linear_6, add_7], Original ATen: [aten.scalar_tensor, aten.where, aten.native_layer_norm, aten.add]
# Source node to ATen node mapping:
#   add_7 => add_20
#   layer_norm_4 => add_17, add_18, clone_3, mul_17, mul_18, rsqrt_4, sub_6, var_mean_4
#   linear_6 => add_19
#   raw_2 => full_default_8, where_11
# Graph fragment:
#   %full_default_8 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_11 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_33, %arg41_1, %full_default_8), kwargs = {})
#   %clone_3 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%where_11,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_3, [4]), kwargs = {correction: 0, keepdim: True})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_3, %getitem_9), kwargs = {})
#   %add_17 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_8, 1e-05), kwargs = {})
#   %rsqrt_4 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_17,), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %rsqrt_4), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_17, %arg28_1), kwargs = {})
#   %add_18 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_18, %arg29_1), kwargs = {})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_17, %arg46_1), kwargs = {})
#   %add_20 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_18, %add_19), kwargs = {})
triton_per_fused_add_native_layer_norm_scalar_tensor_where_10 = async_compile.triton('triton_per_fused_add_native_layer_norm_scalar_tensor_where_10', '''
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
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*i1', 'in_ptr2': '*i1', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_scalar_tensor_where_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 4, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_scalar_tensor_where_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    x4 = xindex // 10
    r0_3 = r0_index
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x1 + 12*x0 + 120*x2), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x4), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp3 = tl.load(in_ptr2 + (3 + 4*x4), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp5 = tl.load(in_ptr3 + (r0_3 + 64*x5), xmask, other=0.0)
    tmp31 = tl.load(in_ptr4 + (r0_3), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr5 + (r0_3), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr6 + (r0_3 + 64*x0 + 768*x4), xmask, other=0.0)
    tmp36 = tl.load(in_ptr7 + (r0_3), None, eviction_policy='evict_last')
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
    tl.store(out_ptr2 + (r0_3 + 64*x5), tmp38, xmask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/qj/cqjyoxmhtmfq2wlmaittxezjnwomoryscwbkj2krmea7gig44gok.py
# Topologically Sorted Source Nodes: [valid_3, any_1], Original ATen: [aten.cat, aten.any]
# Source node to ATen node mapping:
#   any_1 => any_1
#   valid_3 => cat_4
# Graph fragment:
#   %cat_4 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%bitwise_and_1, %bitwise_and_3, %bitwise_and_5], -1), kwargs = {})
#   %any_1 : [num_users=1] = call_function[target=torch.ops.aten.any.dim](args = (%cat_4, -1), kwargs = {})
triton_per_fused_any_cat_11 = async_compile.triton('triton_per_fused_any_cat_11', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_any_cat_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 1, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_any_cat_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp5 = tl.load(in_ptr0 + (4*x3 + (r0_2)), r0_mask & xmask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.int1)
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
    tmp16 = tl.load(in_ptr3 + (12*x3 + ((-4) + r0_2)), r0_mask & xmask & tmp15, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp17 = tl.load(in_ptr4 + (tl.broadcast_to(x3, [XBLOCK, R0_BLOCK])), xmask & r0_mask & tmp15, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp18 = tmp16 & tmp17
    tmp19 = tl.load(in_ptr2 + (tl.broadcast_to(2 + 4*x3, [XBLOCK, R0_BLOCK])), xmask & r0_mask & tmp15, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp20 = tmp18 & tmp19
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp15, tmp20, tmp21)
    tmp23 = tmp0 >= tmp13
    tmp24 = tl.full([1, 1], 26, tl.int64)
    tmp25 = tmp0 < tmp24
    tmp26 = tl.load(in_ptr5 + (x0 + 12*((-16) + r0_2) + 120*x1), r0_mask & xmask & tmp23, other=0.0).to(tl.int1)
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


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/g2/cg2umzonc7rzmcbppk6rcoijn2esihwdt5imdoeafizhojiog3cp.py
# Topologically Sorted Source Nodes: [cat_1], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_1 => cat_2
# Graph fragment:
#   %cat_2 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_7, %expand_2, %clone_1, %unsqueeze_12, %unsqueeze_18, %arg8_1, %convert_element_type_2], -1), kwargs = {})
triton_poi_fused_cat_12 = async_compile.triton('triton_poi_fused_cat_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*i1', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24600576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 143)
    x4 = xindex // 143
    x1 = ((xindex // 143) % 16)
    x2 = ((xindex // 2288) % 12)
    x3 = xindex // 27456
    x5 = xindex // 2288
    x6 = (xindex % 2288)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4), tmp4, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp6 = tmp5 == 0
    tmp7 = x1 + 8*x2
    tmp8 = tl.full([1], 96, tl.int64)
    tmp9 = tmp7 < tmp8
    tmp10 = tmp9 & tmp4
    tmp11 = tl.load(in_ptr1 + (x1 + 8*x2 + 96*(x0) + 6144*x3), tmp10, eviction_policy='evict_last', other=0.0)
    tmp12 = 0.0
    tmp13 = tl.where(tmp6, tmp12, tmp11)
    tmp14 = tl.load(in_ptr2 + (x4), tmp4, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 - tmp14
    tmp16 = tl.load(in_ptr3 + (x4), tmp4, eviction_policy='evict_last', other=0.0)
    tmp17 = 64.0
    tmp18 = (tmp16 / tmp17)
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.rsqrt(tmp20)
    tmp22 = tmp15 * tmp21
    tmp23 = tl.load(in_ptr4 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 * tmp23
    tmp25 = tl.load(in_ptr5 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp26 = tmp24 + tmp25
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp4, tmp26, tmp27)
    tmp29 = tmp0 >= tmp3
    tmp30 = tl.full([1], 128, tl.int64)
    tmp31 = tmp0 < tmp30
    tmp32 = tmp29 & tmp31
    tmp33 = tl.load(in_ptr6 + (64*x5 + ((-64) + x0)), tmp32, eviction_policy='evict_last', other=0.0)
    tmp34 = tmp0 >= tmp30
    tmp35 = tl.full([1], 133, tl.int64)
    tmp36 = tmp0 < tmp35
    tmp37 = tmp34 & tmp36
    tmp38 = tl.load(in_ptr0 + (x4), tmp37, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp39 = tmp38 == 0
    tmp40 = x1 + 8*x2
    tmp41 = tl.full([1], 96, tl.int64)
    tmp42 = tmp40 < tmp41
    tmp43 = tmp42 & tmp37
    tmp44 = tl.load(in_ptr7 + (x1 + 8*x2 + 96*((-128) + x0) + 480*x3), tmp43, eviction_policy='evict_last', other=0.0)
    tmp45 = 0.0
    tmp46 = tl.where(tmp39, tmp45, tmp44)
    tmp47 = tl.full(tmp46.shape, 0.0, tmp46.dtype)
    tmp48 = tl.where(tmp37, tmp46, tmp47)
    tmp49 = tmp0 >= tmp35
    tmp50 = tl.full([1], 134, tl.int64)
    tmp51 = tmp0 < tmp50
    tmp52 = tmp49 & tmp51
    tmp53 = x1
    tmp54 = tmp53.to(tl.float32)
    tmp55 = 8.0
    tmp56 = tmp54 < tmp55
    tmp57 = 0.13333333333333333
    tmp58 = tmp54 * tmp57
    tmp59 = -1.0
    tmp60 = tmp58 + tmp59
    tmp61 = 15 + ((-1)*x1)
    tmp62 = tmp61.to(tl.float32)
    tmp63 = tmp62 * tmp57
    tmp64 = 1.0
    tmp65 = tmp64 - tmp63
    tmp66 = tl.where(tmp56, tmp60, tmp65)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp52, tmp66, tmp67)
    tmp69 = tmp0 >= tmp50
    tmp70 = tl.full([1], 135, tl.int64)
    tmp71 = tmp0 < tmp70
    tmp72 = tmp69 & tmp71
    tmp73 = tl.load(in_ptr8 + (x4), tmp72, eviction_policy='evict_last', other=0.0)
    tmp74 = tl.load(in_ptr9 + (x3), tmp72, eviction_policy='evict_last', other=0.0)
    tmp75 = tmp73 * tmp74
    tmp76 = tmp74 * tmp74
    tmp77 = 0.010000000000000002
    tmp78 = tmp76 + tmp77
    tmp79 = libdevice.sqrt(tmp78)
    tmp80 = (tmp75 / tmp79)
    tmp81 = tl.full(tmp80.shape, 0.0, tmp80.dtype)
    tmp82 = tl.where(tmp72, tmp80, tmp81)
    tmp83 = tmp0 >= tmp70
    tmp84 = tl.full([1], 139, tl.int64)
    tmp85 = tmp0 < tmp84
    tmp86 = tmp83 & tmp85
    tmp87 = tl.load(in_ptr10 + (4*x4 + ((-135) + x0)), tmp86, eviction_policy='evict_last', other=0.0)
    tmp88 = tmp0 >= tmp84
    tmp89 = tl.full([1], 143, tl.int64)
    tmp90 = tmp0 < tmp89
    tmp91 = tl.load(in_ptr11 + (4*x5 + ((-139) + x0)), tmp88, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp92 = tmp91.to(tl.float32)
    tmp93 = tl.full(tmp92.shape, 0.0, tmp92.dtype)
    tmp94 = tl.where(tmp88, tmp92, tmp93)
    tmp95 = tl.where(tmp86, tmp87, tmp94)
    tmp96 = tl.where(tmp72, tmp82, tmp95)
    tmp97 = tl.where(tmp52, tmp68, tmp96)
    tmp98 = tl.where(tmp37, tmp48, tmp97)
    tmp99 = tl.where(tmp32, tmp33, tmp98)
    tmp100 = tl.where(tmp4, tmp28, tmp99)
    tl.store(out_ptr0 + (x6 + 2304*x5), tmp100, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/yz/cyz5tniu3ir4bw2eq24gvtpr2rimtloofqgxlzgeigjpxv2qknmt.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   input_1 => constant_pad_nd_default_4
# Graph fragment:
#   %constant_pad_nd_default_4 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%view_7, [0, 1, 0, 0]), kwargs = {})
triton_poi_fused_addmm_13 = async_compile.triton('triton_poi_fused_addmm_13', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_13(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24772608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 144)
    x1 = xindex // 144
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 143, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x0 + 143*((x1 % 16)) + 2304*(x1 // 16)), tmp2, other=0.0)
    tl.store(out_ptr0 + (x2), tmp3, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/3z/c3zbmq7erjhbahql2hjitjise4tccturz65a6lvfztxotqllq3a5.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   input_1 => constant_pad_nd_default_5
# Graph fragment:
#   %constant_pad_nd_default_5 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%permute_5, [0, 0, 0, 1]), kwargs = {})
triton_poi_fused_addmm_14 = async_compile.triton('triton_poi_fused_addmm_14', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_14(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/cn/ccnsrnlgjspmjfmuhroqthnjwis75ahl5o6uojglj6narst2fyvf.py
# Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   input_2 => add_8, erf, mul_10, mul_8, mul_9
# Graph fragment:
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_8, 0.5), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_8, 0.7071067811865476), kwargs = {})
#   %erf : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_9,), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_8, %add_8), kwargs = {})
triton_poi_fused_gelu_15 = async_compile.triton('triton_poi_fused_gelu_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_15(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 11010048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/ep/cepipuh23akluk5ajyjidb6vxlikqzzasqomcobckym3y5xkjysv.py
# Topologically Sorted Source Nodes: [encoded_3], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   encoded_3 => cat_3
# Graph fragment:
#   %cat_3 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_12, %add_16, %add_21], -2), kwargs = {})
triton_poi_fused_cat_16 = async_compile.triton('triton_poi_fused_cat_16', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 17891328
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
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 16, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tmp10 & tmp12
    tmp14 = tl.load(in_ptr2 + (x0 + 64*((-4) + x1) + 768*x2), tmp13, other=0.0)
    tmp15 = tl.load(in_ptr1 + (64 + x0), tmp13, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp14 + tmp15
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp13, tmp16, tmp17)
    tmp19 = tmp0 >= tmp11
    tmp20 = tl.full([1], 26, tl.int64)
    tmp21 = tmp0 < tmp20
    tmp22 = tl.load(in_ptr3 + (x0 + 64*((-16) + x1) + 640*x2), tmp19, other=0.0)
    tmp23 = tl.load(in_ptr1 + (128 + x0), tmp19, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp19, tmp24, tmp25)
    tmp27 = tl.where(tmp13, tmp18, tmp26)
    tmp28 = tl.where(tmp4, tmp9, tmp27)
    tl.store(out_ptr0 + (x4), tmp28, None)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/hk/chknyblrmd3zcu2nd3axmuhbzpn5txbwxsh4eor3ihy5ced2tkxj.py
# Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul => clone_5
# Graph fragment:
#   %clone_5 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_5,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_17 = async_compile.triton('triton_poi_fused_clone_17', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_17(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 11010048
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


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/j3/cj37t72ubfzcy3y6243kj77lyfpv6x3fbpm5ffay5jkgkiz5rysx.py
# Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul => clone_6
# Graph fragment:
#   %clone_6 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_6,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_18 = async_compile.triton('triton_poi_fused_clone_18', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1048576, 'x': 32}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_18(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (y0 + 64*x2 + 1664*y1), ymask & xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + 26*y3), tmp2, xmask & ymask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/hf/chfgpqgssgw4gnwjuihacokm2wii2krg36bj6vkec54kq6kjsut2.py
# Topologically Sorted Source Nodes: [mass], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   mass => cat_5
# Graph fragment:
#   %cat_5 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%mul_13, %mul_16, %mul_19], -1), kwargs = {})
triton_poi_fused_cat_19 = async_compile.triton('triton_poi_fused_cat_19', '''
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
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/pp/cppoonfwsnkmgmzujzhhe2xwys4tqbmhyna46zhjclhwgjdar32u.py
# Topologically Sorted Source Nodes: [cat_5, attention], Original ATen: [aten.cat, aten._softmax]
# Source node to ATen node mapping:
#   attention => div_3
#   cat_5 => cat_6
# Graph fragment:
#   %cat_6 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%expand_7, %where_14], -1), kwargs = {})
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%cat_6, -1), kwargs = {})
#   %sub_tensor : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_6, %getitem_12), kwargs = {})
#   %exp_default : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor,), kwargs = {})
#   %div_3 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default, %getitem_13), kwargs = {})
triton_per_fused__softmax_cat_20 = async_compile.triton('triton_per_fused__softmax_cat_20', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i1', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_cat_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 4, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax_cat_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/if/cifw7lvul273f3f3yfepjvbphh7a5xwdsvfq32biwcciawp74wo6.py
# Topologically Sorted Source Nodes: [matmul_1], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul_1 => clone_7
# Graph fragment:
#   %clone_7 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_9,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_21 = async_compile.triton('triton_poi_fused_clone_21', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_21(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 17891328
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


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/4z/c4zrqv3o2ahnq63mvaccajlir4wxjiold6s7zdiyvyqntlwwtb6h.py
# Topologically Sorted Source Nodes: [features, input_4], Original ATen: [aten.cat, aten.native_layer_norm]
# Source node to ATen node mapping:
#   features => cat_7
#   input_4 => add_23, add_24, mul_20, mul_21, rsqrt_5, sub_8, var_mean_5
# Graph fragment:
#   %cat_7 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%view_10, %view_31], -1), kwargs = {})
#   %var_mean_5 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%cat_7, [4]), kwargs = {correction: 0, keepdim: True})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_7, %getitem_11), kwargs = {})
#   %add_23 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_10, 1e-05), kwargs = {})
#   %rsqrt_5 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_23,), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %rsqrt_5), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_20, %arg51_1), kwargs = {})
#   %add_24 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_21, %arg52_1), kwargs = {})
triton_red_fused_cat_native_layer_norm_22 = async_compile.triton('triton_red_fused_cat_native_layer_norm_22', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 262144, 'r0_': 128},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_native_layer_norm_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 2, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_cat_native_layer_norm_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 172032
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x3 = xindex
    x0 = (xindex % 16)
    x1 = xindex // 16
    tmp12_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp12_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp12_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = r0_2
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 64, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (64*x3 + (r0_2)), r0_mask & tmp4, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp0 >= tmp3
        tmp7 = tl.full([1, 1], 128, tl.int64)
        tmp8 = tmp0 < tmp7
        tmp9 = tl.load(in_ptr1 + (8*x0 + 128*(((((-64) + r0_2) // 8) % 8)) + 1024*x1 + ((((-64) + r0_2) % 8))), r0_mask & tmp6, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.where(tmp4, tmp5, tmp9)
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp12_mean_next, tmp12_m2_next, tmp12_weight_next = triton_helpers.welford_reduce(
            tmp11, tmp12_mean, tmp12_m2, tmp12_weight, roffset == 0
        )
        tmp12_mean = tl.where(r0_mask, tmp12_mean_next, tmp12_mean)
        tmp12_m2 = tl.where(r0_mask, tmp12_m2_next, tmp12_m2)
        tmp12_weight = tl.where(r0_mask, tmp12_weight_next, tmp12_weight)
    tmp15, tmp16, tmp17 = triton_helpers.welford(tmp12_mean, tmp12_m2, tmp12_weight, 1)
    tmp12 = tmp15[:, None]
    tmp13 = tmp16[:, None]
    tmp14 = tmp17[:, None]
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp36 = tl.load(in_ptr2 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp38 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp18 = r0_2
        tmp19 = tl.full([1, 1], 0, tl.int64)
        tmp20 = tmp18 >= tmp19
        tmp21 = tl.full([1, 1], 64, tl.int64)
        tmp22 = tmp18 < tmp21
        tmp23 = tl.load(in_ptr0 + (64*x3 + (r0_2)), r0_mask & tmp22, eviction_policy='evict_last', other=0.0)
        tmp24 = tmp18 >= tmp21
        tmp25 = tl.full([1, 1], 128, tl.int64)
        tmp26 = tmp18 < tmp25
        tmp27 = tl.load(in_ptr1 + (8*x0 + 128*(((((-64) + r0_2) // 8) % 8)) + 1024*x1 + ((((-64) + r0_2) % 8))), r0_mask & tmp24, eviction_policy='evict_last', other=0.0)
        tmp28 = tl.where(tmp22, tmp23, tmp27)
        tmp29 = tmp28 - tmp12
        tmp30 = 128.0
        tmp31 = (tmp13 / tmp30)
        tmp32 = 1e-05
        tmp33 = tmp31 + tmp32
        tmp34 = libdevice.rsqrt(tmp33)
        tmp35 = tmp29 * tmp34
        tmp37 = tmp35 * tmp36
        tmp39 = tmp37 + tmp38
        tl.store(out_ptr2 + (r0_2 + 128*x3), tmp39, r0_mask)
''', device_str='cuda')


# kernel path: /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008/inductor_cache/w5/cw5x37o4bz6fn6mj7fdg3zvzyu2vm37vvecdrlvr5tlral3abj77.py
# Topologically Sorted Source Nodes: [invert_5, and__6, eligible, correction_1], Original ATen: [aten.bitwise_not, aten.bitwise_and, aten.scalar_tensor, aten.where]
# Source node to ATen node mapping:
#   and__6 => bitwise_and_6
#   correction_1 => full_default_12, where_15
#   eligible => bitwise_and_7
#   invert_5 => bitwise_not_5
# Graph fragment:
#   %bitwise_not_5 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_not.default](args = (%arg1_1,), kwargs = {})
#   %bitwise_and_6 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%arg2_1, %bitwise_not_5), kwargs = {})
#   %bitwise_and_7 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and_6, %unsqueeze_41), kwargs = {})
#   %full_default_12 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_15 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_and_7, %squeeze, %full_default_12), kwargs = {})
triton_poi_fused_bitwise_and_bitwise_not_scalar_tensor_where_23 = async_compile.triton('triton_poi_fused_bitwise_and_bitwise_not_scalar_tensor_where_23', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i1', 'in_ptr1': '*i1', 'in_ptr2': '*i1', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=170, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_and_bitwise_not_scalar_tensor_where_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '521ADE5958B7D85691853587C3BFF053B7D903BABA51246A58177490EECFCE4B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_bitwise_and_bitwise_not_scalar_tensor_where_23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 172032
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1 = args
    args.clear()
    assert_size_stride(arg0_1, (128, 7, 12, 64), (5376, 768, 64, 1))
    assert_size_stride(arg1_1, (128, 7, 12, 16), (1344, 192, 16, 1))
    assert_size_stride(arg2_1, (128, 7, 12, 16), (1344, 192, 16, 1))
    assert_size_stride(arg3_1, (128, 7), (7, 1))
    assert_size_stride(arg4_1, (896, 64, 96), (6144, 96, 1))
    assert_size_stride(arg5_1, (896, 5, 96), (480, 96, 1))
    assert_size_stride(arg6_1, (128, 7, 12, 64), (5376, 768, 64, 1))
    assert_size_stride(arg7_1, (128, 7, 12, 4), (336, 48, 4, 1))
    assert_size_stride(arg8_1, (128, 7, 12, 16, 4), (5376, 768, 64, 4, 1))
    assert_size_stride(arg9_1, (128, 7, 12, 64), (5376, 768, 64, 1))
    assert_size_stride(arg10_1, (128, 7, 12, 64), (5376, 768, 64, 1))
    assert_size_stride(arg11_1, (128, 7, 12, 64), (5376, 768, 64, 1))
    assert_size_stride(arg12_1, (16, 64), (64, 1))
    assert_size_stride(arg13_1, (16, ), (1, ))
    assert_size_stride(arg14_1, (64, ), (1, ))
    assert_size_stride(arg15_1, (64, ), (1, ))
    assert_size_stride(arg16_1, (128, 7, 12, 16), (1344, 192, 16, 1))
    assert_size_stride(arg17_1, (64, ), (1, ))
    assert_size_stride(arg18_1, (64, ), (1, ))
    assert_size_stride(arg19_1, (64, 143), (143, 1))
    assert_size_stride(arg20_1, (64, ), (1, ))
    assert_size_stride(arg21_1, (64, 64), (64, 1))
    assert_size_stride(arg22_1, (64, ), (1, ))
    assert_size_stride(arg23_1, (128, 7, 12), (84, 12, 1))
    assert_size_stride(arg24_1, (128, 7, 12, 4, 64), (21504, 3072, 256, 64, 1))
    assert_size_stride(arg25_1, (128, 7, 12, 4, 5), (1680, 240, 20, 5, 1))
    assert_size_stride(arg26_1, (128, 7, 12, 4), (336, 48, 4, 1))
    assert_size_stride(arg27_1, (128, 7, 12, 4), (336, 48, 4, 1))
    assert_size_stride(arg28_1, (64, ), (1, ))
    assert_size_stride(arg29_1, (64, ), (1, ))
    assert_size_stride(arg30_1, (64, 5), (5, 1))
    assert_size_stride(arg31_1, (64, ), (1, ))
    assert_size_stride(arg32_1, (3, 64), (64, 1))
    assert_size_stride(arg33_1, (128, 7, 12), (84, 12, 1))
    assert_size_stride(arg34_1, (128, 7, 12, 12, 64), (64512, 9216, 768, 64, 1))
    assert_size_stride(arg35_1, (128, 7, 12, 12, 4), (4032, 576, 48, 4, 1))
    assert_size_stride(arg36_1, (128, 7, 12, 12), (1008, 144, 12, 1))
    assert_size_stride(arg37_1, (128, 7, 12, 12), (1008, 144, 12, 1))
    assert_size_stride(arg38_1, (64, 4), (4, 1))
    assert_size_stride(arg39_1, (64, ), (1, ))
    assert_size_stride(arg40_1, (128, 7, 12), (84, 12, 1))
    assert_size_stride(arg41_1, (128, 7, 12, 10, 64), (53760, 7680, 640, 64, 1))
    assert_size_stride(arg42_1, (128, 7, 12, 10, 5), (4200, 600, 50, 5, 1))
    assert_size_stride(arg43_1, (128, 7, 12, 10), (840, 120, 10, 1))
    assert_size_stride(arg44_1, (128, 7, 12, 10), (840, 120, 1, 12))
    assert_size_stride(arg45_1, (64, 5), (5, 1))
    assert_size_stride(arg46_1, (64, ), (1, ))
    assert_size_stride(arg47_1, (64, 64), (64, 1))
    assert_size_stride(arg48_1, (64, ), (1, ))
    assert_size_stride(arg49_1, (64, 64), (64, 1))
    assert_size_stride(arg50_1, (64, ), (1, ))
    assert_size_stride(arg51_1, (128, ), (1, ))
    assert_size_stride(arg52_1, (128, ), (1, ))
    assert_size_stride(arg53_1, (64, 128), (128, 1))
    assert_size_stride(arg54_1, (64, ), (1, ))
    assert_size_stride(arg55_1, (1, 64), (64, 1))
    assert_size_stride(arg56_1, (1, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf16 = empty_strided_cuda((128, 7, 12, 64), (5376, 768, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_layer_norm_0.run(arg6_1, arg14_1, arg15_1, buf16, 10752, 64, stream=stream0)
        del arg14_1
        del arg15_1
        del arg6_1
        buf3 = empty_strided_cuda((128, 7, 12, 16, 1), (1344, 192, 16, 1, 172032), torch.float32)
        buf4 = empty_strided_cuda((128, 7, 12, 16, 1), (1344, 192, 16, 1, 172032), torch.float32)
        # Topologically Sorted Source Nodes: [invert, context, layer_norm_1], Original ATen: [aten.bitwise_not, aten.masked_fill, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_bitwise_not_masked_fill_native_layer_norm_1.run(arg2_1, arg4_1, buf3, buf4, 172032, 64, stream=stream0)
        buf23 = empty_strided_cuda((128, 7, 12, 4, 5), (1696, 240, 20, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [quality], Original ATen: [aten.scalar_tensor, aten.where]
        stream0 = get_raw_stream(0)
        triton_poi_fused_scalar_tensor_where_2.run(arg27_1, arg23_1, arg7_1, arg25_1, buf23, 215040, stream=stream0)
        del arg25_1
        buf24 = empty_strided_cuda((43008, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_3.run(buf23, buf24, 344064, stream=stream0)
        del buf23
        buf25 = empty_strided_cuda((8, 64), (1, 8), torch.float32)
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_4.run(arg30_1, buf25, 512, stream=stream0)
        del arg30_1
        buf26 = empty_strided_cuda((43008, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.addmm]
        extern_kernels.mm(buf24, buf25, out=buf26)
        del buf24
        del buf25
        buf27 = reinterpret_tensor(buf26, (128, 7, 12, 4, 64), (21504, 3072, 256, 64, 1), 0); del buf26  # reuse
        # Topologically Sorted Source Nodes: [raw, layer_norm_2, add_3], Original ATen: [aten.scalar_tensor, aten.where, aten.native_layer_norm, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_scalar_tensor_where_5.run(buf27, arg27_1, arg23_1, arg7_1, arg24_1, arg28_1, arg29_1, arg31_1, 43008, 64, stream=stream0)
        del arg24_1
        del arg31_1
        buf28 = empty_strided_cuda((128, 7, 12, 12, 4), (4032, 576, 48, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [quality_1], Original ATen: [aten.scalar_tensor, aten.where]
        stream0 = get_raw_stream(0)
        triton_poi_fused_scalar_tensor_where_6.run(arg37_1, arg33_1, arg7_1, arg35_1, buf28, 516096, stream=stream0)
        del arg35_1
        buf29 = empty_strided_cuda((129024, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf28, (129024, 4), (4, 1), 0), reinterpret_tensor(arg38_1, (4, 64), (1, 4), 0), out=buf29)
        del arg38_1
        del buf28
        buf30 = reinterpret_tensor(buf29, (128, 7, 12, 12, 64), (64512, 9216, 768, 64, 1), 0); del buf29  # reuse
        # Topologically Sorted Source Nodes: [raw_1, layer_norm_3, add_5], Original ATen: [aten.scalar_tensor, aten.where, aten.native_layer_norm, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_scalar_tensor_where_7.run(buf30, arg37_1, arg33_1, arg7_1, arg34_1, arg28_1, arg29_1, arg39_1, 129024, 64, stream=stream0)
        del arg34_1
        del arg39_1
        buf31 = empty_strided_cuda((10752, 12, 8), (96, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten.bmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bmm_8.run(arg44_1, arg40_1, arg7_1, arg42_1, buf31, 1032192, stream=stream0)
        del arg42_1
        buf32 = empty_strided_cuda((10752, 8, 64), (512, 1, 8), torch.float32)
        # Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten.bmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bmm_9.run(arg45_1, buf32, 5505024, stream=stream0)
        del arg45_1
        buf33 = empty_strided_cuda((10752, 12, 64), (768, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf31, buf32, out=buf33)
        del buf31
        del buf32
        buf34 = empty_strided_cuda((128, 7, 12, 10, 64), (53760, 7680, 640, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [raw_2, layer_norm_4, linear_6, add_7], Original ATen: [aten.scalar_tensor, aten.where, aten.native_layer_norm, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_scalar_tensor_where_10.run(arg44_1, arg40_1, arg7_1, arg41_1, arg28_1, arg29_1, buf33, arg46_1, buf34, 107520, 64, stream=stream0)
        del arg28_1
        del arg29_1
        del arg41_1
        del arg46_1
        del buf33
        buf15 = empty_strided_cuda((128, 7, 12, 26), (2184, 312, 26, 1), torch.bool)
        buf50 = empty_strided_cuda((128, 7, 12), (84, 12, 1), torch.bool)
        # Topologically Sorted Source Nodes: [valid_3, any_1], Original ATen: [aten.cat, aten.any]
        stream0 = get_raw_stream(0)
        triton_per_fused_any_cat_11.run(arg27_1, arg23_1, arg7_1, arg37_1, arg33_1, arg44_1, arg40_1, buf15, buf50, 10752, 26, stream=stream0)
        buf17 = empty_strided_cuda((128, 7, 12, 16, 143), (193536, 27648, 2304, 143, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_12.run(arg2_1, arg4_1, buf3, buf4, arg17_1, arg18_1, buf16, arg5_1, arg16_1, arg3_1, arg8_1, arg7_1, buf17, 24600576, stream=stream0)
        del arg16_1
        del arg17_1
        del arg18_1
        del arg3_1
        del arg4_1
        del arg5_1
        del buf16
        del buf3
        buf18 = empty_strided_cuda((172032, 144), (144, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_13.run(buf17, buf18, 24772608, stream=stream0)
        del buf17
        buf19 = empty_strided_cuda((144, 64), (1, 144), torch.float32)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_14.run(arg19_1, buf19, 9216, stream=stream0)
        del arg19_1
        buf20 = empty_strided_cuda((172032, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.addmm]
        extern_kernels.mm(buf18, buf19, out=buf20)
        del buf18
        del buf19
        buf21 = reinterpret_tensor(buf20, (128, 7, 12, 16, 64), (86016, 12288, 1024, 64, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_15.run(buf21, arg20_1, 11010048, stream=stream0)
        del arg20_1
        buf22 = empty_strided_cuda((172032, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg22_1, reinterpret_tensor(buf21, (172032, 64), (64, 1), 0), reinterpret_tensor(arg21_1, (64, 64), (1, 64), 0), alpha=1, beta=1, out=buf22)
        del arg21_1
        del arg22_1
        buf35 = empty_strided_cuda((128, 7, 12, 26, 64), (139776, 19968, 1664, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [encoded_3], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_16.run(buf27, arg32_1, buf30, buf34, buf35, 17891328, stream=stream0)
        del arg32_1
        del buf27
        del buf30
        del buf34
        buf36 = empty_strided_cuda((279552, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf35, (279552, 64), (64, 1), 0), reinterpret_tensor(arg47_1, (64, 64), (1, 64), 0), out=buf36)
        del arg47_1
        buf37 = reinterpret_tensor(buf21, (128, 7, 12, 8, 16, 8), (86016, 12288, 1024, 128, 8, 1), 0); del buf21  # reuse
        # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_17.run(buf22, buf37, 11010048, stream=stream0)
        buf38 = empty_strided_cuda((128, 7, 12, 8, 8, 26), (139776, 19968, 1664, 208, 26, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_18.run(buf36, arg48_1, buf38, 688128, 26, stream=stream0)
        del arg48_1
        del buf36
        buf39 = empty_strided_cuda((86016, 16, 26), (416, 26, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf37, (86016, 16, 8), (128, 8, 1), 0), reinterpret_tensor(buf38, (86016, 8, 26), (208, 26, 1), 0), out=buf39)
        buf40 = empty_strided_cuda((128, 7, 12, 16, 26), (34944, 4992, 416, 26, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mass], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_19.run(arg8_1, arg27_1, arg23_1, arg7_1, arg26_1, arg37_1, arg33_1, arg36_1, arg44_1, arg40_1, arg43_1, buf40, 4472832, stream=stream0)
        del arg23_1
        del arg26_1
        del arg27_1
        del arg33_1
        del arg36_1
        del arg37_1
        del arg40_1
        del arg43_1
        del arg44_1
        del arg7_1
        buf43 = empty_strided_cuda((128, 7, 12, 8, 16, 27), (290304, 41472, 3456, 432, 27, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_5, attention], Original ATen: [aten.cat, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_cat_20.run(arg8_1, buf15, buf39, buf40, buf43, 1376256, 27, stream=stream0)
        del arg8_1
        del buf39
        del buf40
        buf44 = reinterpret_tensor(buf38, (279552, 64), (64, 1), 0); del buf38  # reuse
        # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf35, (279552, 64), (64, 1), 0), reinterpret_tensor(arg49_1, (64, 64), (1, 64), 0), out=buf44)
        del arg49_1
        buf45 = reinterpret_tensor(buf35, (128, 7, 12, 8, 26, 8), (139776, 19968, 1664, 208, 8, 1), 0); del buf35  # reuse
        # Topologically Sorted Source Nodes: [matmul_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_21.run(buf44, arg50_1, buf45, 17891328, stream=stream0)
        del arg50_1
        del buf44
        buf46 = reinterpret_tensor(buf37, (86016, 16, 8), (128, 8, 1), 0); del buf37  # reuse
        # Topologically Sorted Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf43, (86016, 16, 26), (432, 27, 1), 1), reinterpret_tensor(buf45, (86016, 26, 8), (208, 8, 1), 0), out=buf46)
        del buf45
        buf51 = empty_strided_cuda((128, 7, 12, 16, 128), (172032, 24576, 2048, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features, input_4], Original ATen: [aten.cat, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_cat_native_layer_norm_22.run(buf22, buf46, arg51_1, arg52_1, buf51, 172032, 128, stream=stream0)
        del arg51_1
        del arg52_1
        del buf22
        buf52 = reinterpret_tensor(buf46, (172032, 64), (64, 1), 0); del buf46  # reuse
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf51, (172032, 128), (128, 1), 0), reinterpret_tensor(arg53_1, (128, 64), (1, 128), 0), out=buf52)
        del arg53_1
        del buf51
        buf53 = reinterpret_tensor(buf52, (128, 7, 12, 16, 64), (86016, 12288, 1024, 64, 1), 0); del buf52  # reuse
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_15.run(buf53, arg54_1, 11010048, stream=stream0)
        del arg54_1
        buf54 = reinterpret_tensor(buf4, (172032, 1), (1, 1), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf53, (172032, 64), (64, 1), 0), reinterpret_tensor(arg55_1, (64, 1), (1, 64), 0), out=buf54)
        del arg55_1
        del buf53
        buf55 = reinterpret_tensor(buf54, (128, 7, 12, 16), (1344, 192, 16, 1), 0); del buf54  # reuse
        # Topologically Sorted Source Nodes: [invert_5, and__6, eligible, correction_1], Original ATen: [aten.bitwise_not, aten.bitwise_and, aten.scalar_tensor, aten.where]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bitwise_and_bitwise_not_scalar_tensor_where_23.run(buf55, arg2_1, arg1_1, buf50, arg56_1, 172032, stream=stream0)
        del arg1_1
        del arg2_1
        del arg56_1
        del buf50
    return (buf55, buf43, buf15, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((128, 7, 12, 64), (5376, 768, 64, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((128, 7, 12, 16), (1344, 192, 16, 1), device='cuda:0', dtype=torch.bool)
    arg2_1 = rand_strided((128, 7, 12, 16), (1344, 192, 16, 1), device='cuda:0', dtype=torch.bool)
    arg3_1 = rand_strided((128, 7), (7, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((896, 64, 96), (6144, 96, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((896, 5, 96), (480, 96, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((128, 7, 12, 64), (5376, 768, 64, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((128, 7, 12, 4), (336, 48, 4, 1), device='cuda:0', dtype=torch.bool)
    arg8_1 = rand_strided((128, 7, 12, 16, 4), (5376, 768, 64, 4, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((128, 7, 12, 64), (5376, 768, 64, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((128, 7, 12, 64), (5376, 768, 64, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((128, 7, 12, 64), (5376, 768, 64, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((16, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((128, 7, 12, 16), (1344, 192, 16, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((64, 143), (143, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((128, 7, 12), (84, 12, 1), device='cuda:0', dtype=torch.bool)
    arg24_1 = rand_strided((128, 7, 12, 4, 64), (21504, 3072, 256, 64, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((128, 7, 12, 4, 5), (1680, 240, 20, 5, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((128, 7, 12, 4), (336, 48, 4, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((128, 7, 12, 4), (336, 48, 4, 1), device='cuda:0', dtype=torch.bool)
    arg28_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((64, 5), (5, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((3, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((128, 7, 12), (84, 12, 1), device='cuda:0', dtype=torch.bool)
    arg34_1 = rand_strided((128, 7, 12, 12, 64), (64512, 9216, 768, 64, 1), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((128, 7, 12, 12, 4), (4032, 576, 48, 4, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((128, 7, 12, 12), (1008, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((128, 7, 12, 12), (1008, 144, 12, 1), device='cuda:0', dtype=torch.bool)
    arg38_1 = rand_strided((64, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((128, 7, 12), (84, 12, 1), device='cuda:0', dtype=torch.bool)
    arg41_1 = rand_strided((128, 7, 12, 10, 64), (53760, 7680, 640, 64, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((128, 7, 12, 10, 5), (4200, 600, 50, 5, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((128, 7, 12, 10), (840, 120, 10, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((128, 7, 12, 10), (840, 120, 1, 12), device='cuda:0', dtype=torch.bool)
    arg45_1 = rand_strided((64, 5), (5, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((64, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((1, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
