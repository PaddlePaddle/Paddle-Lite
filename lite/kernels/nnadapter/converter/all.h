// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef __NNADAPTER_CONVERTER_ALL_H__  // NOLINT
#define __NNADAPTER_CONVERTER_ALL_H__

REGISTER_CONVERTER(batch_norm, ConvertBatchNorm, "huawei_ascend_npu");
REGISTER_CONVERTER(cast, ConvertCast, "huawei_ascend_npu");
REGISTER_CONVERTER(clip, ConvertClip, "huawei_ascend_npu");
REGISTER_CONVERTER(conv2d,
                   ConvertConv2D,
                   "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_"
                   "npu,amlogic_npu,imagination_nna,cambricon_mlu,verisilicon_"
                   "timvx");
REGISTER_CONVERTER(depthwise_conv2d,
                   ConvertConv2D,
                   "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_"
                   "npu,amlogic_npu,imagination_nna,verisilicon_timvx");
REGISTER_CONVERTER(deformable_conv, ConvertDeformableConv, "huawei_ascend_npu");
REGISTER_CONVERTER(dropout, ConvertDropout, "huawei_ascend_npu");
REGISTER_CONVERTER(pool2d,
                   ConvertPool,
                   "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_"
                   "npu,amlogic_npu,imagination_nna,cambricon_mlu,verisilicon_"
                   "timvx");
REGISTER_CONVERTER(matmul,
                   ConvertMatmul,
                   "huawei_ascend_npu,huawei_kirin_npu,imagination_nna");
REGISTER_CONVERTER(matmul_v2,
                   ConvertMatmulV2,
                   "huawei_ascend_npu,huawei_kirin_npu,imagination_nna");
REGISTER_CONVERTER(softmax,
                   ConvertSoftmax,
                   "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_"
                   "npu,amlogic_npu,imagination_nna,cambricon_mlu,verisilicon_"
                   "timvx");
REGISTER_CONVERTER(cumsum, ConvertCumsum, "huawei_ascend_npu");
REGISTER_CONVERTER(conv2d_transpose,
                   ConvertConv2dTranspose,
                   "huawei_ascend_npu,amlogic_npu");
REGISTER_CONVERTER(reshape,
                   ConvertReshape,
                   "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_"
                   "npu,amlogic_npu,imagination_nna,verisilicon_timvx");
REGISTER_CONVERTER(reshape2,
                   ConvertReshape,
                   "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_"
                   "npu,amlogic_npu,imagination_nna,verisilicon_timvx");
REGISTER_CONVERTER(unsqueeze,
                   ConvertUnsqueeze,
                   "huawei_ascend_npu,cambricon_mlu");
REGISTER_CONVERTER(unsqueeze2,
                   ConvertUnsqueeze,
                   "huawei_ascend_npu,cambricon_mlu");
REGISTER_CONVERTER(lookup_table_v2, ConvertLookupTableV2, "huawei_ascend_npu");
REGISTER_CONVERTER(elementwise_add,
                   ConvertElementwise,
                   "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_"
                   "npu,amlogic_npu,imagination_nna,cambricon_mlu,verisilicon_"
                   "timvx");
REGISTER_CONVERTER(elementwise_sub,
                   ConvertElementwise,
                   "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_"
                   "npu,amlogic_npu,imagination_nna,cambricon_mlu,verisilicon_"
                   "timvx");
REGISTER_CONVERTER(elementwise_mul,
                   ConvertElementwise,
                   "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_"
                   "npu,amlogic_npu,imagination_nna,cambricon_mlu,verisilicon_"
                   "timvx");
REGISTER_CONVERTER(elementwise_div,
                   ConvertElementwise,
                   "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_"
                   "npu,amlogic_npu,imagination_nna,verisilicon_timvx");
REGISTER_CONVERTER(elementwise_max,
                   ConvertElementwise,
                   "huawei_ascend_npu,imagination_nna");
REGISTER_CONVERTER(elementwise_min,
                   ConvertElementwise,
                   "huawei_ascend_npu,imagination_nna");
REGISTER_CONVERTER(elementwise_pow, ConvertElementwise, "huawei_ascend_npu");
REGISTER_CONVERTER(fusion_elementwise_add_activation,
                   ConvertElementwise,
                   "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_"
                   "npu,amlogic_npu,imagination_nna,verisilicon_timvx");
REGISTER_CONVERTER(fusion_elementwise_sub_activation,
                   ConvertElementwise,
                   "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_"
                   "npu,amlogic_npu,imagination_nna,verisilicon_timvx");
REGISTER_CONVERTER(fusion_elementwise_mul_activation,
                   ConvertElementwise,
                   "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_"
                   "npu,amlogic_npu,imagination_nna,verisilicon_timvx");
REGISTER_CONVERTER(fusion_elementwise_div_activation,
                   ConvertElementwise,
                   "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_"
                   "npu,amlogic_npu,imagination_nna,verisilicon_timvx");
REGISTER_CONVERTER(fusion_elementwise_min_activation,
                   ConvertElementwise,
                   "huawei_ascend_npu,imagination_nna");
REGISTER_CONVERTER(fusion_elementwise_max_activation,
                   ConvertElementwise,
                   "huawei_ascend_npu,imagination_nna");
REGISTER_CONVERTER(fusion_elementwise_pow_activation,
                   ConvertElementwise,
                   "huawei_ascend_npu");
REGISTER_CONVERTER(pow, ConvertPow, "huawei_ascend_npu");
REGISTER_CONVERTER(sigmoid,
                   ConvertUnaryActivations,
                   "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_"
                   "npu,amlogic_npu,cambricon_mlu,verisilicon_timvx");
REGISTER_CONVERTER(relu,
                   ConvertUnaryActivations,
                   "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_"
                   "npu,amlogic_npu,imagination_nna,cambricon_mlu,verisilicon_"
                   "timvx");
REGISTER_CONVERTER(relu6,
                   ConvertUnaryActivations,
                   "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_"
                   "npu,amlogic_npu,imagination_nna,cambricon_mlu,verisilicon_"
                   "timvx");
REGISTER_CONVERTER(leaky_relu, ConvertLeakyRelu, "huawei_ascend_npu");
REGISTER_CONVERTER(tanh,
                   ConvertUnaryActivations,
                   "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_"
                   "npu,amlogic_npu,cambricon_mlu,verisilicon_timvx");
REGISTER_CONVERTER(abs, ConvertUnaryActivations, "huawei_ascend_npu");
REGISTER_CONVERTER(exp, ConvertUnaryActivations, "huawei_ascend_npu");
REGISTER_CONVERTER(instance_norm, ConvertInstanceNorm, "huawei_ascend_npu");
REGISTER_CONVERTER(layer_norm, ConvertLayerNorm, "huawei_ascend_npu");
REGISTER_CONVERTER(group_norm, ConvertGroupNorm, "huawei_ascend_npu");
REGISTER_CONVERTER(log, ConvertUnaryActivations, "huawei_ascend_npu");
REGISTER_CONVERTER(swish, ConvertUnaryActivations, "huawei_ascend_npu");
REGISTER_CONVERTER(prelu, ConvertPRelu, "huawei_ascend_npu");
REGISTER_CONVERTER(gelu, ConvertGelu, "huawei_ascend_npu");
REGISTER_CONVERTER(hard_sigmoid, ConvertHardSigmoid, "huawei_ascend_npu");
REGISTER_CONVERTER(hard_swish, ConvertHardSwish, "huawei_ascend_npu");
REGISTER_CONVERTER(arg_max, ConvertArgMinMax, "huawei_ascend_npu");
REGISTER_CONVERTER(arg_min, ConvertArgMinMax, "huawei_ascend_npu");
REGISTER_CONVERTER(assign, ConvertAssign, "huawei_ascend_npu");
REGISTER_CONVERTER(equal, ConvertComparisons, "huawei_ascend_npu");
REGISTER_CONVERTER(expand_v2, ConvertExpandV2, "huawei_ascend_npu");
REGISTER_CONVERTER(not_equal, ConvertComparisons, "huawei_ascend_npu");
REGISTER_CONVERTER(greater_than, ConvertComparisons, "huawei_ascend_npu");
REGISTER_CONVERTER(greater_equal, ConvertComparisons, "huawei_ascend_npu");
REGISTER_CONVERTER(less_than, ConvertComparisons, "huawei_ascend_npu");
REGISTER_CONVERTER(less_equal, ConvertComparisons, "huawei_ascend_npu");
REGISTER_CONVERTER(less_than, ConvertComparisons, "huawei_ascend_npu");
REGISTER_CONVERTER(reduce_mean, ConvertReduce, "huawei_ascend_npu");
REGISTER_CONVERTER(reduce_sum, ConvertReduce, "huawei_ascend_npu");
REGISTER_CONVERTER(top_k, ConvertTopK, "huawei_ascend_npu");
REGISTER_CONVERTER(top_k_v2, ConvertTopK, "huawei_ascend_npu");
REGISTER_CONVERTER(scale,
                   ConvertScale,
                   "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_"
                   "npu,amlogic_npu,verisilicon_timvx");
REGISTER_CONVERTER(transpose,
                   ConvertTranspose,
                   "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_"
                   "npu,amlogic_npu,verisilicon_timvx");
REGISTER_CONVERTER(transpose2,
                   ConvertTranspose,
                   "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_"
                   "npu,amlogic_npu,verisilicon_timvx");
REGISTER_CONVERTER(shape, ConvertShape, "huawei_ascend_npu");
REGISTER_CONVERTER(slice, ConvertSlice, "huawei_ascend_npu");
REGISTER_CONVERTER(strided_slice, ConvertStridedSlice, "huawei_ascend_npu");
REGISTER_CONVERTER(squeeze, ConvertSqueeze, "huawei_ascend_npu");
REGISTER_CONVERTER(squeeze2, ConvertSqueeze, "huawei_ascend_npu");
REGISTER_CONVERTER(range, ConvertRange, "huawei_ascend_npu");
REGISTER_CONVERTER(stack, ConvertStack, "huawei_ascend_npu");
REGISTER_CONVERTER(fill_constant, ConvertFillConstant, "huawei_ascend_npu");
REGISTER_CONVERTER(fill_any_like, ConvertFillAnyLike, "huawei_ascend_npu");
REGISTER_CONVERTER(fill_constant_batch_size_like,
                   ConvertFillConstantBatchSizeLike,
                   "huawei_ascend_npu");
REGISTER_CONVERTER(concat,
                   ConvertConcat,
                   "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_"
                   "npu,amlogic_npu,verisilicon_timvx");
REGISTER_CONVERTER(split, ConvertSplit, "huawei_kirin_npu,huawei_ascend_npu");
REGISTER_CONVERTER(calib, ConvertCalib, "huawei_ascend_npu");
REGISTER_CONVERTER(nearest_interp, ConvertInterpolate, "huawei_ascend_npu");
REGISTER_CONVERTER(nearest_interp_v2, ConvertInterpolate, "huawei_ascend_npu");
REGISTER_CONVERTER(bilinear_interp, ConvertInterpolate, "huawei_ascend_npu");
REGISTER_CONVERTER(bilinear_interp_v2, ConvertInterpolate, "huawei_ascend_npu");
REGISTER_CONVERTER(flatten,
                   ConvertFlatten,
                   "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_"
                   "npu,amlogic_npu,verisilicon_timvx");
REGISTER_CONVERTER(flatten2,
                   ConvertFlatten,
                   "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_"
                   "npu,amlogic_npu,verisilicon_timvx");
REGISTER_CONVERTER(flatten_contiguous_range,
                   ConvertFlattenContiguousRange,
                   "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_"
                   "npu,amlogic_npu,verisilicon_timvx");
REGISTER_CONVERTER(fc,
                   ConvertFC,
                   "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_"
                   "npu,amlogic_npu,imagination_nna,cambricon_mlu,verisilicon_"
                   "timvx");
REGISTER_CONVERTER(norm, ConvertNorm, "huawei_ascend_npu");
REGISTER_CONVERTER(pad2d, ConvertPad, "huawei_ascend_npu");
REGISTER_CONVERTER(pad3d, ConvertPad, "huawei_ascend_npu");
REGISTER_CONVERTER(gather, ConvertGather, "huawei_ascend_npu");
REGISTER_CONVERTER(logical_not, ConvertUnaryLogicalOp, "huawei_ascend_npu");
REGISTER_CONVERTER(logical_and, ConvertBinaryLogicalOp, "huawei_ascend_npu");
REGISTER_CONVERTER(floor, ConvertUnaryActivations, "huawei_ascend_npu");
REGISTER_CONVERTER(meshgrid, ConvertMeshgrid, "huawei_ascend_npu");
REGISTER_CONVERTER(square, ConvertUnaryActivations, "huawei_ascend_npu");
REGISTER_CONVERTER(tile, ConvertTile, "huawei_ascend_npu");
REGISTER_CONVERTER(sum, ConvertSum, "huawei_ascend_npu");
REGISTER_CONVERTER(roi_align, ConvertRoiAlign, "huawei_ascend_npu");
REGISTER_CONVERTER(grid_sample, ConvertGridSample, "huawei_ascend_npu");
#endif  // NOLINT
