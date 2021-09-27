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

REGISTER_CONVERTER(conv2d,
                   ConvertConv2D,
                   "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_"
                   "npu,amlogic_npu,imagination_nna");
REGISTER_CONVERTER(depthwise_conv2d,
                   ConvertConv2D,
                   "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_"
                   "npu,amlogic_npu,imagination_nna");
REGISTER_CONVERTER(deformable_conv, ConvertDeformableConv, "huawei_ascend_npu");
REGISTER_CONVERTER(matmul, ConvertMatmul, "huawei_ascend_npu");
REGISTER_CONVERTER(matmul_v2, ConvertMatmulV2, "huawei_ascend_npu");
REGISTER_CONVERTER(softmax,
                   ConvertSoftmax,
                   "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_"
                   "npu,amlogic_npu,imagination_nna");
REGISTER_CONVERTER(cumsum, ConvertCumsum, "huawei_ascend_npu");
REGISTER_CONVERTER(
    reshape,
    ConvertReshape,
    "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_npu,amlogic_npu");
REGISTER_CONVERTER(
    reshape2,
    ConvertReshape,
    "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_npu,amlogic_npu");
REGISTER_CONVERTER(unsqueeze, ConvertUnsqueeze, "huawei_ascend_npu");
REGISTER_CONVERTER(unsqueeze2, ConvertUnsqueeze, "huawei_ascend_npu");
REGISTER_CONVERTER(lookup_table_v2, ConvertLookupTableV2, "huawei_ascend_npu");
REGISTER_CONVERTER(
    elementwise_add,
    ConvertElementwise,
    "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_npu,amlogic_npu");
REGISTER_CONVERTER(
    elementwise_sub,
    ConvertElementwise,
    "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_npu,amlogic_npu");
REGISTER_CONVERTER(
    elementwise_mul,
    ConvertElementwise,
    "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_npu,amlogic_npu");
REGISTER_CONVERTER(
    elementwise_div,
    ConvertElementwise,
    "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_npu,amlogic_npu");
REGISTER_CONVERTER(elementwise_max, ConvertElementwise, "huawei_ascend_npu");
REGISTER_CONVERTER(elementwise_min, ConvertElementwise, "huawei_ascend_npu");
REGISTER_CONVERTER(elementwise_pow, ConvertElementwise, "huawei_ascend_npu");
REGISTER_CONVERTER(
    fusion_elementwise_add_activation,
    ConvertElementwise,
    "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_npu,amlogic_npu");
REGISTER_CONVERTER(
    fusion_elementwise_sub_activation,
    ConvertElementwise,
    "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_npu,amlogic_npu");
REGISTER_CONVERTER(
    fusion_elementwise_mul_activation,
    ConvertElementwise,
    "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_npu,amlogic_npu");
REGISTER_CONVERTER(
    fusion_elementwise_div_activation,
    ConvertElementwise,
    "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_npu,amlogic_npu");
REGISTER_CONVERTER(fusion_elementwise_min_activation,
                   ConvertElementwise,
                   "huawei_ascend_npu");
REGISTER_CONVERTER(fusion_elementwise_max_activation,
                   ConvertElementwise,
                   "huawei_ascend_npu");
REGISTER_CONVERTER(fusion_elementwise_pow_activation,
                   ConvertElementwise,
                   "huawei_ascend_npu");
REGISTER_CONVERTER(
    sigmoid,
    ConvertUnaryActivations,
    "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_npu,amlogic_npu");
REGISTER_CONVERTER(relu,
                   ConvertUnaryActivations,
                   "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_"
                   "npu,amlogic_npu,imagination_nna");
REGISTER_CONVERTER(relu6,
                   ConvertUnaryActivations,
                   "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_"
                   "npu,amlogic_npu,imagination_nna");
REGISTER_CONVERTER(leaky_relu, ConvertLeakyRelu, "huawei_ascend_npu");
REGISTER_CONVERTER(
    tanh,
    ConvertUnaryActivations,
    "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_npu,amlogic_npu");
REGISTER_CONVERTER(abs, ConvertUnaryActivations, "huawei_ascend_npu");
REGISTER_CONVERTER(exp, ConvertUnaryActivations, "huawei_ascend_npu");
REGISTER_CONVERTER(instance_norm, ConvertInstanceNorm, "huawei_ascend_npu");
REGISTER_CONVERTER(layer_norm, ConvertLayerNorm, "huawei_ascend_npu");
REGISTER_CONVERTER(log, ConvertUnaryActivations, "huawei_ascend_npu");
REGISTER_CONVERTER(swish, ConvertUnaryActivations, "huawei_ascend_npu");
REGISTER_CONVERTER(prelu, ConvertPRelu, "huawei_ascend_npu");
REGISTER_CONVERTER(gelu, ConvertGelu, "huawei_ascend_npu");
REGISTER_CONVERTER(hard_sigmoid, ConvertHardSigmoid, "huawei_ascend_npu");
REGISTER_CONVERTER(hard_swish, ConvertHardSwish, "huawei_ascend_npu");
REGISTER_CONVERTER(arg_max, ConvertArgMinMax, "huawei_ascend_npu");
REGISTER_CONVERTER(arg_min, ConvertArgMinMax, "huawei_ascend_npu");
REGISTER_CONVERTER(equal, ConvertComparisons, "huawei_ascend_npu");
REGISTER_CONVERTER(not_equal, ConvertComparisons, "huawei_ascend_npu");
REGISTER_CONVERTER(greater_than, ConvertComparisons, "huawei_ascend_npu");
REGISTER_CONVERTER(greater_equal, ConvertComparisons, "huawei_ascend_npu");
REGISTER_CONVERTER(less_than, ConvertComparisons, "huawei_ascend_npu");
REGISTER_CONVERTER(less_equal, ConvertComparisons, "huawei_ascend_npu");
REGISTER_CONVERTER(less_than, ConvertComparisons, "huawei_ascend_npu");
REGISTER_CONVERTER(top_k, ConvertTopK, "huawei_ascend_npu");
REGISTER_CONVERTER(top_k_v2, ConvertTopK, "huawei_ascend_npu");
REGISTER_CONVERTER(shape, ConvertShape, "huawei_ascend_npu");
REGISTER_CONVERTER(slice, ConvertSlice, "huawei_ascend_npu");
REGISTER_CONVERTER(squeeze, ConvertSqueeze, "huawei_ascend_npu");
REGISTER_CONVERTER(squeeze2, ConvertSqueeze, "huawei_ascend_npu");
REGISTER_CONVERTER(fill_constant, ConvertFillConstant, "huawei_ascend_npu");
REGISTER_CONVERTER(fill_any_like, ConvertFillAnyLike, "huawei_ascend_npu");
REGISTER_CONVERTER(
    flatten,
    ConvertFlatten,
    "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_npu,amlogic_npu");
REGISTER_CONVERTER(
    flatten2,
    ConvertFlatten,
    "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_npu,amlogic_npu");
REGISTER_CONVERTER(
    flatten_contiguous_range,
    ConvertFlattenContiguousRange,
    "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_npu,amlogic_npu");

#endif  // NOLINT
