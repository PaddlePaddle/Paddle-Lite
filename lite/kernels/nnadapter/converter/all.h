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
REGISTER_CONVERTER(softmax,
                   ConvertSoftmax,
                   "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_"
                   "npu,amlogic_npu,imagination_nna");
REGISTER_CONVERTER(cumsum, ConvertCumsum, "huawei_ascend_npu");
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
REGISTER_CONVERTER(
    tanh,
    ConvertUnaryActivations,
    "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_npu,amlogic_npu");
REGISTER_CONVERTER(abs, ConvertUnaryActivations, "huawei_ascend_npu");
REGISTER_CONVERTER(exp, ConvertUnaryActivations, "huawei_ascend_npu");
REGISTER_CONVERTER(log, ConvertUnaryActivations, "huawei_ascend_npu");
REGISTER_CONVERTER(shape, ConvertShape, "huawei_ascend_npu");
REGISTER_CONVERTER(fill_constant, ConvertFillConstant, "huawei_ascend_npu");
REGISTER_CONVERTER(fill_any_like, ConvertFillAnyLike, "huawei_ascend_npu");

#endif  // NOLINT
