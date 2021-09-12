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

#pragma once

USE_SUBGRAPH_BRIDGE(conv2d,
                    kNNAdapter,
                    "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_"
                    "npu,amlogic_npu,imagination_nna");
USE_SUBGRAPH_BRIDGE(depthwise_conv2d,
                    kNNAdapter,
                    "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_"
                    "npu,amlogic_npu,imagination_nna");
USE_SUBGRAPH_BRIDGE(fc,
                    kNNAdapter,
                    "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_"
                    "npu,amlogic_npu,imagination_nna");
USE_SUBGRAPH_BRIDGE(pool2d,
                    kNNAdapter,
                    "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_"
                    "npu,amlogic_npu,imagination_nna");
USE_SUBGRAPH_BRIDGE(
    elementwise_add,
    kNNAdapter,
    "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_npu,amlogic_npu");
USE_SUBGRAPH_BRIDGE(
    elementwise_sub,
    kNNAdapter,
    "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_npu,amlogic_npu");
USE_SUBGRAPH_BRIDGE(
    elementwise_mul,
    kNNAdapter,
    "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_npu,amlogic_npu");
USE_SUBGRAPH_BRIDGE(
    elementwise_div,
    kNNAdapter,
    "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_npu,amlogic_npu");
USE_SUBGRAPH_BRIDGE(elementwise_max, kNNAdapter, "huawei_ascend_npu");
USE_SUBGRAPH_BRIDGE(elementwise_min, kNNAdapter, "huawei_ascend_npu");
USE_SUBGRAPH_BRIDGE(
    fusion_elementwise_add_activation,
    kNNAdapter,
    "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_npu,amlogic_npu");
USE_SUBGRAPH_BRIDGE(
    fusion_elementwise_sub_activation,
    kNNAdapter,
    "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_npu,amlogic_npu");
USE_SUBGRAPH_BRIDGE(
    fusion_elementwise_mul_activation,
    kNNAdapter,
    "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_npu,amlogic_npu");
USE_SUBGRAPH_BRIDGE(
    fusion_elementwise_div_activation,
    kNNAdapter,
    "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_npu,amlogic_npu");
USE_SUBGRAPH_BRIDGE(fusion_elementwise_min_activation,
                    kNNAdapter,
                    "huawei_ascend_npu");
USE_SUBGRAPH_BRIDGE(fusion_elementwise_max_activation,
                    kNNAdapter,
                    "huawei_ascend_npu");
USE_SUBGRAPH_BRIDGE(
    scale,
    kNNAdapter,
    "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_npu,amlogic_npu");
USE_SUBGRAPH_BRIDGE(
    reshape,
    kNNAdapter,
    "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_npu,amlogic_npu");
USE_SUBGRAPH_BRIDGE(
    reshape2,
    kNNAdapter,
    "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_npu,amlogic_npu");
USE_SUBGRAPH_BRIDGE(
    transpose,
    kNNAdapter,
    "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_npu");
USE_SUBGRAPH_BRIDGE(
    transpose2,
    kNNAdapter,
    "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_npu");
USE_SUBGRAPH_BRIDGE(
    concat,
    kNNAdapter,
    "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_npu,amlogic_npu");
USE_SUBGRAPH_BRIDGE(
    flatten,
    kNNAdapter,
    "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_npu,amlogic_npu");
USE_SUBGRAPH_BRIDGE(
    flatten2,
    kNNAdapter,
    "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_npu,amlogic_npu");
USE_SUBGRAPH_BRIDGE(split, kNNAdapter, "huawei_kirin_npu,huawei_ascend_npu");
USE_SUBGRAPH_BRIDGE(cast, kNNAdapter, "huawei_ascend_npu");
USE_SUBGRAPH_BRIDGE(assign, kNNAdapter, "huawei_ascend_npu");
USE_SUBGRAPH_BRIDGE(assign_value, kNNAdapter, "huawei_ascend_npu");
USE_SUBGRAPH_BRIDGE(norm, kNNAdapter, "huawei_ascend_npu");
USE_SUBGRAPH_BRIDGE(deformable_conv, kNNAdapter, "huawei_ascend_npu");
USE_SUBGRAPH_BRIDGE(conv2d_transpose, kNNAdapter, "amlogic_npu");
USE_SUBGRAPH_BRIDGE(pow, kNNAdapter, "huawei_ascend_npu");
USE_SUBGRAPH_BRIDGE(batch_norm, kNNAdapter, "huawei_ascend_npu");
USE_SUBGRAPH_BRIDGE(clip, kNNAdapter, "huawei_ascend_npu");
USE_SUBGRAPH_BRIDGE(leaky_relu, kNNAdapter, "huawei_ascend_npu");
USE_SUBGRAPH_BRIDGE(slice, kNNAdapter, "huawei_ascend_npu");
USE_SUBGRAPH_BRIDGE(reduce_mean, kNNAdapter, "huawei_ascend_npu");
USE_SUBGRAPH_BRIDGE(dropout, kNNAdapter, "huawei_ascend_npu");
USE_SUBGRAPH_BRIDGE(expand_v2, kNNAdapter, "huawei_ascend_npu");
// USE_SUBGRAPH_BRIDGE(range, kNNAdapter, "huawei_ascend_npu");
USE_SUBGRAPH_BRIDGE(nearest_interp, kNNAdapter, "huawei_ascend_npu");
USE_SUBGRAPH_BRIDGE(nearest_interp_v2, kNNAdapter, "huawei_ascend_npu");
USE_SUBGRAPH_BRIDGE(bilinear_interp, kNNAdapter, "huawei_ascend_npu");
USE_SUBGRAPH_BRIDGE(bilinear_interp_v2, kNNAdapter, "huawei_ascend_npu");
USE_SUBGRAPH_BRIDGE(hard_swish, kNNAdapter, "huawei_ascend_npu");
USE_SUBGRAPH_BRIDGE(hard_sigmoid, kNNAdapter, "huawei_ascend_npu");
USE_SUBGRAPH_BRIDGE(squeeze, kNNAdapter, "huawei_ascend_npu");
USE_SUBGRAPH_BRIDGE(squeeze2, kNNAdapter, "huawei_ascend_npu");
USE_SUBGRAPH_BRIDGE(unsqueeze, kNNAdapter, "huawei_ascend_npu");
USE_SUBGRAPH_BRIDGE(unsqueeze2, kNNAdapter, "huawei_ascend_npu");
USE_SUBGRAPH_BRIDGE(elementwise_pow, kNNAdapter, "huawei_ascend_npu");
USE_SUBGRAPH_BRIDGE(p_norm, kNNAdapter, "huawei_ascend_npu");
USE_SUBGRAPH_BRIDGE(pad2d, kNNAdapter, "huawei_ascend_npu");
USE_SUBGRAPH_BRIDGE(pad3d, kNNAdapter, "huawei_ascend_npu");
USE_SUBGRAPH_BRIDGE(stack, kNNAdapter, "huawei_ascend_npu");
