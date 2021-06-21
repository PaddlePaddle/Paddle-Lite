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
                    "rockchip_npu,mediatek_apu,huawei_kirin_npu");
USE_SUBGRAPH_BRIDGE(depthwise_conv2d,
                    kNNAdapter,
                    "rockchip_npu,mediatek_apu,huawei_kirin_npu");
USE_SUBGRAPH_BRIDGE(fc,
                    kNNAdapter,
                    "rockchip_npu,mediatek_apu,huawei_kirin_npu");
USE_SUBGRAPH_BRIDGE(softmax,
                    kNNAdapter,
                    "rockchip_npu,mediatek_apu,huawei_kirin_npu");
USE_SUBGRAPH_BRIDGE(pool2d,
                    kNNAdapter,
                    "rockchip_npu,mediatek_apu,huawei_kirin_npu");
USE_SUBGRAPH_BRIDGE(sigmoid,
                    kNNAdapter,
                    "rockchip_npu,mediatek_apu,huawei_kirin_npu");
USE_SUBGRAPH_BRIDGE(relu,
                    kNNAdapter,
                    "rockchip_npu,mediatek_apu,huawei_kirin_npu");
USE_SUBGRAPH_BRIDGE(relu6,
                    kNNAdapter,
                    "rockchip_npu,mediatek_apu,huawei_kirin_npu");
USE_SUBGRAPH_BRIDGE(tanh,
                    kNNAdapter,
                    "rockchip_npu,mediatek_apu,huawei_kirin_npu");
USE_SUBGRAPH_BRIDGE(elementwise_add,
                    kNNAdapter,
                    "rockchip_npu,mediatek_apu,huawei_kirin_npu");
USE_SUBGRAPH_BRIDGE(elementwise_sub,
                    kNNAdapter,
                    "rockchip_npu,mediatek_apu,huawei_kirin_npu");
USE_SUBGRAPH_BRIDGE(elementwise_mul,
                    kNNAdapter,
                    "rockchip_npu,mediatek_apu,huawei_kirin_npu");
USE_SUBGRAPH_BRIDGE(elementwise_div,
                    kNNAdapter,
                    "rockchip_npu,mediatek_apu,huawei_kirin_npu");
USE_SUBGRAPH_BRIDGE(fusion_elementwise_add_activation,
                    kNNAdapter,
                    "rockchip_npu,mediatek_apu,huawei_kirin_npu");
USE_SUBGRAPH_BRIDGE(fusion_elementwise_sub_activation,
                    kNNAdapter,
                    "rockchip_npu,mediatek_apu,huawei_kirin_npu");
USE_SUBGRAPH_BRIDGE(fusion_elementwise_mul_activation,
                    kNNAdapter,
                    "rockchip_npu,mediatek_apu,huawei_kirin_npu");
USE_SUBGRAPH_BRIDGE(fusion_elementwise_div_activation,
                    kNNAdapter,
                    "rockchip_npu,mediatek_apu,huawei_kirin_npu");
USE_SUBGRAPH_BRIDGE(scale,
                    kNNAdapter,
                    "rockchip_npu,mediatek_apu,huawei_kirin_npu");
USE_SUBGRAPH_BRIDGE(reshape, kNNAdapter, "rockchip_npu,mediatek_apu");
USE_SUBGRAPH_BRIDGE(reshape2, kNNAdapter, "rockchip_npu,mediatek_apu");
USE_SUBGRAPH_BRIDGE(transpose, kNNAdapter, "rockchip_npu,mediatek_apu");
USE_SUBGRAPH_BRIDGE(transpose2, kNNAdapter, "rockchip_npu,mediatek_apu");
USE_SUBGRAPH_BRIDGE(concat, kNNAdapter, "rockchip_npu,mediatek_apu");
