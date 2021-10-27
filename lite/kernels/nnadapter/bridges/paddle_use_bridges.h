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

USE_SUBGRAPH_BRIDGE(fc,
                    kNNAdapter,
                    "rockchip_npu,mediatek_apu,huawei_kirin_npu,huawei_ascend_"
                    "npu,amlogic_npu,imagination_nna");
USE_SUBGRAPH_BRIDGE(cast, kNNAdapter, "huawei_ascend_npu");
USE_SUBGRAPH_BRIDGE(norm, kNNAdapter, "huawei_ascend_npu");
USE_SUBGRAPH_BRIDGE(dropout, kNNAdapter, "huawei_ascend_npu");
USE_SUBGRAPH_BRIDGE(p_norm, kNNAdapter, "huawei_ascend_npu");
USE_SUBGRAPH_BRIDGE(pad2d, kNNAdapter, "huawei_ascend_npu");
USE_SUBGRAPH_BRIDGE(pad3d, kNNAdapter, "huawei_ascend_npu");
