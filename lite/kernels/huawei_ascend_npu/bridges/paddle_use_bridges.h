// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

// activation
USE_SUBGRAPH_BRIDGE(sigmoid, kHuaweiAscendNPU);
USE_SUBGRAPH_BRIDGE(relu, kHuaweiAscendNPU);
USE_SUBGRAPH_BRIDGE(tanh, kHuaweiAscendNPU);
USE_SUBGRAPH_BRIDGE(relu6, kHuaweiAscendNPU);
USE_SUBGRAPH_BRIDGE(leaky_relu, kHuaweiAscendNPU);
USE_SUBGRAPH_BRIDGE(softsign, kHuaweiAscendNPU);
USE_SUBGRAPH_BRIDGE(softplus, kHuaweiAscendNPU);
USE_SUBGRAPH_BRIDGE(conv2d, kHuaweiAscendNPU);
USE_SUBGRAPH_BRIDGE(depthwise_conv2d, kHuaweiAscendNPU);
USE_SUBGRAPH_BRIDGE(bilinear_interp, kHuaweiAscendNPU);
USE_SUBGRAPH_BRIDGE(nearest_interp, kHuaweiAscendNPU);
USE_SUBGRAPH_BRIDGE(concat, kHuaweiAscendNPU);
USE_SUBGRAPH_BRIDGE(pool2d, kHuaweiAscendNPU);
USE_SUBGRAPH_BRIDGE(elementwise_add, kHuaweiAscendNPU);
USE_SUBGRAPH_BRIDGE(elementwise_sub, kHuaweiAscendNPU);
USE_SUBGRAPH_BRIDGE(elementwise_mul, kHuaweiAscendNPU);
USE_SUBGRAPH_BRIDGE(elementwise_div, kHuaweiAscendNPU);
USE_SUBGRAPH_BRIDGE(elementwise_max, kHuaweiAscendNPU);
USE_SUBGRAPH_BRIDGE(fusion_elementwise_add_activation, kHuaweiAscendNPU);
USE_SUBGRAPH_BRIDGE(fusion_elementwise_sub_activation, kHuaweiAscendNPU);
USE_SUBGRAPH_BRIDGE(fusion_elementwise_mul_activation, kHuaweiAscendNPU);
USE_SUBGRAPH_BRIDGE(fusion_elementwise_div_activation, kHuaweiAscendNPU);
USE_SUBGRAPH_BRIDGE(fusion_elementwise_max_activation, kHuaweiAscendNPU);
USE_SUBGRAPH_BRIDGE(batch_norm, kHuaweiAscendNPU);
USE_SUBGRAPH_BRIDGE(softmax, kHuaweiAscendNPU);
USE_SUBGRAPH_BRIDGE(dropout, kHuaweiAscendNPU);
USE_SUBGRAPH_BRIDGE(fc, kHuaweiAscendNPU);
USE_SUBGRAPH_BRIDGE(reshape, kHuaweiAscendNPU);
USE_SUBGRAPH_BRIDGE(reshape2, kHuaweiAscendNPU);
USE_SUBGRAPH_BRIDGE(transpose, kHuaweiAscendNPU);
USE_SUBGRAPH_BRIDGE(transpose2, kHuaweiAscendNPU);
USE_SUBGRAPH_BRIDGE(flatten, kHuaweiAscendNPU);
USE_SUBGRAPH_BRIDGE(flatten2, kHuaweiAscendNPU);
USE_SUBGRAPH_BRIDGE(layer_norm, kHuaweiAscendNPU);
USE_SUBGRAPH_BRIDGE(matmul, kHuaweiAscendNPU);
USE_SUBGRAPH_BRIDGE(cast, kHuaweiAscendNPU);
USE_SUBGRAPH_BRIDGE(scale, kHuaweiAscendNPU);
USE_SUBGRAPH_BRIDGE(slice, kHuaweiAscendNPU);
USE_SUBGRAPH_BRIDGE(gather, kHuaweiAscendNPU);
