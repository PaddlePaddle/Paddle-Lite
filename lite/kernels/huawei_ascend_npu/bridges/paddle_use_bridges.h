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
// conv
USE_SUBGRAPH_BRIDGE(conv2d, kHuaweiAscendNPU);
USE_SUBGRAPH_BRIDGE(depthwise_conv2d, kHuaweiAscendNPU);
USE_SUBGRAPH_BRIDGE(bilinear_interp, kHuaweiAscendNPU);
USE_SUBGRAPH_BRIDGE(nearest_interp, kHuaweiAscendNPU);
USE_SUBGRAPH_BRIDGE(concat, kHuaweiAscendNPU);
