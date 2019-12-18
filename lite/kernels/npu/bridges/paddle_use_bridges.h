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

USE_SUBGRAPH_BRIDGE(NPU, sigmoid);
USE_SUBGRAPH_BRIDGE(NPU, relu);
USE_SUBGRAPH_BRIDGE(NPU, tanh);
USE_SUBGRAPH_BRIDGE(NPU, relu_clipped);
USE_SUBGRAPH_BRIDGE(NPU, leaky_relu);
USE_SUBGRAPH_BRIDGE(NPU, softsign);
USE_SUBGRAPH_BRIDGE(NPU, hard_sigmoid);

USE_SUBGRAPH_BRIDGE(NPU, batch_norm);
USE_SUBGRAPH_BRIDGE(NPU, concat);
USE_SUBGRAPH_BRIDGE(NPU, conv2d);
USE_SUBGRAPH_BRIDGE(NPU, depthwise_conv2d);
USE_SUBGRAPH_BRIDGE(NPU, conv2d_transpose);

USE_SUBGRAPH_BRIDGE(NPU, elementwise_add);
USE_SUBGRAPH_BRIDGE(NPU, fusion_elementwise_add_activation);
USE_SUBGRAPH_BRIDGE(NPU, elementwise_sub);
USE_SUBGRAPH_BRIDGE(NPU, elementwise_mul);
USE_SUBGRAPH_BRIDGE(NPU, elementwise_div);

USE_SUBGRAPH_BRIDGE(NPU, fc);
USE_SUBGRAPH_BRIDGE(NPU, bilinear_interp);
USE_SUBGRAPH_BRIDGE(NPU, nearest_interp);
USE_SUBGRAPH_BRIDGE(NPU, mul);
USE_SUBGRAPH_BRIDGE(NPU, pad2d);
USE_SUBGRAPH_BRIDGE(NPU, pool2d);
USE_SUBGRAPH_BRIDGE(NPU, reduce_mean);
USE_SUBGRAPH_BRIDGE(NPU, reshape);
USE_SUBGRAPH_BRIDGE(NPU, reshape2);
USE_SUBGRAPH_BRIDGE(NPU, scale);
USE_SUBGRAPH_BRIDGE(NPU, shuffle_channel);
USE_SUBGRAPH_BRIDGE(NPU, softmax);
USE_SUBGRAPH_BRIDGE(NPU, split);
USE_SUBGRAPH_BRIDGE(NPU, sqrt);
USE_SUBGRAPH_BRIDGE(NPU, square);
USE_SUBGRAPH_BRIDGE(NPU, transpose);
USE_SUBGRAPH_BRIDGE(NPU, transpose2);
