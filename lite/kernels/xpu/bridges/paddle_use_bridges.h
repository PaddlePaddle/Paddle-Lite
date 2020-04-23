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

USE_SUBGRAPH_BRIDGE(relu, kXPU);
USE_SUBGRAPH_BRIDGE(tanh, kXPU);
USE_SUBGRAPH_BRIDGE(conv2d, kXPU);
USE_SUBGRAPH_BRIDGE(depthwise_conv2d, kXPU);
USE_SUBGRAPH_BRIDGE(elementwise_add, kXPU);
USE_SUBGRAPH_BRIDGE(pool2d, kXPU);
USE_SUBGRAPH_BRIDGE(softmax, kXPU);
USE_SUBGRAPH_BRIDGE(mul, kXPU);
USE_SUBGRAPH_BRIDGE(batch_norm, kXPU);
USE_SUBGRAPH_BRIDGE(stack, kXPU);
USE_SUBGRAPH_BRIDGE(gather, kXPU);
USE_SUBGRAPH_BRIDGE(scale, kXPU);
USE_SUBGRAPH_BRIDGE(lookup_table, kXPU);
USE_SUBGRAPH_BRIDGE(slice, kXPU);
USE_SUBGRAPH_BRIDGE(transpose, kXPU);
USE_SUBGRAPH_BRIDGE(transpose2, kXPU);
USE_SUBGRAPH_BRIDGE(reshape, kXPU);
USE_SUBGRAPH_BRIDGE(reshape2, kXPU);
USE_SUBGRAPH_BRIDGE(layer_norm, kXPU);
USE_SUBGRAPH_BRIDGE(gelu, kXPU);
USE_SUBGRAPH_BRIDGE(dropout, kXPU);
USE_SUBGRAPH_BRIDGE(matmul, kXPU);
USE_SUBGRAPH_BRIDGE(cast, kXPU);
USE_SUBGRAPH_BRIDGE(yolo_box, kXPU);
