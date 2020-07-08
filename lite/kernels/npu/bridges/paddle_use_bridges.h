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

USE_SUBGRAPH_BRIDGE(sigmoid, kNPU);
USE_SUBGRAPH_BRIDGE(relu, kNPU);
USE_SUBGRAPH_BRIDGE(tanh, kNPU);
USE_SUBGRAPH_BRIDGE(relu_clipped, kNPU);
USE_SUBGRAPH_BRIDGE(relu6, kNPU);
USE_SUBGRAPH_BRIDGE(leaky_relu, kNPU);
USE_SUBGRAPH_BRIDGE(softsign, kNPU);
USE_SUBGRAPH_BRIDGE(hard_sigmoid, kNPU);
USE_SUBGRAPH_BRIDGE(log, kNPU);
USE_SUBGRAPH_BRIDGE(sqrt, kNPU);
USE_SUBGRAPH_BRIDGE(square, kNPU);
USE_SUBGRAPH_BRIDGE(thresholded_relu, kNPU);

USE_SUBGRAPH_BRIDGE(batch_norm, kNPU);
USE_SUBGRAPH_BRIDGE(less_than, kNPU);
USE_SUBGRAPH_BRIDGE(concat, kNPU);
USE_SUBGRAPH_BRIDGE(conv2d, kNPU);
USE_SUBGRAPH_BRIDGE(depthwise_conv2d, kNPU);
USE_SUBGRAPH_BRIDGE(conv2d_transpose, kNPU);

USE_SUBGRAPH_BRIDGE(dropout, kNPU);
USE_SUBGRAPH_BRIDGE(elementwise_add, kNPU);
USE_SUBGRAPH_BRIDGE(elementwise_sub, kNPU);
USE_SUBGRAPH_BRIDGE(elementwise_mul, kNPU);
USE_SUBGRAPH_BRIDGE(elementwise_div, kNPU);
USE_SUBGRAPH_BRIDGE(expand, kNPU);
USE_SUBGRAPH_BRIDGE(fusion_elementwise_add_activation, kNPU);
USE_SUBGRAPH_BRIDGE(fusion_elementwise_sub_activation, kNPU);
USE_SUBGRAPH_BRIDGE(fusion_elementwise_mul_activation, kNPU);
USE_SUBGRAPH_BRIDGE(fusion_elementwise_div_activation, kNPU);
// USE_SUBGRAPH_BRIDGE(fill_constant, kNPU)
// USE_SUBGRAPH_BRIDGE(fill_constant_batch_size_like, kNPU)

// USE_SUBGRAPH_BRIDGE(gather, kNPU);
// USE_SUBGRAPH_BRIDGE(lookup_table, kNPU);
USE_SUBGRAPH_BRIDGE(increment, kNPU);
USE_SUBGRAPH_BRIDGE(instance_norm, kNPU);
USE_SUBGRAPH_BRIDGE(fc, kNPU);
USE_SUBGRAPH_BRIDGE(bilinear_interp, kNPU);
USE_SUBGRAPH_BRIDGE(nearest_interp, kNPU);
USE_SUBGRAPH_BRIDGE(layer_norm, kNPU);
USE_SUBGRAPH_BRIDGE(matmul, kNPU);
USE_SUBGRAPH_BRIDGE(mul, kNPU);
USE_SUBGRAPH_BRIDGE(pad2d, kNPU);
USE_SUBGRAPH_BRIDGE(pool2d, kNPU);
USE_SUBGRAPH_BRIDGE(reduce_mean, kNPU);
USE_SUBGRAPH_BRIDGE(reshape, kNPU);
USE_SUBGRAPH_BRIDGE(reshape2, kNPU);
USE_SUBGRAPH_BRIDGE(scale, kNPU);
// USE_SUBGRAPH_BRIDGE(shape, kNPU);
USE_SUBGRAPH_BRIDGE(shuffle_channel, kNPU);
USE_SUBGRAPH_BRIDGE(softmax, kNPU);
USE_SUBGRAPH_BRIDGE(split, kNPU);
// USE_SUBGRAPH_BRIDGE(top_k, kNPU);
USE_SUBGRAPH_BRIDGE(transpose, kNPU);
USE_SUBGRAPH_BRIDGE(transpose2, kNPU);
USE_SUBGRAPH_BRIDGE(unsqueeze, kNPU);
USE_SUBGRAPH_BRIDGE(unsqueeze2, kNPU);
