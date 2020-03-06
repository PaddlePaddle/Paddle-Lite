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

//USE_SUBGRAPH_BRIDGE(sigmoid, kAPU);
USE_SUBGRAPH_BRIDGE(relu, kAPU);
//USE_SUBGRAPH_BRIDGE(tanh, kAPU);
//USE_SUBGRAPH_BRIDGE(relu_clipped, kAPU);
//USE_SUBGRAPH_BRIDGE(leaky_relu, kAPU);
//USE_SUBGRAPH_BRIDGE(softsign, kAPU);
//USE_SUBGRAPH_BRIDGE(hard_sigmoid, kAPU);

//USE_SUBGRAPH_BRIDGE(batch_norm, kAPU);
//USE_SUBGRAPH_BRIDGE(concat, kAPU);
USE_SUBGRAPH_BRIDGE(conv2d, kAPU);
USE_SUBGRAPH_BRIDGE(depthwise_conv2d, kAPU);
//USE_SUBGRAPH_BRIDGE(conv2d_transpose, kAPU);

//USE_SUBGRAPH_BRIDGE(dropout, kAPU);
USE_SUBGRAPH_BRIDGE(elementwise_add, kAPU);
//USE_SUBGRAPH_BRIDGE(elementwise_sub, kAPU);
USE_SUBGRAPH_BRIDGE(elementwise_mul, kAPU);
//USE_SUBGRAPH_BRIDGE(elementwise_div, kAPU);
//USE_SUBGRAPH_BRIDGE(fusion_elementwise_add_activation, kAPU);
//USE_SUBGRAPH_BRIDGE(fusion_elementwise_sub_activation, kAPU);
//USE_SUBGRAPH_BRIDGE(fusion_elementwise_mul_activation, kAPU);
//USE_SUBGRAPH_BRIDGE(fusion_elementwise_div_activation, kAPU);

USE_SUBGRAPH_BRIDGE(fc, kAPU);
//USE_SUBGRAPH_BRIDGE(bilinear_interp, kAPU);
//USE_SUBGRAPH_BRIDGE(nearest_interp, kAPU);
//USE_SUBGRAPH_BRIDGE(matmul, kAPU);
//USE_SUBGRAPH_BRIDGE(mul, kAPU);
//USE_SUBGRAPH_BRIDGE(pad2d, kAPU);
USE_SUBGRAPH_BRIDGE(pool2d, kAPU);
//USE_SUBGRAPH_BRIDGE(reduce_mean, kAPU);
//USE_SUBGRAPH_BRIDGE(reshape, kAPU);
//USE_SUBGRAPH_BRIDGE(reshape2, kAPU);
//USE_SUBGRAPH_BRIDGE(scale, kAPU);
//USE_SUBGRAPH_BRIDGE(shuffle_channel, kAPU);
USE_SUBGRAPH_BRIDGE(softmax, kAPU);
//USE_SUBGRAPH_BRIDGE(split, kAPU);
//USE_SUBGRAPH_BRIDGE(sqrt, kAPU);
//USE_SUBGRAPH_BRIDGE(square, kAPU);
//USE_SUBGRAPH_BRIDGE(transpose, kAPU);
//USE_SUBGRAPH_BRIDGE(transpose2, kAPU);
//USE_SUBGRAPH_BRIDGE(unsqueeze, kAPU);
//USE_SUBGRAPH_BRIDGE(unsqueeze2, kAPU);
//USE_SUBGRAPH_BRIDGE(instance_norm, kAPU);
//USE_SUBGRAPH_BRIDGE(layer_norm, kAPU);
