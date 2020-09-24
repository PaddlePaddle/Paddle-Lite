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

USE_SUBGRAPH_BRIDGE(relu, kBM);
USE_SUBGRAPH_BRIDGE(leaky_relu, kBM);
USE_SUBGRAPH_BRIDGE(conv2d, kBM);
USE_SUBGRAPH_BRIDGE(depthwise_conv2d, kBM);
USE_SUBGRAPH_BRIDGE(elementwise_add, kBM);
USE_SUBGRAPH_BRIDGE(elementwise_mul, kBM);
USE_SUBGRAPH_BRIDGE(elementwise_sub, kBM);
USE_SUBGRAPH_BRIDGE(elementwise_div, kBM);
USE_SUBGRAPH_BRIDGE(pool2d, kBM);
USE_SUBGRAPH_BRIDGE(softmax, kBM);
USE_SUBGRAPH_BRIDGE(mul, kBM);
USE_SUBGRAPH_BRIDGE(batch_norm, kBM);
USE_SUBGRAPH_BRIDGE(scale, kBM);
USE_SUBGRAPH_BRIDGE(concat, kBM);
USE_SUBGRAPH_BRIDGE(dropout, kBM);
USE_SUBGRAPH_BRIDGE(transpose, kBM);
USE_SUBGRAPH_BRIDGE(transpose2, kBM);
USE_SUBGRAPH_BRIDGE(reshape, kBM);
USE_SUBGRAPH_BRIDGE(reshape2, kBM);
USE_SUBGRAPH_BRIDGE(flatten, kBM);
USE_SUBGRAPH_BRIDGE(flatten2, kBM);
USE_SUBGRAPH_BRIDGE(norm, kBM);
USE_SUBGRAPH_BRIDGE(prior_box, kBM);
USE_SUBGRAPH_BRIDGE(box_coder, kBM);
USE_SUBGRAPH_BRIDGE(multiclass_nms, kBM);
USE_SUBGRAPH_BRIDGE(multiclass_nms2, kBM);
USE_SUBGRAPH_BRIDGE(nearest_interp, kBM);
USE_SUBGRAPH_BRIDGE(bilinear_interp, kBM);
USE_SUBGRAPH_BRIDGE(yolo_box, kBM);
USE_SUBGRAPH_BRIDGE(sqrt, kBM);
USE_SUBGRAPH_BRIDGE(square, kBM);
USE_SUBGRAPH_BRIDGE(slice, kBM);
USE_SUBGRAPH_BRIDGE(conv2d_transpose, kBM);
USE_SUBGRAPH_BRIDGE(reduce_sum, kBM);
USE_SUBGRAPH_BRIDGE(reduce_mean, kBM);
USE_SUBGRAPH_BRIDGE(reduce_max, kBM);
USE_SUBGRAPH_BRIDGE(squeeze, kBM);
USE_SUBGRAPH_BRIDGE(squeeze2, kBM);
USE_SUBGRAPH_BRIDGE(cast, kBM);
USE_SUBGRAPH_BRIDGE(fill_constant, kBM);
USE_SUBGRAPH_BRIDGE(assign_value, kBM);
USE_SUBGRAPH_BRIDGE(depthwise_conv2d_transpose, kBM);
USE_SUBGRAPH_BRIDGE(shape, kBM);
USE_SUBGRAPH_BRIDGE(split, kBM);
USE_SUBGRAPH_BRIDGE(matmul, kBM);
USE_SUBGRAPH_BRIDGE(max_pool2d_with_index, kBM);
USE_SUBGRAPH_BRIDGE(sigmoid, kBM);
USE_SUBGRAPH_BRIDGE(density_prior_box, kBM);
USE_SUBGRAPH_BRIDGE(swish, kBM);
