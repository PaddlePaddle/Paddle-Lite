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

USE_SUBGRAPH_BRIDGE(relu, kAPU);
USE_SUBGRAPH_BRIDGE(conv2d, kAPU);
USE_SUBGRAPH_BRIDGE(depthwise_conv2d, kAPU);
USE_SUBGRAPH_BRIDGE(elementwise_add, kAPU);
USE_SUBGRAPH_BRIDGE(elementwise_mul, kAPU);
USE_SUBGRAPH_BRIDGE(fc, kAPU);
USE_SUBGRAPH_BRIDGE(pool2d, kAPU);
USE_SUBGRAPH_BRIDGE(softmax, kAPU);
USE_SUBGRAPH_BRIDGE(concat, kAPU);
USE_SUBGRAPH_BRIDGE(fusion_elementwise_add_activation, kAPU);
USE_SUBGRAPH_BRIDGE(conv2d_transpose, kAPU);
