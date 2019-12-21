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

USE_SUBGRAPH_BRIDGE(XPU, relu);
USE_SUBGRAPH_BRIDGE(XPU, tanh);
USE_SUBGRAPH_BRIDGE(XPU, conv2d);
USE_SUBGRAPH_BRIDGE(XPU, depthwise_conv2d);
USE_SUBGRAPH_BRIDGE(XPU, elementwise_add);
USE_SUBGRAPH_BRIDGE(XPU, pool2d);
USE_SUBGRAPH_BRIDGE(XPU, softmax);
USE_SUBGRAPH_BRIDGE(XPU, mul);
USE_SUBGRAPH_BRIDGE(XPU, batch_norm);
USE_SUBGRAPH_BRIDGE(XPU, stack);
USE_SUBGRAPH_BRIDGE(XPU, gather);
USE_SUBGRAPH_BRIDGE(XPU, scale);
USE_SUBGRAPH_BRIDGE(XPU, lookup_table);
USE_SUBGRAPH_BRIDGE(XPU, slice);
USE_SUBGRAPH_BRIDGE(XPU, transpose);
USE_SUBGRAPH_BRIDGE(XPU, transpose2);
USE_SUBGRAPH_BRIDGE(XPU, reshape);
USE_SUBGRAPH_BRIDGE(XPU, reshape2);
USE_SUBGRAPH_BRIDGE(XPU, layer_norm);
