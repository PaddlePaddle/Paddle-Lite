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

#include "lite/backends/npu/bridge/registry.h"

USE_NPU_BRIDGE(mul);
USE_NPU_BRIDGE(fc);
USE_NPU_BRIDGE(conv2d);
USE_NPU_BRIDGE(depthwise_conv2d);
USE_NPU_BRIDGE(pool2d);
USE_NPU_BRIDGE(relu);
USE_NPU_BRIDGE(elementwise_add);
USE_NPU_BRIDGE(scale);
USE_NPU_BRIDGE(softmax);
USE_NPU_BRIDGE(concat);
USE_NPU_BRIDGE(split);
USE_NPU_BRIDGE(transpose);
USE_NPU_BRIDGE(transpose2);
USE_NPU_BRIDGE(shuffle_channel);
USE_NPU_BRIDGE(batch_norm);
USE_NPU_BRIDGE(bilinear_interp);
USE_NPU_BRIDGE(conv2d_transpose);
USE_NPU_BRIDGE(reshape);
USE_NPU_BRIDGE(reshape2);
