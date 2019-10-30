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

#include "lite/kernels/xpu/bridges/registry.h"

USE_XPU_BRIDGE(relu);
USE_XPU_BRIDGE(conv2d);
USE_XPU_BRIDGE(depthwise_conv2d);
USE_XPU_BRIDGE(elementwise_add);
USE_XPU_BRIDGE(pool2d);
USE_XPU_BRIDGE(softmax);
USE_XPU_BRIDGE(mul);
