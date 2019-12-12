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

#include "lite/kernels/bm/bridges/registry.h"

USE_BM_BRIDGE(relu);
USE_BM_BRIDGE(conv2d);
USE_BM_BRIDGE(elementwise_add);
USE_BM_BRIDGE(pool2d);
USE_BM_BRIDGE(softmax);
USE_BM_BRIDGE(mul);
USE_BM_BRIDGE(batch_norm);
USE_BM_BRIDGE(scale);
