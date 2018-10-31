/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef POOL_OP

#pragma once

#include "framework/tensor.h"
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif  // __ARM_NEON
namespace paddle_mobile {
namespace operators {
namespace math {
using framework::Tensor;
using std::vector;

void Pool2x2Maxs2p0(vector<int> strides, vector<int> paddings,
                    const Tensor *input, Tensor *output);

void Pool2x2Avgs2p0(vector<int> strides, vector<int> paddings,
                    const Tensor *in_x, Tensor *out);
}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile
#endif
