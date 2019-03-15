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

#ifdef CONV_OP

#pragma once

#include "framework/context.h"
#include "framework/tensor.h"

namespace paddle_mobile {
namespace operators {
namespace math {

template <int tile, int kernel>
void winograd_transform_weight(const framework::Tensor &weight,
                               framework::Tensor *output);

template <int tile, int kernel>
void winograd_transform_input(const framework::Tensor &input,
                              framework::Tensor *output);

template <int tile, int kernel>
void winograd_transform_output(const framework::Tensor &input,
                               const framework::Tensor &weight,
                               framework::Tensor *output);

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile

#endif
