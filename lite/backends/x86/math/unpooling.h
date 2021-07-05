/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include "lite/core/context.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite_metal {
namespace x86 {
namespace math {
template <lite_metal::TargetType Target, typename T>
class Unpool2dMaxFunctor {
 public:
  void operator()(const lite_metal::Context<Target>& context,
                  const lite_metal::Tensor& input,
                  const lite_metal::Tensor& indices,
                  lite_metal::Tensor* output);
};
template <lite_metal::TargetType Target, class T>
class Unpool2dMaxGradFunctor {
 public:
  void operator()(const lite_metal::Context<Target>& context,
                  const lite_metal::Tensor& input,
                  const lite_metal::Tensor& indices,
                  const lite_metal::Tensor& output,
                  const lite_metal::Tensor& output_grad,
                  lite_metal::Tensor* input_grad);
};
}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
