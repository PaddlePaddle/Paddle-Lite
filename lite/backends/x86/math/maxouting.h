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
#include "lite/utils/macros.h"

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

template <lite::TargetType Target, typename T>
class MaxOutFunctor {
 public:
  void operator()(const lite::Context<Target>& context,
                  const lite::Tensor& input,
                  lite::Tensor* output,
                  int groups);
};

template <lite::TargetType Target, class T>
class MaxOutGradFunctor {
 public:
  void operator()(const lite::Context<Target>& context,
                  const lite::Tensor& input,
                  lite::Tensor* input_grad,
                  const lite::Tensor& output,
                  const lite::Tensor& output_grad,
                  int groups);
};
}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
