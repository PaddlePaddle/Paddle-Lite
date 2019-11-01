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
#include <string>
#include "lite/core/context.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

template <lite::TargetType Target, typename T>
class SequencePoolFunctor {
 public:
  /* max pool has index output */
  void operator()(const lite::Context<Target>& context,
                  const std::string pooltype,
                  T pad_value,
                  const lite::Tensor& input,
                  lite::Tensor* output,
                  bool is_test = false,
                  lite::Tensor* index = nullptr);
};

template <lite::TargetType Target, typename T>
class SequencePoolGradFunctor {
 public:
  void operator()(const lite::Context<Target>& context,
                  const std::string pooltype,
                  const lite::Tensor& out_grad,
                  lite::Tensor* in_grad,
                  /* max pool has index */
                  const lite::Tensor* index = nullptr);
};

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
