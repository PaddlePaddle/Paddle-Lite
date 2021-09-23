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
#include <cmath>
#include <vector>

#include "lite/backends/x86/fluid/float16.h"
#include "lite/core/context.h"
#include "lite/core/op_lite.h"
#include "lite/core/tensor.h"
#include "lite/utils/log/cp_logging.h"
// #include "lite/tensor_util.h"

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

// template <typename T, int Rank>
//    struct Transpose {
//        void operator()(const lite::Context<Target::kX86> &context)
//    };

template <lite::TargetType Target, typename T, int Rank>
struct Transpose {
  void operator()(const lite::Context<Target>& context,
                  const lite::Tensor& in,
                  lite::Tensor* out,
                  const std::vector<int>& axis);
};

template <lite::TargetType Target, typename T>
struct SetConstant {
  void operator()(const lite::Context<Target>& context,
                  lite::Tensor* tensor,
                  T num);
};

template <lite::TargetType Target>
void set_constant_with_place(const lite::Context<Target>& context,
                             lite::Tensor* tensor,
                             float value);

template <lite::TargetType Target>
void set_constant(const lite::Context<Target>& context,
                  lite::Tensor* tensor,
                  float value);

template <lite::TargetType Target, typename T>
struct RowwiseAdd {
  void operator()(const lite::Context<Target>& context,
                  const lite::Tensor& input,
                  const lite::Tensor& vec,
                  lite::Tensor* output);
};

template <lite::TargetType Target, typename T>
struct ColwiseSum {
  void operator()(const lite::Context<Target>& context,
                  const lite::Tensor& input,
                  lite::Tensor* vec);
};

template <lite::TargetType Target, typename T>
struct RowwiseSum {
  void operator()(const lite::Context<Target>& context,
                  const lite::Tensor& input,
                  lite::Tensor* vec);
};

template <lite::TargetType Target, typename T>
struct RowwiseMean {
  void operator()(const lite::Context<Target>& context,
                  const lite::Tensor& input,
                  lite::Tensor* vec);
};

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
