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

#include <map>
#include <vector>

#include "lite/backends/x86/fluid/eigen.h"
#include "lite/backends/x86/fluid/selected_rows.h"
#include "lite/backends/x86/math/blas.h"
#include "lite/backends/x86/math/math_function.h"
#include "lite/core/context.h"

#define INLINE_FOR2(sizei, sizej)     \
  for (int64_t i = 0; i < sizei; i++) \
    for (int64_t j = 0; j < sizej; j++)

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

template <lite::TargetType Target, typename T>
struct SelectedRowsAdd {
  void operator()(const lite::Context<Target>& context,
                  const fluid::SelectedRows& input1,
                  const fluid::SelectedRows& input2,
                  fluid::SelectedRows* output);
};

template <lite::TargetType Target, typename T>
struct SelectedRowsAddTensor {
  void operator()(const lite::Context<Target>& context,
                  const fluid::SelectedRows& input1,
                  const lite::Tensor& input2,
                  lite::Tensor* output);
};

// input2 = input1 + input2
template <lite::TargetType Target, typename T>
struct SelectedRowsAddTo {
  void operator()(const lite::Context<Target>& context,
                  const fluid::SelectedRows& input1,
                  const int64_t input2_offset,
                  fluid::SelectedRows* input2);
};

// input2 = [all input in input1] + input2
template <lite::TargetType Target, typename T>
struct SelectedRowsSumTo {
  void operator()(const lite::Context<Target>& context,
                  const std::vector<fluid::SelectedRows*>& input1,
                  const std::vector<int64_t>& input2_offsets,
                  fluid::SelectedRows* input2);
};

// FIXME: The result of SelectedRowsAddToTensor maybe non deterministic,
// because it uses CudaAtomicAdd.
// input2 = input1 + input2
template <lite::TargetType Target, typename T>
struct SelectedRowsAddToTensor {
  void operator()(const lite::Context<Target>& context,
                  const fluid::SelectedRows& input1,
                  lite::Tensor* input2);
};

namespace scatter {
// functors for manuplating SelectedRows data
template <lite::TargetType Target, typename T>
struct MergeAdd {
  // unary functor, merge by adding duplicated rows in
  // the input SelectedRows object.
  fluid::SelectedRows operator()(const lite::Context<Target>& context,
                                 const fluid::SelectedRows& input,
                                 const bool sorted_result = false);
  void operator()(const lite::Context<Target>& context,
                  const fluid::SelectedRows& input,
                  fluid::SelectedRows* output,
                  const bool sorted_result = false);
  void operator()(const lite::Context<Target>& context,
                  const std::vector<const fluid::SelectedRows*>& inputs,
                  fluid::SelectedRows* output,
                  const bool sorted_result = false);
};

enum class ScatterOps { ASSIGN, ADD, SUB, SUBBY, MUL, DIV, DIVBY };

// out = selected_rows_in / tensor
template <lite::TargetType Target, typename T>
struct UpdateToTensor {
  void operator()(const lite::Context<Target>& context,
                  const ScatterOps& op,
                  const fluid::SelectedRows& input1,
                  lite::Tensor* input2);
};

}  // namespace scatter
}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
