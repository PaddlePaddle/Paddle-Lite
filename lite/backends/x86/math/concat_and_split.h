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

#pragma once
#include <vector>
#include "lite/core/context.h"
#include "lite/core/tensor.h"
#include "lite/fluid/data_type.h"

namespace paddle {
namespace lite_metal {
namespace x86 {
namespace math {

/*
 * \brief Concatenate the input tensors along the dimension axis.
 *  TODO(zcd): maybe it needs to be more detailed.
 *  Examples:
 *     Input[0] = [[1,2],[3,4]]
 *     Input[1] = [[5,6]]
 *     axis = 0
 *
 *     Output = [[1,2],
 *               [3,4],
 *               [5,6]]
 */
template <lite_metal::TargetType Target, typename T>
class ConcatFunctor {
 public:
  void operator()(const lite_metal::Context<Target>& context,
                  const std::vector<lite_metal::Tensor>& input,
                  int axis,
                  lite_metal::Tensor* output);
};

/*
 * \brief Split the input tensors along the dimension axis into outputs.
 *  TODO(zcd): maybe it needs to be more detailed.
 *  Examples:
 *     Input = [[1,2],
 *              [3,4],
 *              [5,6]]
 *     axis = 0
 *
 *     Output[0] = [[1,2],[3,4]]
 *     Output[1] = [[5,6]]
 */
template <lite_metal::TargetType Target, typename T>
class SplitFunctor {
 public:
  void operator()(const lite_metal::Context<Target>& context,
                  const lite_metal::Tensor& input,
                  const std::vector<const lite_metal::Tensor*>& ref_inputs,
                  int axis,
                  std::vector<lite_metal::Tensor*>* outputs);
};

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle

#define FOR_ALL_TYPES(macro) \
  macro(int);                \
  macro(float);              \
  macro(double);             \
  macro(bool);               \
  macro(int64_t);            \
  macro(int16_t);            \
  macro(uint8_t);            \
  macro(int8_t);             \
  macro(::paddle::lite_metal::fluid::float16)
