// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "operation/math/elementwise.h"
#include "operation/math/dequantize.h"
#include "operation/math/quantize.h"

namespace nnadapter {
namespace operation {
namespace math {

int elementwise(ElementwiseTypeCode eltwise_type,
                const int8_t* input0_data,
                const std::vector<int32_t>& input0_shape,
                float input0_scale,
                const int8_t* input1_data,
                const std::vector<int32_t>& input1_shape,
                float input1_scale,
                FuseCode fuse_code,
                int8_t* output_data,
                float output_scale) {
  auto input0_count = shape_production(input0_shape);
  std::vector<float> dequantized_input0_data(input0_count);
  auto input1_count = shape_production(input1_shape);
  std::vector<float> dequantized_input1_data(input1_count);
  auto output_shape = shape_broadcast(input0_shape, input1_shape);
  auto output_count = shape_production(output_shape);
  std::vector<float> dequantized_output_data(output_count);
  int status = dequantize(
      input0_data, input0_shape, input0_scale, dequantized_input0_data.data());
  if (status) return status;
  status = dequantize(
      input1_data, input1_shape, input1_scale, dequantized_input1_data.data());
  if (status) return status;
  status = elementwise<float>(eltwise_type,
                              dequantized_input0_data.data(),
                              input0_shape,
                              dequantized_input1_data.data(),
                              input1_shape,
                              fuse_code,
                              dequantized_output_data.data());
  if (status) return status;
  return quantize(
      dequantized_output_data.data(), output_shape, output_scale, output_data);
}

}  // namespace math
}  // namespace operation
}  // namespace nnadapter
