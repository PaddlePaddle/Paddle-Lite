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

#include "operation/math/softmax.h"
#include "operation/math/dequantize.h"
#include "operation/math/quantize.h"

namespace nnadapter {
namespace operation {
namespace math {

int softmax(const int8_t* input_data,
            const std::vector<int32_t>& input_shape,
            float input_scale,
            int32_t axis,
            int8_t* output_data,
            float output_scale) {
  auto input_count = shape_production(input_shape);
  std::vector<float> dequantized_input_data(input_count);
  std::vector<float> dequantized_output_data(input_count);
  int status = dequantize(
      input_data, input_shape, input_scale, dequantized_input_data.data());
  if (status) return status;
  status = softmax<float>(dequantized_input_data.data(),
                          input_shape,
                          axis,
                          dequantized_output_data.data());
  if (status) return status;
  return quantize(
      dequantized_output_data.data(), input_shape, output_scale, output_data);
}

}  // namespace math
}  // namespace operation
}  // namespace nnadapter
