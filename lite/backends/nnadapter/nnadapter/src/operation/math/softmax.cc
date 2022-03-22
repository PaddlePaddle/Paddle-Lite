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

void softmax(const int8_t* input_data_ptr,
             const std::vector<int32_t>& input_shape,
             float* input_scale_ptr,
             size_t input_scale_count,
             int32_t axis,
             int8_t* output_data_ptr,
             float* output_scale_ptr,
             size_t output_scale_count) {
  auto input_data_count = production_of_shape(input_shape);
  std::vector<float> dequantized_input_data(input_data_count);
  std::vector<float> dequantized_output_data(input_data_count);
  dequantize(input_data_ptr,
             input_shape,
             input_scale_ptr,
             input_scale_count,
             dequantized_input_data.data());
  softmax<float>(dequantized_input_data.data(),
                 input_shape,
                 axis,
                 dequantized_output_data.data());
  quantize(dequantized_output_data.data(),
           input_shape,
           output_scale_ptr,
           output_scale_count,
           output_data_ptr);
}

}  // namespace math
}  // namespace operation
}  // namespace nnadapter
