// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "operation/slice.h"
#include <algorithm>
#include "driver/nvidia_tensorrt/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"

namespace nnadapter {
namespace nvidia_tensorrt {

int ConvertSlice(Converter* converter, core::Operation* operation) {
  SLICE_OPERATION_EXTRACT_INPUTS_OUTPUTS
  NNADAPTER_CHECK(!IsOperandWithDynamicShape(output_operand));

  // Convert to trt tensors and node
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  NNADAPTER_CHECK(!IsOperandWithDynamicShape(input_operand));
  auto dims_data = input_operand->type.dimensions.data;
  auto dims_count = input_operand->type.dimensions.count;
  nvinfer1::Dims new_starts_dims;
  nvinfer1::Dims new_steps_dims;
  nvinfer1::Dims out_dims;
  std::vector<int32_t> new_axes(axes, axes + axes_count);
  new_starts_dims.nbDims = dims_count - 1;
  new_steps_dims.nbDims = dims_count - 1;
  out_dims.nbDims = dims_count - 1;
  int j = 0;
  for (int i = 0; i < dims_count - 1; i++) {
    if (std::find(new_axes.begin(), new_axes.end(), i + 1) == new_axes.end()) {
      new_starts_dims.d[i] = 0;
      new_steps_dims.d[i] = 1;
    } else {
      new_starts_dims.d[i] =
          starts[j] < 0 ? starts[j] + dims_data[i + 1] : starts[j];
      new_steps_dims.d[i] = steps[j];
      j++;
    }
  }
  memcpy(out_dims.d,
         output_operand->type.dimensions.data + 1,
         sizeof(int32_t) * out_dims.nbDims);

  auto slice_layer = converter->network()->addSlice(
      *input_tensor, new_starts_dims, out_dims, new_steps_dims);
  auto output_tensor = slice_layer->getOutput(0);
  converter->UpdateTensorMap(output_operand, output_tensor);
  return NNADAPTER_NO_ERROR;
}

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
