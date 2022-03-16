// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "operation/flatten.h"
#include "driver/nvidia_tensorrt/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace nvidia_tensorrt {

int ConvertFlatten(Converter* converter, core::Operation* operation) {
  FLATTEN_OPERATION_EXTRACT_INPUTS_OUTPUTS

  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  if (start_axis < 0) {
    start_axis += input_operand->type.dimensions.count;
  }
  if (end_axis < 0) {
    end_axis += input_operand->type.dimensions.count;
  }
  uint32_t dim_prod = 1;
  nvinfer1::Dims flatten_dim;
  flatten_dim.nbDims = input_operand->type.dimensions.count - (end_axis - start_axis);
  for (int i = 0, j = 0; i < input_operand->type.dimensions.count; ++i) {     
    if (start_axis <= i && i <= end_axis) {  
      int dim_i = input_operand->type.dimensions.data[i];
      dim_prod *= dim_i;     
      if (i == end_axis) {     
        flatten_dim.d[j++] = dim_prod;
      }      
    } else {     
      flatten_dim.d[j++] = input_operand->type.dimensions.data[i];   
    }  
  }
  auto flatten_layer = converter->network()->addShuffle(*input_tensor);
  flatten_layer->setReshapeDimensions(flatten_dim);

  auto output_tensor = flatten_layer->getOutput(0);
  converter->UpdateTensorMap(output_operand, output_tensor);

  return NNADAPTER_NO_ERROR;
}

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
