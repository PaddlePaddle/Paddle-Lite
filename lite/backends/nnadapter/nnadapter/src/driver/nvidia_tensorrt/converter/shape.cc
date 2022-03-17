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

#include "operation/shape.h"
#include "driver/nvidia_tensorrt/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include <iostream>

namespace nnadapter {
namespace nvidia_tensorrt {

int ConvertShape(Converter* converter, core::Operation* operation) {
  SHAPE_OPERATION_EXTRACT_INPUTS_OUTPUTS
  // Convert to trt tensors and node
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }

  auto shape_layer = converter->network()->addShape(*input_tensor);
  auto output_tensor = shape_layer->getOutput(0);
  converter->UpdateTensorMap(output_operand, output_tensor);
  /*std :: cout << " shape_layer " << (int32_t)shape_layer->getType() << std :: endl; 
  std :: cout << " shape_layer->hasImplicitBatchDimension " << (int)converter->network()->hasImplicitBatchDimension() << std :: endl;    
  std :: cout << " shape_layer->getNbInputs " << shape_layer->getNbInputs() << std :: endl;    
  std :: cout << " shape_layer->getNbOutputs " << shape_layer->getNbOutputs() << std :: endl;     
  std :: cout << " output_operand->length " << output_operand->length << std :: endl;
  std :: cout << " output_operand->length 111111111111111" << output_operand->length << std :: endl;  */
  return NNADAPTER_NO_ERROR;
}

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
