// // Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
// //
// // Licensed under the Apache License, Version 2.0 (the "License");
// // you may not use this file except in compliance with the License.
// // You may obtain a copy of the License at
// //
// //     http://www.apache.org/licenses/LICENSE-2.0
// //
// // Unless required by applicable law or agreed to in writing, software
// // distributed under the License is distributed on an "AS IS" BASIS,
// // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// // See the License for the specific language governing permissions and
// // limitations under the License.

// #include "operation/fill.h"
// #include "operation/assign.h"
// #include "driver/nvidia_tensorrt/converter/converter.h"
// #include "utility/debug.h"
// #include "utility/logging.h"
// #include <iostream>
// namespace nnadapter {
// namespace nvidia_tensorrt {

// int ConvertFill(Converter* converter, core::Operation* operation) {
//   FILL_OPERATION_EXTRACT_INPUTS_OUTPUTS

//   // Convert to trt tensors and node
//   //auto fill_layer = converter->network()->addFill(nvinfer1::Dims3{5,4,4},
//   nvinfer1::FillOperation::kRANDOM_UNIFORM);
//   nvinfer1::Weights ws;
//   float* s = new float[100];
//   ws.values = s;
//   ws.count = 1;
//   ws.type = nvinfer1::DataType::kFLOAT;
//   auto constant_layer =
//   converter->network()->addConstant(nvinfer1::Dims3{1,1,1}, ws);

//   auto output_tensor = constant_layer->getOutput(0);
//   converter->UpdateTensorMap(output_operand, output_tensor);
//   return NNADAPTER_NO_ERROR;
// }

// }  // namespace nvidia_tensorrt
// }  // namespace nnadapter

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

#include "driver/nvidia_tensorrt/converter/plugin/fill.h"
#include <iostream>
#include "driver/nvidia_tensorrt/converter/converter.h"
#include "operation/fill.h"
#include "utility/debug.h"
#include "utility/logging.h"
namespace nnadapter {
namespace nvidia_tensorrt {

int ConvertFill(Converter* converter, core::Operation* operation) {
  FILL_OPERATION_EXTRACT_INPUTS_OUTPUTS
  // Convert to trt tensors and node
  auto value_tensor = converter->GetMappedTensor(value_operand);
  if (!value_tensor) {
    value_tensor = converter->ConvertOperand(value_operand);
  }

  int shape_rank =
      shape_operand->length / static_cast<int64_t>(sizeof(int64_t));
  auto shape_data = static_cast<int64_t*>(shape_operand->buffer);
  std::vector<int64_t> shape_dims(shape_data, shape_data + shape_rank);

  float value;
  bool bool_value_tensor;
  std::vector<nvinfer1::ITensor*> tensors;
  if (value_operand->buffer) {
    value = *(static_cast<float*>(value_operand->buffer));
    bool_value_tensor = false;
  } else {
    bool_value_tensor = true;
    tensors.push_back(value_tensor);
  }

  FillPluginDynamic fill_plugin(value, bool_value_tensor, shape_dims);
  auto fill_layer =
      converter->network()->addPluginV2(tensors.data(), 1, fill_plugin);
  NNADAPTER_CHECK(fill_layer);
  auto output_tensor = fill_layer->getOutput(0);
  converter->UpdateTensorMap(output_operand, output_tensor);
  return NNADAPTER_NO_ERROR;
}

}  // namespace nvidia_tensorrt
}  // namespace nnadapter