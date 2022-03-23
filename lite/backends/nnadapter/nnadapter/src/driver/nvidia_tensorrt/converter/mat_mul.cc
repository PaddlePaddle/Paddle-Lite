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

#include "operation/mat_mul.h"
#include "driver/nvidia_tensorrt/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace nvidia_tensorrt {

int ConvertMatMul(Converter* converter, core::Operation* operation) {
  MAT_MUL_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to trt tensors and node
  auto x_tensor = converter->GetMappedTensor(x_operand);
  if (x_tensor == nullptr) {
    x_tensor = converter->ConvertOperand(x_operand);
  }
  auto y_tensor = converter->GetMappedTensor(y_operand);
  if (y_tensor == nullptr) {
    y_tensor = converter->ConvertOperand(y_operand);
  }
  nvinfer1::MatrixOperation matrix_operation_x =
      transpose_x ? nvinfer1::MatrixOperation::kTRANSPOSE
                  : nvinfer1::MatrixOperation::kNONE;
  nvinfer1::MatrixOperation matrix_operation_y =
      transpose_y ? nvinfer1::MatrixOperation::kTRANSPOSE
                  : nvinfer1::MatrixOperation::kNONE;
  auto matmul_layer = converter->network()->addMatrixMultiply(
      *x_tensor, matrix_operation_x, *y_tensor, matrix_operation_y);
  NNADAPTER_CHECK(matmul_layer);
  auto output_tensor = matmul_layer->getOutput(0);
  converter->UpdateTensorMap(output_operand, output_tensor);
  return NNADAPTER_NO_ERROR;
}

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
