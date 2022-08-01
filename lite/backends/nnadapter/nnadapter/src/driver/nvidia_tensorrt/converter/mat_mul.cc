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
#include "utility/modeling.h"

namespace nnadapter {
namespace nvidia_tensorrt {

int ConvertMatMul(Converter* converter, core::Operation* operation) {
  MAT_MUL_OPERATION_EXTRACT_INPUTS_OUTPUTS
  NNADAPTER_CHECK(!IsConstantOperand(x_operand));

  // Convert to trt tensors and node
  auto x_tensor = converter->GetMappedTensor(x_operand);
  if (x_tensor == nullptr) {
    x_tensor = converter->ConvertOperand(x_operand);
  }
  auto y_tensor = converter->GetMappedTensor(y_operand);
  if (y_tensor == nullptr) {
    uint32_t x_dims_count = x_operand->type.dimensions.count;
    uint32_t y_dims_count = y_operand->type.dimensions.count;
    auto y_dims_data = y_operand->type.dimensions.data;
    std::vector<int32_t> y_dims(y_dims_data, y_dims_data + y_dims_count);
    if (IsConstantOperand(y_operand)) {
      if (y_dims_count + 1 < x_dims_count) {
        y_dims.insert(y_dims.begin(), x_dims_count - y_dims_count - 1, 1);
      }
    } else {
      y_dims.erase(y_dims.begin());
    }
    y_tensor = converter->ConvertOperand(y_operand, y_dims);
  }
  nvinfer1::MatrixOperation matrix_operation_x =
      transpose_x ? nvinfer1::MatrixOperation::kTRANSPOSE
                  : nvinfer1::MatrixOperation::kNONE;
  int x_dims_count = x_operand->type.dimensions.count;
  if (!IsConstantOperand(x_operand) && x_dims_count == 2) {
    NNADAPTER_CHECK(!transpose_x);
    matrix_operation_x = nvinfer1::MatrixOperation::kVECTOR;
  }
  nvinfer1::MatrixOperation matrix_operation_y =
      transpose_y ? nvinfer1::MatrixOperation::kTRANSPOSE
                  : nvinfer1::MatrixOperation::kNONE;
  int y_dims_count = y_operand->type.dimensions.count;
  if (!IsConstantOperand(y_operand) && y_dims_count == 2) {
    NNADAPTER_CHECK(!transpose_y);
    matrix_operation_y = nvinfer1::MatrixOperation::kVECTOR;
  }
  auto matmul_layer = converter->network()->addMatrixMultiply(
      *x_tensor, matrix_operation_x, *y_tensor, matrix_operation_y);
  NNADAPTER_CHECK(matmul_layer);
  auto output_tensor = matmul_layer->getOutput(0);
  converter->UpdateTensorMap(output_operand, output_tensor);
  return NNADAPTER_NO_ERROR;
}

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
