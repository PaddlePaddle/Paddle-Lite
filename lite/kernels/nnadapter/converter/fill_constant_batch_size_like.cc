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

#include "lite/kernels/nnadapter/converter/converter.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace nnadapter {

int ConvertFillConstantBatchSizeLike(Converter* converter,
                                     OpInfo* op,
                                     Scope* scope) {
  // Extract op attributes
  auto input_name = op->Input("Input").front();
  auto input_scale_name = "Input0_scale";
  std::vector<float> input_scales;
  if (op->HasInputScale(input_scale_name, true)) {
    input_scales = op->GetInputScale(input_scale_name, true);
  }
  auto dtype = op->GetAttr<int>("dtype");
  auto shape = op->GetAttr<std::vector<int>>("shape");
  auto shape_size = shape.size();
  float value = op->HasAttr("value") ? op->GetAttr<float>("value") : 0.0f;
  int input_dim_idx =
      op->HasAttr("input_dim_idx") ? op->GetAttr<int>("input_dim_idx") : 0;
  int output_dim_idx =
      op->HasAttr("output_dim_idx") ? op->GetAttr<int>("output_dim_idx") : 0;
  auto out_name = op->Output("Out").front();
  // Convert to NNAdapter operands and operation
  // Input Operand
  auto input_operand =
      converter->AddInputOperand(scope, input_name, {}, input_scales);
  uint32_t input_rank =
      converter->GetOperandType(input_operand)->dimensions.count;
  // Value operand
  float zero_value = 0.0f;
  NNAdapterOperand* zero_value_operand = nullptr;
  switch (dtype) {
    case static_cast<int32_t>(lite::core::FluidType::FP32):
      zero_value_operand = converter->AddConstantOperand(zero_value);
      break;
    case static_cast<int32_t>(lite::core::FluidType::INT32):
      zero_value_operand =
          converter->AddConstantOperand(static_cast<int32_t>(zero_value));
      break;
    case static_cast<int32_t>(lite::core::FluidType::INT64):
      zero_value_operand =
          converter->AddConstantOperand(static_cast<int64_t>(zero_value));
      break;
    case static_cast<int32_t>(lite::core::FluidType::BOOL):
      zero_value_operand =
          converter->AddConstantOperand(static_cast<bool>(zero_value));
      break;
    default:
      LOG(FATAL) << "Not supported dtype: " << dtype;
      break;
  }
  // Fill like out operand
  auto fill_like_out_operand = converter->AddOutputOperand(out_name);
  // Fill like operation
  converter->AddOperation(NNADAPTER_FILL_LIKE,
                          {input_operand, zero_value_operand},
                          {fill_like_out_operand});
  // Reduce mean operation
  std::vector<int> axes = {};
  for (int i = 0; i < input_rank; i++) {
    if (i == input_dim_idx) continue;
    axes.push_back(i);
  }
  auto axes_operand = converter->AddConstantOperand(axes);
  auto keep_dim_operand = converter->AddConstantOperand(1);
  auto reduce_mean_output_operand = converter->AddOutputOperand(out_name);
  converter->AddOperation(
      NNADAPTER_REDUCE_MEAN,
      {fill_like_out_operand, axes_operand, keep_dim_operand},
      {reduce_mean_output_operand});
  // Reshape operation
  std::vector<int> input_shape = std::vector<int>(shape_size, 1);
  input_shape[output_dim_idx] = -1;
  auto input_shape_operand = converter->AddConstantOperand(input_shape);
  auto reshape_output_operand = converter->AddOutputOperand(out_name);
  converter->AddOperation(NNADAPTER_RESHAPE,
                          {reduce_mean_output_operand, input_shape_operand},
                          {reshape_output_operand});
  // Elementwise add
  std::vector<int64_t> out_shape;
  int32_t shape_count = 1;
  shape[output_dim_idx] = 1;
  for (size_t i = 0; i < shape_size; i++) {
    out_shape.push_back(shape[i]);
    shape_count *= shape[i];
  }
  auto fuse_code_operand =
      converter->AddConstantOperand(static_cast<int32_t>(NNADAPTER_FUSED_NONE));
  auto dummy_input_operand = converter->AddConstantOperand(
      std::vector<float>(shape_count, value), DDim(out_shape));
  auto dummy_output_operand = converter->AddOutputOperand(out_name);
  converter->AddOperation(
      NNADAPTER_ADD,
      {reshape_output_operand, dummy_input_operand, fuse_code_operand},
      {dummy_output_operand});
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
