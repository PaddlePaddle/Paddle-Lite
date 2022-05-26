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
/**
 * fill_constant_batch_size_like -> fill_like + transpose + slice +
 * elementwise_add
 *
 * Such as:
 * input: dims=(2, 5, 4, 3)
 * value: 2
 * input_dims_idx: 1
 * output_dims_idx: 2
 * shape: (7, 9, ?, 2)
 *
 *             input        zero_value(0)
 *                \            /
 *                  \        /
 *                  [fill_like]
 *                      |
 *                      |  dims=(2,5,4,3)
 *                      |
 *                   [slice]
 *                      |
 *                      |  dims=(1,5,1,1)
 *                      |
 *                  [reshape]
 *                      |
 *                      |  dims=(1,1,5,1)
 *                      |
 *                  new_input                   dummy_input
 *            (dims=(1,1,5,1),value=0)  (dims=(7,9,1,2),value=2)
 *                      |               /
 *                      |             /
 *                      |           /
 *                    [elementwise_add]
 *                            |
 *                            |
 *                          output
 *                 (dims=(7,9,5,2), value=2)
 */
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
  std::vector<int> shape = op->GetAttr<std::vector<int>>("shape");
  auto shape_size = shape.size();
  float value = op->HasAttr("value") ? op->GetAttr<float>("value") : 0.0f;
  int input_dim_idx =
      op->HasAttr("input_dim_idx") ? op->GetAttr<int>("input_dim_idx") : 0;
  int output_dim_idx =
      op->HasAttr("output_dim_idx") ? op->GetAttr<int>("output_dim_idx") : 0;
  auto out_name = op->Output("Out").front();
  auto out_scale_name = "Out0_scale";
  std::vector<float> out_scales;
  if (op->HasOutputScale(out_scale_name, true)) {
    out_scales = op->GetOutputScale(out_scale_name, true);
  }
  // Convert to NNAdapter operands and operation
  // Input Operand
  auto input_operand =
      converter->AddInputOperand(scope, input_name, {}, input_scales);
  uint32_t input_rank =
      converter->GetOperandType(input_operand)->dimensions.count;
  std::vector<int64_t> out_shape;
  int32_t shape_count = 1;
  shape[output_dim_idx] = 1;
  for (size_t i = 0; i < shape_size; i++) {
    out_shape.push_back(shape[i]);
    shape_count *= shape[i];
  }
  // Dummy input operand for Elementwise add operation
  NNAdapterOperand* dummy_input_operand = nullptr;
  // Value operand
  float zero_value = 0.0f;
  NNAdapterOperand* zero_value_operand = nullptr;
  switch (dtype) {
    case static_cast<int32_t>(lite::core::FluidType::FP32):
      zero_value_operand = converter->AddConstantOperand(zero_value);
      dummy_input_operand = converter->AddConstantOperand(
          std::vector<float>(shape_count, value), DDim(out_shape), out_scales);
      break;
    case static_cast<int32_t>(lite::core::FluidType::INT8):
      zero_value_operand = converter->AddConstantOperand(
          static_cast<int8_t>(zero_value), out_scales);
      dummy_input_operand = converter->AddConstantOperand(
          std::vector<int8_t>(shape_count, static_cast<int8_t>(value)),
          DDim(out_shape),
          out_scales);
      break;
    case static_cast<int32_t>(lite::core::FluidType::INT32):
      zero_value_operand =
          converter->AddConstantOperand(static_cast<int32_t>(zero_value));
      dummy_input_operand = converter->AddConstantOperand(
          std::vector<int32_t>(shape_count, static_cast<int32_t>(value)),
          DDim(out_shape),
          out_scales);
      break;
    case static_cast<int32_t>(lite::core::FluidType::INT64):
      zero_value_operand =
          converter->AddConstantOperand(static_cast<int64_t>(zero_value));
      dummy_input_operand = converter->AddConstantOperand(
          std::vector<int64_t>(shape_count, static_cast<int64_t>(value)),
          DDim(out_shape),
          out_scales);
      break;
    case static_cast<int32_t>(lite::core::FluidType::BOOL):
      zero_value_operand =
          converter->AddConstantOperand(static_cast<bool>(zero_value));
      dummy_input_operand = converter->AddConstantOperand(
          reinterpret_cast<bool*>(
              std::vector<int8_t>(shape_count, static_cast<int8_t>(value))
                  .data()),
          DDim(out_shape),
          true,
          out_scales);
      break;
    default:
      LOG(FATAL) << "Not supported dtype: " << dtype;
      break;
  }
  // Fill like out operand
  auto fill_like_out_operand =
      converter->AddOutputOperand(out_name, out_scales);
  // Fill like operation
  converter->AddOperation(NNADAPTER_FILL_LIKE,
                          {input_operand, zero_value_operand},
                          {fill_like_out_operand});
  // Slice operation
  std::vector<int> slice_axes = {};
  for (int i = 0; i < input_rank; i++) {
    if (i == input_dim_idx) continue;
    slice_axes.push_back(i);
  }
  auto starts = std::vector<int>(slice_axes.size(), 0);
  auto ends = std::vector<int>(slice_axes.size(), 1);
  auto steps = std::vector<int>(slice_axes.size(), 1);
  auto slice_out_operand = converter->AddSliceOperation(fill_like_out_operand,
                                                        slice_axes,
                                                        starts,
                                                        ends,
                                                        steps,
                                                        out_name,
                                                        out_scales);
  // Reshape operation
  std::vector<int> input_shape = std::vector<int>(shape_size, 1);
  input_shape[output_dim_idx] = -1;
  auto input_shape_operand = converter->AddConstantOperand(input_shape);
  auto reshape_output_operand =
      converter->AddOutputOperand(out_name, out_scales);
  converter->AddOperation(NNADAPTER_RESHAPE,
                          {slice_out_operand, input_shape_operand},
                          {reshape_output_operand});
  auto fuse_code_operand =
      converter->AddConstantOperand(static_cast<int32_t>(NNADAPTER_FUSED_NONE));
  auto dummy_output_operand = converter->AddOutputOperand(out_name, out_scales);
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
