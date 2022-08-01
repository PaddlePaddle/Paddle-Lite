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

int ConvertMul(Converter* converter, OpInfo* op, Scope* scope) {
  // X operand
  auto x_name = op->Input("X").front();
  auto x_scale_name = "X0_scale";
  std::vector<float> x_scales;
  if (op->HasInputScale(x_scale_name, true)) {
    x_scales = op->GetInputScale(x_scale_name, true);
    CHECK(IsValidSymmPerLayerQuantParams(x_scales));
  }
  auto x_operand = converter->GetMappedOperand(x_name);
  int x_num_col_dims = op->GetAttr<int>("x_num_col_dims");
  if (!x_operand) {
    auto x_tensor = scope->FindTensor(x_name);
    CHECK(x_tensor->persistable());
    auto x_dims = x_tensor->dims();
    std::vector<int64_t> reshaped_x_shape(x_num_col_dims + 1);
    for (int i = 0; i < x_num_col_dims; i++) {
      reshaped_x_shape[i] = x_dims[i];
    }
    reshaped_x_shape[x_num_col_dims] =
        x_dims.Slice(x_num_col_dims, x_dims.size()).production();
    x_operand = converter->AddConstantOperand(
        *x_tensor, DDim(reshaped_x_shape), false, x_scales);
  } else {
    auto x_rank = converter->GetOperandType(x_operand)->dimensions.count;
    if (x_rank != x_num_col_dims + 1) {
      std::vector<int32_t> shape;
      for (int i = 0; i < x_num_col_dims; i++) {
        shape.push_back(0);
      }
      shape.push_back(-1);
      auto shape_operand = converter->AddConstantOperand(shape);
      auto reshaped_x_operand =
          converter->AddOutputOperand(x_name + "_reshaped", x_scales);
      converter->AddOperation(
          NNADAPTER_RESHAPE, {x_operand, shape_operand}, {reshaped_x_operand});
      x_operand = reshaped_x_operand;
    }
  }

  // Y operand
  auto y_name = op->Input("Y").front();
  auto y_scale_name = "Y0_scale";
  std::vector<float> y_scales;
  if (op->HasInputScale(y_scale_name, true)) {
    y_scales = op->GetInputScale(y_scale_name, true);
    if (!IsValidSymmPerChannelQuantParams(y_scales)) {
      y_scales = {y_scales[0]};
    }
  }
  auto y_operand = converter->GetMappedOperand(y_name);
  int y_num_col_dims = op->GetAttr<int>("y_num_col_dims");
  if (!y_operand) {
    auto y_tensor = scope->FindTensor(y_name);
    CHECK(y_tensor->persistable());
    auto y_dims = y_tensor->dims();
    std::vector<int64_t> reshaped_y_shape(y_num_col_dims + 1);
    for (int i = 0; i < y_num_col_dims; i++) {
      reshaped_y_shape[i] = y_dims[i];
    }
    reshaped_y_shape[y_num_col_dims] =
        y_dims.Slice(y_num_col_dims, y_dims.size()).production();
    y_operand = converter->AddConstantOperand(
        *y_tensor, DDim(reshaped_y_shape), false, y_scales);
  } else {
    auto y_rank = converter->GetOperandType(y_operand)->dimensions.count;
    if (y_rank != y_num_col_dims + 1) {
      std::vector<int32_t> shape;
      for (int i = 0; i < y_num_col_dims; i++) {
        shape.push_back(0);
      }
      shape.push_back(-1);
      auto shape_operand = converter->AddConstantOperand(shape);
      auto reshaped_y_operand =
          converter->AddOutputOperand(y_name + "_reshaped", y_scales);
      converter->AddOperation(
          NNADAPTER_RESHAPE, {y_operand, shape_operand}, {reshaped_y_operand});
      y_operand = reshaped_y_operand;
    }
  }

  // The attribute `transpose_x` and `transpose_y` is not supported by mul op.
  auto transpose_x_operand = converter->AddConstantOperand(false);
  auto transpose_y_operand = converter->AddConstantOperand(false);

  // Output operand
  auto out_name = op->Output("Out").front();
  auto output_operand = converter->AddOutputOperand(out_name);

  // Mat_mul operation
  converter->AddOperation(
      NNADAPTER_MAT_MUL,
      {x_operand, y_operand, transpose_x_operand, transpose_y_operand},
      {output_operand});
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
