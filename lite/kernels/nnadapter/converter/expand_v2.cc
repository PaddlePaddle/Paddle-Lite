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

int ConvertExpandV2(Converter* converter, OpInfo* op, Scope* scope) {
  // Input operand
  auto x_name = op->Input("X").front();
  auto x_scale_name = "X0_scale";
  std::vector<float> x_scales;
  if (op->HasInputScale(x_scale_name, true)) {
    x_scales = op->GetInputScale(x_scale_name, true);
  }
  auto input_operand = converter->AddInputOperand(scope, x_name, {}, x_scales);
  CHECK(input_operand);
  auto input_type = converter->GetOperandType(input_operand);

  // Shape operand
  NNAdapterOperand* shape_operand = nullptr;
  if (op->HasInput("Shape") && !op->Input("Shape").empty()) {
    auto shape_name = op->Input("Shape").front();
    shape_operand = converter->AddInputOperand(scope, shape_name);
  } else if (op->HasInput("expand_shapes_tensor") &&
             !op->Input("expand_shapes_tensor").empty()) {
    LOG(ERROR) << "Not support expand_shapes_tensor now.";
  } else {
    auto shape = op->GetAttr<std::vector<int>>("shape");
    shape_operand = converter->AddConstantOperand(shape);

    // std::vector<int64_t> input_dims_vec;
    // for (uint32_t i = 0; i < input_type->dimensions.count; i++) {
    //   input_dims_vec.push_back(input_type->dimensions.data[i]);
    // }
    // auto diff = expand_shape.size() - input_dims_vec.size();
    // input_dims_vec.insert(input_dims_vec.begin(), diff, 1);
    // std::vector<int> final_expand_shape(input_dims_vec.size());
    // for (size_t i = 0; i < input_dims_vec.size(); ++i) {
    //   CHECK_NE(expand_shape[i], 0) << "The expanded size cannot be zero.";
    //   if (i < diff) {  // expand_shape = [3,4,-1,-1], X = [10,2] -->
    //                    // final_expand_shape = [3,4,10,2]
    //     CHECK_GT(expand_shape[i], 0) << "The expanded size " <<
    //     expand_shape[i]
    //                                  << "for non-existing dimensions must be
    //                                  positive for expand_v2 op.";
    //     final_expand_shape[i] = expand_shape[i];
    //   } else if (expand_shape[i] > 0) {  // expand_shape = [3,4,10,4], X =
    //                                      // [10,1] --> final_expand_shape =
    //                                      // [3,4,10,4]
    //     if (input_dims_vec[i] != 1) {
    //       CHECK_EQ(input_dims_vec[i], expand_shape[i])
    //               << "The value " << input_dims_vec[i]
    //               << " of the non-singleton dimension does not match the
    //               corresponding value "
    //               << expand_shape[i]
    //               << " in shape for expand_v2 op.";
    //       final_expand_shape[i] = expand_shape[i];
    //     } else {
    //       final_expand_shape[i] = expand_shape[i];
    //     }
    //   } else {  // expand_shape = [3,4,-1,-1], X = [10,2] -->
    //   final_expand_shape
    //             // = [3,4,10,2]
    //     CHECK_EQ(expand_shape[i], -1)
    //               << "When the value in shape is negative for expand_v2 op, "
    //               "only -1 is supported, but the value received is " <<
    //               expand_shape[i];
    //     final_expand_shape[i] = input_dims_vec[i];
    //   }
    // }
  }
  // Output operand
  auto out_name = op->Output("Out").front();
  auto output_operand = converter->AddOutputOperand(out_name);
  // Expand operation
  std::vector<NNAdapterOperand*> input_operands = {input_operand,
                                                   shape_operand};
  std::vector<NNAdapterOperand*> output_operands = {output_operand};
  converter->AddOperation(NNADAPTER_EXPAND, &input_operands, &output_operands);

  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
