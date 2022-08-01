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

int ConvertPool(Converter* converter, OpInfo* op, Scope* scope) {
  // Input operand
  auto x_name = op->Input("X").front();
  auto x_scale_name = "X0_scale";
  std::vector<float> x_scales = op->HasInputScale(x_scale_name, true)
                                    ? op->GetInputScale(x_scale_name, true)
                                    : std::vector<float>{};
  auto input_operand = converter->AddInputOperand(scope, x_name, {}, x_scales);

  // Auto_pad operand
  std::string padding_algorithm =
      op->HasAttr("padding_algorithm")
          ? op->GetAttr<std::string>("padding_algorithm")
          : "";
  auto auto_pad_operand = converter->AddConstantOperand(static_cast<int32_t>(
      ConvertPaddingAlgorithmToNNAutoPadCode(padding_algorithm)));

  // Pads operand(optional)
  std::vector<int> paddings = op->GetAttr<std::vector<int>>("paddings");
  if (paddings.size() == 2L) {
    paddings.insert(paddings.begin(), paddings[0]);
    paddings.insert(paddings.begin() + 2, paddings[2]);
  }
  CHECK_EQ(paddings.size(), 4L);
  bool global_pooling = op->GetAttr<bool>("global_pooling");
  if (global_pooling) {
    paddings = std::vector<int>(4, 0);
  }
  auto pads_operand = converter->AddConstantOperand(paddings);

  // Kernel_shape operand
  bool adaptive =
      op->HasAttr("adaptive") ? op->GetAttr<bool>("adaptive") : false;
  std::vector<int> ksize = op->GetAttr<std::vector<int>>("ksize");
  bool trans_adaptive_to_global = adaptive && ksize[0] == 1 && ksize[1] == 1;
  if (global_pooling || trans_adaptive_to_global) {
    auto in_dims_data =
        converter->GetOperandType(input_operand)->dimensions.data;
    ksize[0] = in_dims_data[2];
    ksize[1] = in_dims_data[3];
    if (trans_adaptive_to_global) {
      adaptive = false;
    }
  }
  auto kernel_shape_operand = converter->AddConstantOperand(ksize);

  // Strides operand
  std::vector<int> strides = op->GetAttr<std::vector<int>>("strides");
  auto strides_operand = converter->AddConstantOperand(strides);

  // Ceil_mode(optional)
  bool ceil_mode =
      op->HasAttr("ceil_mode") ? op->GetAttr<bool>("ceil_mode") : false;
  auto ceil_mode_operand = converter->AddConstantOperand(ceil_mode);

  // Count_include_pad(optional), only for avg_pool
  bool exclusive =
      op->HasAttr("exclusive") ? op->GetAttr<bool>("exclusive") : true;
  auto count_include_pad_operand = converter->AddConstantOperand(!exclusive);

  // Return_indices(optional), only for max_pool
  auto return_indices_operand = converter->AddConstantOperand(false);

  // Return_indices_type(optional), only for max_pool
  auto return_indices_type_operand =
      converter->AddConstantOperand(static_cast<int32_t>(NNADAPTER_INT32));

  // Fuse code operand
  auto fuse_code_operand =
      converter->AddConstantOperand<int32_t>(NNADAPTER_FUSED_NONE);

  // Output operand
  auto out_name = op->Output("Out").front();
  auto out_scale_name = "Out0_scale";
  std::vector<float> out_scales = op->HasOutputScale(out_scale_name, true)
                                      ? op->GetOutputScale(out_scale_name, true)
                                      : std::vector<float>{};
  auto output_operand = converter->AddOutputOperand(out_name, out_scales);

  // Pool operation
  std::vector<NNAdapterOperand*> input_operands;
  if (adaptive) {
    input_operands.push_back(input_operand);
    input_operands.push_back(kernel_shape_operand);
  } else {
    input_operands.push_back(input_operand);
    input_operands.push_back(auto_pad_operand);
    input_operands.push_back(pads_operand);
    input_operands.push_back(kernel_shape_operand);
    input_operands.push_back(strides_operand);
    input_operands.push_back(ceil_mode_operand);
    input_operands.push_back(fuse_code_operand);
  }

  std::vector<NNAdapterOperand*> output_operands = {output_operand};
  NNAdapterOperationType pool2d_operation_type;
  std::string pooling_type = op->GetAttr<std::string>("pooling_type");
  if (adaptive && pooling_type == "avg") {
    pool2d_operation_type = NNADAPTER_ADAPTIVE_AVERAGE_POOL_2D;
  } else if (adaptive && pooling_type == "max") {
    pool2d_operation_type = NNADAPTER_ADAPTIVE_MAX_POOL_2D;
    input_operands.push_back(return_indices_operand);
    input_operands.push_back(return_indices_type_operand);
    output_operands.push_back(nullptr);
  } else if (pooling_type == "avg") {
    pool2d_operation_type = NNADAPTER_AVERAGE_POOL_2D;
    input_operands.insert(input_operands.begin() + 6,
                          count_include_pad_operand);
  } else if (pooling_type == "max") {
    pool2d_operation_type = NNADAPTER_MAX_POOL_2D;
    input_operands.insert(input_operands.begin() + 6, return_indices_operand);
    input_operands.insert(input_operands.begin() + 7,
                          return_indices_type_operand);
    output_operands.push_back(nullptr);
  } else {
    LOG(FATAL) << "Unsupported pooling type: " << pooling_type;
    return UNSUPPORTED_FEATURE;
  }

  converter->AddOperation(
      pool2d_operation_type, &input_operands, &output_operands);
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
