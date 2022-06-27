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
#include "lite/operators/prior_box_op.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace nnadapter {

int ConvertPriorBox(Converter* converter, OpInfo* op, Scope* scope) {
  // Extract the inputs, outputs and attributes
  auto input_name = op->Input("Input").front();
  auto image_name = op->Input("Image").front();
  auto boxes_name = op->Output("Boxes").front();
  auto Variances_name = op->Output("Variances").front();
  auto min_sizes = op->GetAttr<std::vector<float>>("min_sizes");
  auto max_sizes = op->GetAttr<std::vector<float>>("max_sizes");
  auto aspect_ratios = op->GetAttr<std::vector<float>>("aspect_ratios");
  auto variances = op->GetAttr<std::vector<float>>("variances");
  if (max_sizes.size() > 0) {
    CHECK_EQ(max_sizes.size(), min_sizes.size());
    for (int32_t i = 0; i < max_sizes.size(); i++) {
      CHECK_GT(max_sizes[i], min_sizes[i]);
    }
  }
  auto flip = op->GetAttr<bool>("flip");
  auto clip = op->GetAttr<bool>("clip");
  auto step_w = op->GetAttr<float>("step_w");
  auto step_h = op->GetAttr<float>("step_h");
  auto offset = op->GetAttr<float>("offset");
  auto min_max_aspect_ratios_order =
      op->HasAttr("min_max_aspect_ratios_order")
          ? op->GetAttr<bool>("min_max_aspect_ratios_order")
          : false;
  auto input_operand = converter->AddInputOperand(scope, input_name);
  auto image_operand = converter->AddInputOperand(scope, image_name);
  auto boxes_operand = converter->AddOutputOperand(boxes_name);
  auto Variances_operand = converter->AddOutputOperand(Variances_name);
  auto min_sizes_operand = converter->AddConstantOperand(min_sizes);
  NNAdapterOperand* max_sizes_operand = nullptr;
  if (max_sizes.size() > 0) {
    max_sizes_operand = converter->AddConstantOperand(max_sizes);
  }
  auto aspect_ratios_operand = converter->AddConstantOperand(aspect_ratios);
  auto variances_operand = converter->AddConstantOperand(variances);
  auto flip_operand = converter->AddConstantOperand(flip);
  auto clip_operand = converter->AddConstantOperand(clip);
  auto step_w_operand = converter->AddConstantOperand(step_w);
  auto step_h_operand = converter->AddConstantOperand(step_h);
  auto offset_operand = converter->AddConstantOperand(offset);
  auto min_max_aspect_ratios_order_operand =
      converter->AddConstantOperand(min_max_aspect_ratios_order);
  converter->AddOperation(NNADAPTER_PRIOR_BOX,
                          {input_operand,
                           image_operand,
                           min_sizes_operand,
                           max_sizes_operand,
                           aspect_ratios_operand,
                           variances_operand,
                           flip_operand,
                           clip_operand,
                           step_w_operand,
                           step_h_operand,
                           offset_operand,
                           min_max_aspect_ratios_order_operand},
                          {boxes_operand, Variances_operand});
  return NO_ERROR;
}

}  // namespace nnadapter
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
