// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/core/optimizer/mir/weight_quantization_preprocess_pass.h"
#include <memory>
#include <string>
#include <vector>
#include "lite/core/optimizer/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

bool IsAbsMaxQuantizedOp(const OpInfo& op_info) {
  bool result = false;
  if (op_info.HasAttr("quantization_type") &&
      op_info.GetAttr<std::string>("quantization_type") ==
          "post_weight_abs_max") {
    result = true;
  } else if (!op_info.HasAttr("quantization_type") &&
             op_info.HasAttr("quantize_weight_bits")) {  // Support older model,
                                                         // save this for now
    result = true;
  }
  return result;
}

/*
 * For abs_max method in WeightQuantization, this pass obtains the scale value
 * of conv2d, depthwise_conv2d and mul, expands the scale list, and save the
 * list in the quantized ops.
*/
void WeightQuantizationPreprocessPass::Apply(
    const std::unique_ptr<SSAGraph>& graph) {
  std::vector<std::string> weight_quantized_op = {
      "conv2d", "depthwise_conv2d", "mul"};
  for (auto& node : graph->StmtTopologicalOrder()) {
    if (node->IsStmt() &&
        std::find(weight_quantized_op.begin(),
                  weight_quantized_op.end(),
                  node->AsStmt().op_type()) != weight_quantized_op.end()) {
      auto* scope = node->stmt()->op()->scope();
      auto* op_desc = node->stmt()->mutable_op_info();
      if (IsAbsMaxQuantizedOp(*op_desc)) {
        for (auto& input_name : op_desc->input_vars()) {
          std::string scale_name = input_name + "_quant_scale";
          if (op_desc->HasAttr(scale_name)) {
            VLOG(0) << " WeightQuantizationPreprocessPass op:"
                    << op_desc->Type() << " input_name:" << input_name;
            auto input_tensor =
                scope->FindVar(input_name)->GetMutable<lite::Tensor>();
            int weight_out_channel;
            if (op_desc->Type() == "mul") {
              weight_out_channel = static_cast<int>(input_tensor->dims()[1]);
            } else {
              weight_out_channel = static_cast<int>(input_tensor->dims()[0]);
            }
            auto input_scale = op_desc->GetAttr<std::vector<float>>(scale_name);
            // scale length is equal to weight out channel
            std::vector<float> scale_list(weight_out_channel, input_scale[0]);
            op_desc->SetAttr(scale_name, scale_list);
          }
        }
      }
    }
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(weight_quantization_preprocess_pass,
                  paddle::lite::mir::WeightQuantizationPreprocessPass)
    .BindTargets({TARGET(kAny)});
