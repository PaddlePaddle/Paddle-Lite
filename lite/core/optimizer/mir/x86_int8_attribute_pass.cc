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

#include "lite/core/optimizer/mir/x86_int8_attribute_pass.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include "lite/api/paddle_place.h"
#include "lite/core/optimizer/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {
void X86Int8AttributePass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  std::vector<mir::Node*> nodes;
  for (auto* node : graph->StmtTopologicalOrder()) {
    if (node->IsStmt()) {
      const std::string op_type = node->stmt()->op_type();
      auto iter = std::find(int8_ops_.begin(), int8_ops_.end(), op_type);
      if (iter != int8_ops_.end()) {
        nodes.push_back(node);
      }
    }
  }

  for (auto* node : nodes) {
    const std::string op_type = node->stmt()->op_type();
    VLOG(4) << "op_type: " << op_type;
    OpInfo* op_info = node->stmt()->mutable_op_info();
    auto* scope = node->stmt()->op()->scope();
    for (auto* in_node : node->inlinks) {
      CHECK(in_node->IsArg()) << "The input node should be variable.";
      bool enable_int8 = op_info->HasAttr("enable_int8") ? true : false;
      if (enable_int8) {
        auto weight_name = op_info->Input("Filter").front();
        auto input_name = op_info->Input("Input").front();
        auto weight_scale = op_info->GetInputScale(weight_name);
        auto input_scale = op_info->GetInputScale(input_name);
        if (op_type == "fc") {
          weight_scale = op_info->GetInputScale("W0_scale");
          input_scale = op_info->GetInputScale("Input0_scale");
          weight_name = op_info->Input("W").front();
        }
        auto conv_weight_t =
            scope->FindVar(weight_name)->GetMutable<lite::Tensor>();
        auto conv_weight_d = conv_weight_t->data<int8_t>();
        int out_channel = conv_weight_t->dims()[0];
        int size = conv_weight_t->data_size() / out_channel;
        CHECK_EQ(weight_scale.size(), out_channel)
            << "Int8 size of weight_scale must be equal out_channel, "
            << " actual size of weight_scale is: " << weight_scale.size()
            << ", out_channel is: " << out_channel;

        if (op_info->HasInput("Bias") && op_info->Input("Bias").size() > 0) {
          auto bias_name = op_info->Input("Bias").front();
          auto conv_bias_t =
              scope->FindVar(bias_name)->GetMutable<lite::Tensor>();
          auto conv_bias_d = conv_bias_t->mutable_data<float>();
          compute_new_bias(conv_bias_d,
                           conv_weight_d,
                           conv_bias_d,
                           weight_scale,
                           input_scale,
                           out_channel,
                           size);

        } else {
          auto* bias_tensor = scope->NewTensor("new_bias");
          bias_tensor->Resize({out_channel});
          // data
          auto bias_d = bias_tensor->mutable_data<float>();
          bias_tensor->set_persistable(true);
          compute_new_bias(bias_d,
                           conv_weight_d,
                           nullptr,
                           weight_scale,
                           input_scale,
                           out_channel,
                           size);
          op_info->SetInput("Bias", {"new_bias"});
          node->stmt()->ResetOp(*op_info, graph->valid_places());
        }
      }
    }
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(x86_int8_attribute_pass,
                  paddle::lite::mir::X86Int8AttributePass)
    .BindTargets({TARGET(kX86)});
