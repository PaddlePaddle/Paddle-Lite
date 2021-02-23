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

#include "lite/core/mir/quant_fp16_pass.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include "lite/api/paddle_place.h"
#include "lite/core/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {
void QuantDataFP32toFP16(OpInfo* op_info,
                         Tensor* weight,
                         const std::string weight_name,
                         int quant_bits) {
  const DDim weight_dims = weight->dims();
  CHECK(weight_dims.size() == 2 || weight_dims.size() == 4);

  // quantize weights
  Tensor tmp_tensor;
  tmp_tensor.CopyDataFrom(*weight);
  weight->clear();
  weight->set_precision(PRECISION(kFP16));
  float16_t* weight_data = weight->mutable_data<float16_t>();
  const float* src = tmp_tensor.data<float>();
  for (int i = 0; i < weight_dims.production(); i++) {
    weight_data[i] = static_cast<float16_t>(src[i]);
  }
  op_info->SetAttr<std::string>("quantization_type", "fp32_to_fp16");
  op_info->SetAttr("quantize_weight_bits", quant_bits);
  op_info->SetAttr(weight_name + "_quant_scale", 1.f);
}

void QuantFP16Pass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  std::vector<mir::Node*> nodes;
  for (auto* node : graph->StmtTopologicalOrder()) {
    if (node->IsStmt()) {
      const std::string op_type = node->stmt()->op_type();
      auto iter = std::find(quant_ops_.begin(), quant_ops_.end(), op_type);
      if (iter != quant_ops_.end()) {
        nodes.push_back(node);
      }
    }
  }

  for (auto* node : nodes) {
    const std::string op_type = node->stmt()->op_type();
    OpInfo* op_info = node->stmt()->mutable_op_info();
    auto* scope = node->stmt()->op()->scope();
    for (auto* in_node : node->inlinks) {
      CHECK(in_node->IsArg()) << "The input node should be variable.";
      if (in_node->arg()->is_weight) {
        std::string weight_name = in_node->arg()->name;
        Tensor* weight = scope->FindVar(weight_name)->GetMutable<Tensor>();
        CHECK(weight) << "Can not find the weight in scope.";
        if (weight->precision() != PrecisionType::kFloat) {
          LOG(INFO) << "The dtype of weight is not fp32, "
                    << "so skip quantizing the weight of " << weight_name;
          continue;
        }
        QuantDataFP32toFP16(op_info, weight, weight_name, quant_bits);
      }
    }
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(quant_fp16_pass, paddle::lite::mir::QuantDataFP32toFP16)
    .BindTargets({TARGET(kARM), TARGET(kX86)});
