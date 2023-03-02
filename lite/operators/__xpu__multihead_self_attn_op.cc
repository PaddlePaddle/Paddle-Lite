// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/operators/__xpu__multihead_self_attn_op.h"
#include <vector>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool XPUMhsaOp::CheckShape() const {
  CHECK_EQ(param_.input->dims().size(), 3UL);
  return true;
}

bool XPUMhsaOp::InferShapeImpl() const {
  param_.output->Resize(param_.input->dims());
  return true;
}

bool XPUMhsaOp::AttachImpl(const cpp::OpDesc& op_desc, lite::Scope* scope) {
  param_.input = scope->FindTensor(op_desc.Input("Input").front());
  param_.output = scope->FindMutableTensor(op_desc.Output("Output").front());

  param_.fc_weight.clear();
  for (auto& name : op_desc.Input("FCWeight")) {
    auto t = scope->FindVar(name)->GetMutable<Tensor>();
    param_.fc_weight.push_back(t);
  }
  param_.fc_bias.clear();
  for (auto& name : op_desc.Input("FCBias")) {
    auto t = scope->FindVar(name)->GetMutable<Tensor>();
    param_.fc_bias.push_back(t);
  }
  param_.ln_scale.clear();
  for (auto& name : op_desc.Input("LNScale")) {
    auto t = scope->FindVar(name)->GetMutable<Tensor>();
    param_.ln_scale.push_back(t);
  }
  param_.ln_bias.clear();
  for (auto& name : op_desc.Input("LNBias")) {
    auto t = scope->FindVar(name)->GetMutable<Tensor>();
    param_.ln_bias.push_back(t);
  }

  param_.hidden_dim = op_desc.GetAttr<int>("hidden_dim");
  param_.head_num = op_desc.GetAttr<int>("head_num");
  param_.size_per_head = op_desc.GetAttr<int>("size_per_head");

  param_.weight_max.clear();
  for (const auto& weight_max_tensor :
       op_desc.GetAttr<std::vector<std::string>>("FCWeightMax")) {
    auto tensor = scope->FindMutableTensor(weight_max_tensor);
    CHECK(tensor != nullptr);
    param_.weight_max.push_back(tensor);
  }
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(__xpu__multihead_self_attn,
                 paddle::lite::operators::XPUMhsaOp);
