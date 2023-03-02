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

#include "lite/operators/__xpu__gn_silu_op.h"
#include <vector>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool XPUGnSiluOp::CheckShape() const { return true; }

bool XPUGnSiluOp::InferShapeImpl() const {
  param_.output->Resize(param_.input->dims());
  return true;
}

bool XPUGnSiluOp::AttachImpl(const cpp::OpDesc& op_desc, lite::Scope* scope) {
  param_.input = scope->FindTensor(op_desc.Input("Input").front());
  param_.output = scope->FindMutableTensor(op_desc.Output("Output").front());

  param_.gn_scale.clear();
  for (auto& name : op_desc.Input("GNScale")) {
    auto t = scope->FindVar(name)->GetMutable<Tensor>();
    param_.gn_scale.push_back(t);
  }
  param_.gn_bias.clear();
  for (auto& name : op_desc.Input("GNBias")) {
    auto t = scope->FindVar(name)->GetMutable<Tensor>();
    param_.gn_bias.push_back(t);
  }
  param_.groups = op_desc.GetAttr<int>("groups");
  param_.epsilon = op_desc.GetAttr<float>("epsilon");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(__xpu__gn_silu, paddle::lite::operators::XPUGnSiluOp);
