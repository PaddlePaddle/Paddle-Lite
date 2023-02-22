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
  auto input_shape = param_.input->dims();
  auto batch_size = input_shape[0];
  auto channel = input_shape[1];
  auto h = input_shape[2];
  auto w = input_shape[3];
  param_.output->Resize({batch_size, channel, h, w});
  return true;
}

bool XPUGnSiluOp::AttachImpl(const cpp::OpDesc& op_desc, lite::Scope* scope) {
  param_.input =
      scope->FindVar(op_desc.Input("Input").front())->GetMutable<Tensor>();
  param_.output =
      scope->FindVar(op_desc.Output("Output").front())->GetMutable<Tensor>();

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
