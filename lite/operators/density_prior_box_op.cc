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

#include "lite/operators/density_prior_box_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool DensityPriorBoxOpLite::CheckShape() const {
  // CHECK_OR_FALSE(param_.ins);
  // CHECK_OR_FALSE(param_.outs);

  CHECK_EQ(param_.ins.size(), 2);
  return true;
}

bool DensityPriorBoxOpLite::InferShape() const { return true; }

bool DensityPriorBoxOpLite::AttachImpl(const cpp::OpDesc& opdesc,
                                       lite::Scope* scope) {
  auto inputs = opdesc.Input("Input");
  auto outputs = opdesc.Output("Output");

  for (auto var : inputs) {
    param_.ins.push_back(scope->FindVar(var)->GetMutable<lite::Tensor>());
  }
  for (auto var : outputs) {
    param_.outs.push_back(scope->FindVar(var)->GetMutable<lite::Tensor>());
  }

  param_.is_flip = opdesc.GetAttr<bool>("is_flip");
  param_.is_clip = opdesc.GetAttr<bool>("is_clip");
  param_.min_size = opdesc.GetAttr<std::vector<float>>("min_size");
  param_.fixed_size = opdesc.GetAttr<std::vector<float>>("fixed_size");
  param_.fixed_ratio = opdesc.GetAttr<std::vector<float>>("fixed_ratio");
  param_.density_size = opdesc.GetAttr<std::vector<float>>("density_size");
  param_.max_size = opdesc.GetAttr<std::vector<float>>("max_size");
  param_.aspect_ratio = opdesc.GetAttr<std::vector<float>>("aspect_ratio");
  param_.variance = opdesc.GetAttr<std::vector<float>>("variance");
  param_.img_w = opdesc.GetAttr<int>("img_w");
  param_.img_h = opdesc.GetAttr<int>("img_h");
  param_.step_w = opdesc.GetAttr<float>("step_w");
  param_.step_h = opdesc.GetAttr<float>("step_h");
  param_.offset = opdesc.GetAttr<float>("offset");
  param_.prior_num = opdesc.GetAttr<int>("prior_num");
  param_.order = opdesc.GetAttr<std::vector<std::string>>("order");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(density_prior_box,
                 paddle::lite::operators::DensityPriorBoxOpLite);
