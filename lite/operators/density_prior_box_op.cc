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
  CHECK_OR_FALSE(param_.input);
  CHECK_OR_FALSE(param_.image);
  CHECK_OR_FALSE(param_.boxes);
  CHECK_OR_FALSE(param_.variances);
  return true;
}

bool DensityPriorBoxOpLite::InferShapeImpl() const { return true; }

bool DensityPriorBoxOpLite::AttachImpl(const cpp::OpDesc& opdesc,
                                       lite::Scope* scope) {
  auto input = opdesc.Input("Input").front();
  auto image = opdesc.Input("Image").front();
  auto boxes = opdesc.Output("Boxes").front();
  auto variances = opdesc.Output("Variances").front();

  param_.input = scope->FindVar(input)->GetMutable<lite::Tensor>();
  param_.image = scope->FindVar(image)->GetMutable<lite::Tensor>();
  param_.boxes = scope->FindVar(boxes)->GetMutable<lite::Tensor>();
  param_.variances = scope->FindVar(variances)->GetMutable<lite::Tensor>();

  param_.clip = opdesc.GetAttr<bool>("clip");
  param_.fixed_sizes = opdesc.GetAttr<std::vector<float>>("fixed_sizes");
  param_.fixed_ratios = opdesc.GetAttr<std::vector<float>>("fixed_ratios");
  param_.variances_ = opdesc.GetAttr<std::vector<float>>("variances");

  if (opdesc.HasAttr("aspect_ratios")) {
    param_.aspect_ratios = opdesc.GetAttr<std::vector<float>>("aspect_ratios");
  }
  if (opdesc.HasAttr("max_sizes")) {
    param_.max_sizes = opdesc.GetAttr<std::vector<float>>("max_sizes");
  }
  if (opdesc.HasAttr("density_sizes")) {
    param_.density_sizes = opdesc.GetAttr<std::vector<int>>("density_sizes");
  }
  if (opdesc.HasAttr("densities")) {
    param_.density_sizes = opdesc.GetAttr<std::vector<int>>("densities");
  }
  if (opdesc.HasAttr("min_sizes")) {
    param_.min_sizes = opdesc.GetAttr<std::vector<float>>("min_sizes");
  }
  if (opdesc.HasAttr("flip")) {
    param_.flip = opdesc.GetAttr<bool>("flip");
  }
  if (opdesc.HasAttr("img_w")) {
    param_.img_w = opdesc.GetAttr<int>("img_w");
  }
  if (opdesc.HasAttr("img_h")) {
    param_.img_h = opdesc.GetAttr<int>("img_h");
  }
  if (opdesc.HasAttr("step_w")) {
    param_.step_w = opdesc.GetAttr<float>("step_w");
  }
  if (opdesc.HasAttr("step_h")) {
    param_.step_h = opdesc.GetAttr<float>("step_h");
  }
  param_.offset = opdesc.GetAttr<float>("offset");
  if (opdesc.HasAttr("prior_num")) {
    param_.prior_num = opdesc.GetAttr<int>("prior_num");
  }
  if (opdesc.HasAttr("order")) {
    param_.order = opdesc.GetAttr<std::vector<std::string>>("order");
  }
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(density_prior_box,
                 paddle::lite::operators::DensityPriorBoxOpLite);
