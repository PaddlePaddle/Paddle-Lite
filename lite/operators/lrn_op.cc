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

#include "lite/operators/lrn_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool LrnOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Out);
  const auto in_dims = param_.X->dims();
  CHECK_EQ(in_dims.size(), 4);
  return true;
}

bool LrnOpLite::InferShapeImpl() const {
  param_.Out->Resize(param_.X->dims());
  return true;
}

bool LrnOpLite::AttachImpl(const cpp::OpDesc& opdesc, lite::Scope* scope) {
  auto X_name = opdesc.Input("X").front();
  auto Out_name = opdesc.Output("Out").front();
  param_.X = GetVar<lite::Tensor>(scope, X_name);
  param_.Out = GetMutableVar<lite::Tensor>(scope, Out_name);
  param_.n = opdesc.GetAttr<int>("n");
  param_.alpha = opdesc.GetAttr<float>("alpha");
  param_.beta = opdesc.GetAttr<float>("beta");
  param_.k = opdesc.GetAttr<float>("k");
  if (opdesc.HasAttr("norm_region")) {
    param_.norm_region = opdesc.GetAttr<std::string>("norm_region");
  }
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(lrn, paddle::lite::operators::LrnOpLite);
