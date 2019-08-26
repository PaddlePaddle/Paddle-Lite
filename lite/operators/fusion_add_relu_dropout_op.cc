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

#include "lite/operators/fusion_add_relu_dropout_op.h"
#include <string>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool AddReluDropoutFusionOp::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Y);
  CHECK_OR_FALSE(param_.Out);
  return true;
}

bool AddReluDropoutFusionOp::InferShape() const {
  param_.Out->Resize(param_.X->dims());
  auto lod = param_.Out->mutable_lod();
  *lod = param_.X->lod();
  return true;
}

bool AddReluDropoutFusionOp::AttachImpl(const cpp::OpDesc& opdesc,
                                               lite::Scope* scope) {
  auto X_name = opdesc.Input("X").front();
  auto Y_name = opdesc.Input("Y").front();
  auto Out_name = opdesc.Output("Out").front();

  param_.X = GetVar<lite::Tensor>(scope, X_name);
  param_.Y = GetVar<lite::Tensor>(scope, Y_name);
  param_.Out = GetMutableVar<lite::Tensor>(scope, Out_name);
  param_.scale = opdesc.GetAttr<float>("dropout_prob");
  param_.act_type = "relu";

  return true;
}
}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(fusion_add_relu_dropout,
                 paddle::lite::operators::AddReluDropoutFusionOp);

