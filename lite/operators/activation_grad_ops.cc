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
// limitations under the License.i

#include "lite/operators/activation_grad_ops.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool ActivationGradOp::CheckShape() const {
  CHECK_OR_FALSE(param_.X_grad);
  CHECK_OR_FALSE(param_.Out_grad);
  return true;
}

bool ActivationGradOp::InferShapeImpl() const {
  param_.X_grad->Resize(param_.Out_grad->dims());
  return true;
}

bool ActivationGradOp::AttachImpl(const cpp::OpDesc& opdesc,
                                  lite::Scope* scope) {
  auto Out_grad_name = opdesc.Input("Out@GRAD").front();
  auto X_grad_name = opdesc.Output("X@GRAD").front();

  param_.Out_grad = GetVar<lite::Tensor>(scope, Out_grad_name);
  param_.X_grad = GetMutableVar<Tensor>(scope, X_grad_name);

  if (opdesc.HasInput("X")) {
    auto X_name = opdesc.Input("X").front();
    param_.X = GetVar<lite::Tensor>(scope, X_name);
  } else {
    param_.X = param_.X_grad;
  }

  if (opdesc.HasInput("Out")) {
    auto Out_name = opdesc.Input("Out").front();
    param_.Out = GetVar<lite::Tensor>(scope, Out_name);
  } else {
    param_.Out = param_.Out_grad;
  }

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(square_grad, paddle::lite::operators::ActivationGradOp);
