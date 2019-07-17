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

#include "lite/operators/activation_ops.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool ActivationOp::CheckShape() const {
  CHECK_OR_FALSE(param_.x);
  CHECK_OR_FALSE(param_.out);
  return true;
}

bool ActivationOp::InferShape() const {
  param_.out->Resize(param_.x->dims());
  return true;
}

bool ActivationOp::AttachImpl(const cpp::OpDesc& opdesc, lite::Scope* scope) {
  auto x_name = opdesc.Input("x").front();
  auto prelu_channel_slope_name = opdesc.Input("prelu_channel_slope").front();
  auto out_name = opdesc.Output("out").front();

  param_.x = GetVar<lite::Tensor>(scope, x_name);
  param_.relu_neg_slope = op_desc.GetAttr<float>("relu_neg_slope");
  param_.relu_clipped_coef = op_desc.GetAttr<float>("relu_clipped_coef");
  param_.prelu_channel_shared = op_desc.GetAttr<bool>("prelu_channel_shared");
  param_.prelu_channel_slope =
      GetVar<lite::Tensor>(scope, prelu_channel_slope_name);
  param_.swish_coef = op_desc.GetAttr<float>("swish_coef");
  param_.out = GetMutableVar<lite::Tensor>(scope, out_name);
  return true;
}

#ifdef LITE_WITH_TRAIN

bool ActivationGradOp::CheckShape() const {
  CHECK_OR_FALSE(param_.X_grad);
  CHECK_OR_FALSE(param_.Out_grad);
  return true;
}

bool ActivationGradOp::InferShape() const {
  param_.X_grad->Resize(param_.Out_grad->dims());
  return true;
}

bool ActivationGradOp::AttachImpl(const cpp::OpDesc& opdesc,
                                  lite::Scope* scope) {
  auto Out_grad_name = opdesc.Input(framework::GradVarName("Out")).front();
  auto X_grad_name = opdesc.Output(framework::GradVarName("X")).front();

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

#endif

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(square, paddle::lite::operators::ActivationOp);
// REGISTER_LITE_OP(relu_1, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(relu_neg, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(relu_clipped, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(prelu, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(sigmoid, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(tanh, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(swish, paddle::lite::operators::ActivationOp);

#ifdef LITE_WITH_TRAIN
REGISTER_LITE_OP(square_grad, paddle::lite::operators::ActivationGradOp);
#endif
