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
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Out);
  return true;
}

bool ActivationOp::InferShape() const {
  param_.Out->Resize(param_.X->dims());
  auto out_lod = param_.Out->mutable_lod();
  *out_lod = param_.X->lod();
  return true;
}

bool ActivationOp::AttachImpl(const cpp::OpDesc& opdesc, lite::Scope* scope) {
  auto x_name = opdesc.Input("X").front();
  auto out_name = opdesc.Output("Out").front();
  param_.X = scope->FindVar(x_name)->GetMutable<lite::Tensor>();
  if (opdesc.Type() == "leaky_relu") {
    param_.Leaky_relu_alpha = opdesc.GetAttr<float>("alpha");
  }
  if (opdesc.Type() == "relu_clipped") {
    param_.Relu_clipped_coef = opdesc.GetAttr<float>("Relu_clipped_coef");
  }
  if (opdesc.Type() == "prelu") {
    param_.Prelu_mode = opdesc.GetAttr<std::string>("mode");
    auto prelu_alpha_name = opdesc.Input("Alpha").front();
    param_.Prelu_alpha =
        scope->FindVar(prelu_alpha_name)->GetMutable<lite::Tensor>();
  }
  if (opdesc.Type() == "swish") {
    param_.Swish_beta = opdesc.GetAttr<float>("beta");
  }
  param_.Out = scope->FindVar(out_name)->GetMutable<lite::Tensor>();
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
REGISTER_LITE_OP(relu, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(leaky_relu, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(relu_clipped, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(prelu, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(sigmoid, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(tanh, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(swish, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(relu6, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(log, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(exp, paddle::lite::operators::ActivationOp);
REGISTER_LITE_OP(floor, paddle::lite::operators::ActivationOp);

#ifdef LITE_WITH_TRAIN
REGISTER_LITE_OP(square_grad, paddle::lite::operators::ActivationGradOp);
#endif
