// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/operators/elementwise_grad_ops.h"
#include <algorithm>
#include <cmath>
#include "lite/core/op_registry.h"
namespace paddle {
namespace lite {
namespace operators {

bool ElementwiseGradOp::CheckShape() const {
  CHECK_OR_FALSE(param_.XGrad || param_.YGrad);
  CHECK_OR_FALSE(param_.OutGrad);
  return true;
}

bool ElementwiseGradOp::InferShapeImpl() const {
  auto x_dim = param_.X->dims();
  auto y_dim = param_.Y->dims();
  if (param_.XGrad) {
    param_.XGrad->Resize(x_dim);
  }
  if (param_.YGrad) {
    param_.YGrad->Resize(y_dim);
  }
  return true;
}

bool ElementwiseGradOp::AttachImpl(const cpp::OpDesc& opdesc,
                                   lite::Scope* scope) {
  auto Y_name = opdesc.Input("Y").front();
  auto X_name = opdesc.Input("X").front();
  auto Out_name = opdesc.Input("Out@GRAD").front();
  CHECK(!opdesc.Output("X@GRAD").empty() || !opdesc.Output("Y@GRAD").empty())
      << "at least one of 'X@GRAD' and 'Y@GRAD' is not empty";

  if (!opdesc.Output("X@GRAD").empty()) {
    auto x_grad_name = opdesc.Output("X@GRAD").front();
    param_.XGrad = GetMutableVar<lite::Tensor>(scope, x_grad_name);
  }
  if (!opdesc.Output("Y@GRAD").empty()) {
    auto y_grad_name = opdesc.Output("Y@GRAD").front();
    param_.YGrad = GetMutableVar<lite::Tensor>(scope, y_grad_name);
  }

  param_.X = GetVar<lite::Tensor>(scope, X_name);
  param_.Y = GetVar<lite::Tensor>(scope, Y_name);
  param_.OutGrad = GetVar<lite::Tensor>(scope, Out_name);
  param_.axis = opdesc.GetAttr<int>("axis");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(elementwise_sub_grad,
                 paddle::lite::operators::ElementwiseGradOp);
REGISTER_LITE_OP(elementwise_add_grad,
                 paddle::lite::operators::ElementwiseGradOp);

REGISTER_LITE_OP(elementwise_grad_mul,
                 paddle::lite::operators::ElementwiseGradOp);
REGISTER_LITE_OP(elementwise_grad_max,
                 paddle::lite::operators::ElementwiseGradOp);
