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

#include "lite/operators/mean_grad_op.h"
#include <vector>
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool MeanGradOp::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Out_grad);
  CHECK_OR_FALSE(param_.X_grad);
  return true;
}

bool MeanGradOp::InferShapeImpl() const {
  param_.X_grad->Resize(param_.X->dims());
  return true;
}

bool MeanGradOp::AttachImpl(const cpp::OpDesc& opdesc, lite::Scope* scope) {
  CHECK_EQ(opdesc.InputArgumentNames().size(), 2UL);
  auto X_name = opdesc.Input("X").front();
  auto Out_grad_name = opdesc.Input("Out@GRAD").front();
  auto X_grad_name = opdesc.Output("X@GRAD").front();

  param_.X = GetVar<lite::Tensor>(scope, X_name);
  param_.Out_grad = GetVar<lite::Tensor>(scope, Out_grad_name);
  param_.X_grad = GetMutableVar<Tensor>(scope, X_grad_name);
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(mean_grad, paddle::lite::operators::MeanGradOp);
