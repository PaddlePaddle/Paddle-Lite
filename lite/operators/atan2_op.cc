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

#include "lite/operators/atan2_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool Atan2Op::CheckShape() const {
  CHECK_OR_FALSE(param_.X1);
  CHECK_OR_FALSE(param_.X2);
  CHECK_OR_FALSE(param_.Out);
  return true;
}

bool Atan2Op::InferShapeImpl() const {
  auto x_dim = param_.X1->dims();

  // Set output dims
  param_.Out->Resize(lite::DDim(x_dim));
  return true;
}

bool Atan2Op::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  auto X1_name = opdesc.Input("X1").front();
  auto X2_name = opdesc.Input("X2").front();
  auto Out_name = opdesc.Output("Out").front();

  param_.X1 = scope->FindVar(X1_name)->GetMutable<lite::Tensor>();
  param_.X2 = scope->FindVar(X2_name)->GetMutable<lite::Tensor>();

  param_.Out = scope->FindVar(Out_name)->GetMutable<lite::Tensor>();
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

#ifdef LITE_BUILD_EXTRA
REGISTER_LITE_OP(atan2, paddle::lite::operators::Atan2Op);
#endif
