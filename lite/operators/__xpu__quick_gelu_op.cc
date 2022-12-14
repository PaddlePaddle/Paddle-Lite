// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/operators/__xpu__quick_gelu_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool XPUQuickGeluOp::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Out);
  return true;
}

bool XPUQuickGeluOp::InferShapeImpl() const {
  param_.Out->Resize(param_.X->dims());
  auto out_lod = param_.Out->mutable_lod();
  *out_lod = param_.X->lod();
  return true;
}

bool XPUQuickGeluOp::AttachImpl(const cpp::OpDesc& opdesc, lite::Scope* scope) {
  auto x_name = opdesc.Input("X").front();
  auto out_name = opdesc.Output("Out").front();
  param_.X = scope->FindVar(x_name)->GetMutable<lite::Tensor>();
  param_.Out = scope->FindVar(out_name)->GetMutable<lite::Tensor>();
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(__xpu__quick_gelu, paddle::lite::operators::XPUQuickGeluOp);
