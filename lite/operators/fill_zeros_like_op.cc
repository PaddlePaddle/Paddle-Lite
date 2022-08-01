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

#include "lite/operators/fill_zeros_like_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool FillZerosLikeOp::CheckShape() const {
  CHECK(param_.Out);
  return true;
}

bool FillZerosLikeOp::InferShapeImpl() const {
  param_.Out->Resize(param_.X->dims());
  param_.Out->set_lod(param_.X->lod());
  return true;
}

bool FillZerosLikeOp::AttachImpl(const cpp::OpDesc& opdesc,
                                 lite::Scope* scope) {
  param_.X = scope->FindTensor(opdesc.Input("X").front());
  param_.Out = scope->FindMutableTensor(opdesc.Output("Out").front());
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(fill_zeros_like, paddle::lite::operators::FillZerosLikeOp);
