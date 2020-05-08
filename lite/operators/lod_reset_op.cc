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

#include "lite/operators/lod_reset_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool LodResetOp::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Out);
  return true;
}

bool LodResetOp::InferShapeImpl() const {
  CHECK_OR_FALSE(param_.Out);
  // TODO(Superjomn) Enable data sharing.
  param_.Out->Resize(param_.X->dims());
  if (param_.Y) {
  } else {
    CHECK_GT(param_.target_lod.size(), 0)
        << "target lod must be provided when Y is not exist";
  }
  return true;
}

bool LodResetOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  param_.X =
      scope->FindVar(opdesc.Input("X").front())->GetMutable<lite::Tensor>();
  if (opdesc.Input("Y").size()) {
    param_.Y =
        scope->FindVar(opdesc.Input("Y").front())->GetMutable<lite::Tensor>();
  }
  param_.Out =
      scope->FindVar(opdesc.Output("Out").front())->GetMutable<lite::Tensor>();
  CHECK(param_.X);
  CHECK(param_.Out);
  param_.target_lod = opdesc.GetAttr<std::vector<int>>("target_lod");
  // param_.append = opdesc.GetAttr<bool>("append");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(lod_reset, paddle::lite::operators::LodResetOp);
