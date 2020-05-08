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

#include "lite/operators/increment_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool IncrementOp::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Out);
  return true;
}

bool IncrementOp::InferShapeImpl() const {
  CHECK_OR_FALSE(param_.Out);
  // TODO(Superjomn) Enable data sharing.
  auto out_dims = param_.X->dims();
  param_.Out->Resize(out_dims);
  return true;
}

bool IncrementOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  param_.X = scope->FindMutableTensor(opdesc.Input("X").front());
  param_.Out = scope->FindMutableTensor(opdesc.Output("Out").front());
  CHECK(param_.X);
  CHECK(param_.Out);
  param_.step = opdesc.GetAttr<float>("step");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(increment, paddle::lite::operators::IncrementOp);
