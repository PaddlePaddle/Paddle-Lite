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
// limitations under the License.

#include "lite/operators/share_data_op.h"

#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool ShareDataOp::CheckShape() const {
  CHECK(param_.X);
  CHECK(param_.Out);
  return true;
}

bool ShareDataOp::InferShapeImpl() const {
  param_.Out->Resize(param_.X->dims());
  return true;
}

bool ShareDataOp::AttachImpl(const cpp::OpDesc& opdesc, lite::Scope* scope) {
  // Input
  auto input_name = opdesc.Input("X").front();
  param_.X = GetTensor(scope, input_name);

  // Out
  auto out_name = opdesc.Output("Out").front();
  param_.Out = GetMutableTensor(scope, out_name);
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(share_data, paddle::lite::operators::ShareDataOp);
