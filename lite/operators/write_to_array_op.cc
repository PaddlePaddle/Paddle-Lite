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

#include "lite/operators/write_to_array_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool WriteToArrayOp::CheckShape() const {
  CHECK(param_.X);
  CHECK(param_.I);
  CHECK(param_.Out);
  return true;
}

bool WriteToArrayOp::InferShapeImpl() const { return true; }

bool WriteToArrayOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  auto inputs = opdesc.Input("X").front();
  param_.X = scope->FindTensor(inputs);

  auto id = opdesc.Input("I").front();
  param_.I = scope->FindTensor(id);

  auto out = opdesc.Output("Out").front();
  param_.Out = scope->FindVar(out)->GetMutable<std::vector<Tensor>>();
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(write_to_array, paddle::lite::operators::WriteToArrayOp);
