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

bool WriteToArrayOp::CheckShape() const { return true; }

bool WriteToArrayOp::InferShape() const {
  auto in_dims = param_.X->dims();
  for (auto out : *param_.Out) {
    out.Resize(in_dims);
  }
  return true;
}

bool WriteToArrayOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  auto inputs = opdesc.Input("X").front();
  param_.X = scope->FindVar(inputs)->GetMutable<lite::Tensor>();

  auto id = opdesc.Input("I").front();
  param_.I = scope->FindVar(id)->GetMutable<lite::Tensor>();

  auto out = opdesc.Output("Out").front();
  param_.Out = scope->FindVar(out)->GetMutable<std::vector<lite::Tensor>>();
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(write_to_array, paddle::lite::operators::WriteToArrayOp);
