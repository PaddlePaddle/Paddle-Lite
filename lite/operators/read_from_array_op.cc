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

#include "lite/operators/read_from_array_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool ReadFromArrayOp::CheckShape() const { return true; }

bool ReadFromArrayOp::InferShape() const {
  auto in_dims = param_.X[0]->dims();
  param_.Out->Resize(in_dims);
  return true;
}

bool ReadFromArrayOp::AttachImpl(const cpp::OpDesc &opdesc,
                                 lite::Scope *scope) {
  auto inputs = opdesc.Input("X");
  for (auto in : inputs) {
    param_.X.push_back(scope->FindVar(in)->GetMutable<lite::Tensor>());
  }

  param_.I =
      scope->FindVar(opdesc.Input("I").front())->GetMutable<lite::Tensor>();

  param_.Out =
      scope->FindVar(opdesc.Output("Out").front())->GetMutable<lite::Tensor>();
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(read_from_array, paddle::lite::operators::ReadFromArrayOp);
