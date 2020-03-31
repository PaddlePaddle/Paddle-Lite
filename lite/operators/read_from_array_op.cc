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

bool ReadFromArrayOp::CheckShape() const {
  CHECK(param_.X);
  CHECK(param_.I);
  CHECK(param_.Out);
  return true;
}

bool ReadFromArrayOp::InferShapeImpl() const {
  int id = param_.I->data<int64_t>()[0];
  auto out_dims = (*param_.X)[id].dims();
  param_.Out->Resize(out_dims);
  return true;
}

bool ReadFromArrayOp::AttachImpl(const cpp::OpDesc &opdesc,
                                 lite::Scope *scope) {
  auto in = opdesc.Input("X").front();
  param_.X = scope->FindVar(in)->GetMutable<std::vector<lite::Tensor>>();

  param_.I = scope->FindTensor(opdesc.Input("I").front());

  param_.Out = scope->FindMutableTensor(opdesc.Output("Out").front());
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(read_from_array, paddle::lite::operators::ReadFromArrayOp);
