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

#include "lite/operators/lod_array_length_op.h"

#include <vector>

#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool LoDArrayLengthOp::InferShapeImpl() const {
  std::vector<int64_t> out_dims = {1};
  param_.out->Resize(lite::DDim(out_dims));
  return true;
}

bool LoDArrayLengthOp::AttachImpl(const cpp::OpDesc &opdesc,
                                  paddle::lite::Scope *scope) {
  auto x_name = opdesc.Input("X").front();
  auto out = opdesc.Output("Out").front();
  param_.x = scope->FindVar(x_name)->GetMutable<std::vector<Tensor>>();
  param_.out = GetMutableTensor(scope, out);
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(lod_array_length, paddle::lite::operators::LoDArrayLengthOp);
