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

bool LoDArrayLengthOp::CheckShape() const {
  CHECK(param_.x.empty());
  CHECK(param_.out);
  return true;
}
bool LoDArrayLengthOp::InferShapeImpl() const {
  std::vector<int64_t> out_dims = {1};
  for (auto x : param_.x) {
    x->Resize(x->dims());
  }
  param_.out->Resize(lite::DDim(out_dims));
  return true;
}
bool LoDArrayLengthOp::Run() { return OpLite::Run(); }
bool LoDArrayLengthOp::AttachImpl(const cpp::OpDesc &opdesc,
                                  paddle::lite::Scope *scope) {
  auto x_name = opdesc.Input("X");
  auto out = opdesc.Output("Out").front();
  for (auto var : x_name) {
    param_.x.push_back(GetMutableTensor(scope, var));
  }
  param_.out = GetMutableTensor(scope, out);
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(lod_array_length, paddle::lite::operators::LoDArrayLengthOp);
