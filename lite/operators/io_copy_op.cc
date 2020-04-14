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

#include "lite/operators/io_copy_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool IoCopyOp::CheckShape() const {
  CHECK_OR_FALSE(param_.x);
  CHECK_OR_FALSE(param_.y);
  return true;
}
bool IoCopyOp::InferShapeImpl() const {
  param_.y->Resize(param_.x->dims());
  param_.y->set_lod(param_.x->lod());
  return true;
}
bool IoCopyOp::Run() { return OpLite::Run(); }
bool IoCopyOp::AttachImpl(const cpp::OpDesc &opdesc,
                          paddle::lite::Scope *scope) {
  auto x = opdesc.Input("Input").front();
  auto out = opdesc.Output("Out").front();
  param_.x = GetTensor(scope, x);
  param_.y = GetMutableTensor(scope, out);
  if (opdesc.HasAttr("process_type")) {
    param_.process_type = opdesc.GetAttr<int>("process_type");
  }
  return true;
}
std::string IoCopyOp::DebugString() const { return "io_copy_op"; }

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(io_copy, paddle::lite::operators::IoCopyOp);
