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

#include "lite/operators/__xpu__logit_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool XPULogitOp::CheckShape() const {
  CHECK_OR_FALSE(param_.input);
  CHECK_OR_FALSE(param_.output);
  CHECK_OR_FALSE(param_.eps);
  return true;
}

bool XPULogitOp::InferShapeImpl() const {
  auto out_dims = param_.input->dims();
  auto out = param_.output;
  out->Resize(out_dims);
  out->set_lod(param_.input->lod());
  return true;
}

bool XPULogitOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  CHECK(scope->FindVar(opdesc.Input("X").front()));
  CHECK(scope->FindVar(opdesc.Output("Out").front()));
  param_.input = scope->FindTensor(opdesc.Input("X").front());
  param_.output = scope->FindMutableTensor(opdesc.Output("Out").front());
  param_.eps = opdesc.GetAttr<float>("eps");
  CHECK(param_.input);
  CHECK(param_.output);
  CHECK(param_.eps);
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(__xpu__logit, paddle::lite::operators::XPULogitOp);
