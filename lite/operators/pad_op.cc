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

#include "lite/operators/pad_op.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool PadOpLite::CheckShape() const {
  CHECK_GT_OR_FALSE(param_.X->dims().size(), 1);
  CHECK_OR_FALSE(param_.Out);
  CHECK_EQ(param_.paddings.size(), param_.X->dims().size() * 2);
  for (int i = 0; i < param_.paddings.size(); i++) {
    CHECK_GE_OR_FALSE(param_.paddings[i], 0);
  }
  return true;
}

bool PadOpLite::InferShapeImpl() const {
  auto& x_dims = param_.X->dims();
  auto& paddings = param_.paddings;
  std::vector<int64_t> out_dims(x_dims.size());
  for (int i = 0; i < x_dims.size(); i++) {
    out_dims[i] = x_dims[i] + paddings[i * 2] + paddings[i * 2 + 1];
  }
  param_.Out->Resize(lite::DDim(out_dims));
  return true;
}

bool PadOpLite::AttachImpl(const cpp::OpDesc& op_desc, lite::Scope* scope) {
  param_.X = scope->FindVar(op_desc.Input("X").front())->GetMutable<Tensor>();
  param_.Out =
      scope->FindVar(op_desc.Output("Out").front())->GetMutable<Tensor>();
  param_.pad_value = op_desc.GetAttr<float>("pad_value");
  param_.paddings = op_desc.GetAttr<std::vector<int>>("paddings");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(pad, paddle::lite::operators::PadOpLite);
