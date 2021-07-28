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

#include "lite/operators/reverse_op.h"
#include <vector>
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool ReverseOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Out);
  for (auto axis : param_.Axis) {
    CHECK_OR_FALSE(axis < static_cast<int>((param_.X)->dims().size()));
    CHECK_OR_FALSE(axis >= static_cast<int>(-(param_.X)->dims().size()));
  }
  return true;
}

bool ReverseOpLite::InferShapeImpl() const {
  auto x_dims = param_.X->dims();
  int x_rank = x_dims.size();

  std::vector<int64_t> out_dims;
  for (int64_t i = 0; i < x_rank; i++) out_dims.push_back(x_dims[i]);

  param_.Out->Resize(lite::DDim(out_dims));
  return true;
}

// TODO(Superjomn) replace framework::OpDesc with a lite one.
bool ReverseOpLite::AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) {
  auto x = op_desc.Input("X").front();
  auto out = op_desc.Output("Out").front();

  param_.X = scope->FindVar(x)->GetMutable<lite::Tensor>();
  param_.Out = scope->FindVar(out)->GetMutable<lite::Tensor>();
  param_.Axis = op_desc.GetAttr<std::vector<int>>("axis");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(reverse, paddle::lite::operators::ReverseOpLite);
