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

#include "lite/operators/unbind_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool UnbindOp::CheckShape() const {
  CHECK_OR_FALSE(param_.x);
  CHECK_GT_OR_FALSE(param_.output.size(), 1UL);
  auto x_dims = param_.x->dims();
  auto x_rank = x_dims.size();
  CHECK_OR_FALSE(param_.axis >= -static_cast<int>(x_rank) &&
                 param_.axis < static_cast<int>(x_rank));
  return true;
}

bool UnbindOp::InferShapeImpl() const {
  const auto &outs = param_.output;
  auto in_dims = param_.x->dims();

  std::vector<int64_t> outs_dims;
  param_.axis = param_.axis >= 0 ? param_.axis : param_.axis + in_dims.size();
  for (int i = 0; i < in_dims.size(); i++) {
    if (i == param_.axis) continue;
    outs_dims.push_back(in_dims[i]);
  }

  for (size_t j = 0; j < outs.size(); ++j) {
    outs[j]->Resize(outs_dims);
  }

  return true;
}

bool UnbindOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  param_.axis = opdesc.GetAttr<int>("axis");
  auto input = opdesc.Input("X").front();
  auto outs = opdesc.Output("Out");
  param_.x = scope->FindVar(input)->GetMutable<lite::Tensor>();
  param_.output.clear();
  for (auto var : outs) {
    param_.output.push_back(scope->FindVar(var)->GetMutable<lite::Tensor>());
  }
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(unbind, paddle::lite::operators::UnbindOp);
