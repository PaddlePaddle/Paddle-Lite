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

#include "lite/operators/one_hot_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool OneHotOp::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Out);
  return true;
}

bool OneHotOp::InferShape() const {
  CHECK_OR_FALSE(param_.Out);
  // TODO(Superjomn) Enable data sharing.
  auto out_dims = param_.X->dims();

  out_dims[out_dims.size() - 1] = param_.depth;
  param_.Out->Resize(out_dims);
  return true;
}

bool OneHotOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  param_.X =
      scope->FindVar(opdesc.Input("X").front())->GetMutable<lite::Tensor>();
  param_.Out =
      scope->FindVar(opdesc.Output("Out").front())->GetMutable<lite::Tensor>();

  if (opdesc.HasInput("depth_tensor")) {
    auto depth_tensor = opdesc.Input("depth_tensor").front();
    param_.depth_tensor =
        scope->FindVar(depth_tensor)->GetMutable<lite::Tensor>();
  }

  CHECK(param_.X);
  CHECK(param_.Out);
  param_.depth = opdesc.GetAttr<int>("depth");
  param_.dtype = opdesc.GetAttr<int>("dtype");

  if (opdesc.HasAttr("allow_out_of_range")) {
    param_.allow_out_of_range = opdesc.GetAttr<bool>("allow_out_of_range");
  }

  auto out_lod = param_.Out->mutable_lod();
  *out_lod = param_.X->lod();
  // param_.allow_out_of_range = opdesc.GetAttr<bool>("allow_out_of_range");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(one_hot, paddle::lite::operators::OneHotOp);
