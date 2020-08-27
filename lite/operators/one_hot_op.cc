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

bool OneHotOp::InferShapeImpl() const {
  // Set output dims
  auto out_dims = param_.X->dims();
  CHECK_GE(out_dims.size(), 2);
  out_dims[out_dims.size() - 1] = param_.depth;
  param_.Out->Resize(out_dims);
  param_.Out->set_lod(param_.X->lod());
  return true;
}

bool OneHotOp::AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) {
  auto x = op_desc.Input("X").front();
  auto out = op_desc.Output("Out").front();
  param_.X = scope->FindVar(x)->GetMutable<Tensor>();
  param_.Out = scope->FindMutableTensor(out);

  if (op_desc.HasAttr("depth")) {
    param_.depth = op_desc.GetAttr<int>("depth");
  }

  if (op_desc.HasInput("depth_tensor") &&
      !op_desc.Input("depth_tensor").empty()) {
    auto depth_tensor = op_desc.Input("depth_tensor").front();
    param_.depth_tensor = scope->FindVar(depth_tensor)->GetMutable<Tensor>();
    param_.depth = param_.depth_tensor->data<int32_t>()[0];
  }

  if (op_desc.HasAttr("allow_out_of_range")) {
    param_.allow_out_of_range = op_desc.GetAttr<bool>("allow_out_of_range");
  }
  param_.dtype = op_desc.GetAttr<int>("dtype");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(one_hot, paddle::lite::operators::OneHotOp);
