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

#include "lite/operators/pad2d_op.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool Pad2dOpLite::CheckShape() const {
  CHECK_GT_OR_FALSE(param_.X->dims().size(), 1UL);
  CHECK_OR_FALSE(param_.Out);
  return true;
}

bool Pad2dOpLite::InferShape() const {
  // nchw
  auto x_dims = param_.X->dims();
  int out_h = x_dims[2] + param_._pad_h[0] + param_._pad_h[1];
  int out_w = x_dims[3] + param_._pad_w[0] + param_._pad_w[1];
  param_.Out->Resize(lite::DDim({x_dims[0], x_dims[1], out_h, out_w}));
  return true;
}

// TODO(Superjomn) replace framework::OpDesc with a lite one.
bool Pad2dOpLite::AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) {
  //////////////////////////
  ///////  lite::Tensor类型（输入输出类型）的写入。-----先写入输入类型
  param_.X = scope->FindVar(op_desc.Input("X").front())->GetMutable<Tensor>();
  //输出类型设置
  param_.Out =
      scope->FindVar(op_desc.Output("Out").front())->GetMutable<Tensor>();
  // int型变量输入
  param_._mode = op_desc.GetAttr<int>("_mode");
  // float型变量输入
  param_._pad_value = op_desc.GetAttr<float>("_pad_value");
  // vector型变量的输入
  param_._pad_h = op_desc.GetAttr<std::vector<int>>("_pad_h");
  param_._pad_w = op_desc.GetAttr<std::vector<int>>("_pad_w");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(pad2d, paddle::lite::operators::Pad2dOpLite);
