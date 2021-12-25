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

#include "lite/operators/__xpu__softmax_topk_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool XPUSoftmaxTopkOp::CheckShape() const {
  CHECK_OR_FALSE(param_.x);
  CHECK_OR_FALSE(param_.output);
  CHECK_OR_FALSE(param_.indices);
  auto x_dims = param_.x->dims();
  auto x_rank = x_dims.size();
  CHECK_OR_FALSE(param_.axis >= -static_cast<int>(x_rank) &&
                 param_.axis < static_cast<int>(x_rank));
  return true;
}

bool XPUSoftmaxTopkOp::InferShapeImpl() const {
  auto out_dims = param_.x->dims();
  out_dims[out_dims.size() - 1] = param_.K;
  auto out = param_.output;
  out->Resize(out_dims);
  out->set_lod(param_.x->lod());
  auto indices = param_.indices;
  indices->Resize(out_dims);
  indices->set_lod(param_.x->lod());

  return true;
}

bool XPUSoftmaxTopkOp::AttachImpl(const cpp::OpDesc &opdesc,
                                  lite::Scope *scope) {
  param_.x = scope->FindTensor(opdesc.Input("X").front());
  param_.output = scope->FindMutableTensor(opdesc.Output("Out").front());
  param_.indices = scope->FindMutableTensor(opdesc.Output("Indices").front());
  param_.K = opdesc.GetAttr<int>("k");
  if (opdesc.HasAttr("axis")) {
    param_.axis = opdesc.GetAttr<int>("axis");
  } else {
    param_.axis = -1;
  }
  CHECK(param_.x);
  CHECK(param_.output);
  CHECK(param_.indices);
  CHECK_GE(param_.K, 1) << "XPUSoftmaxTopk param K is " << param_.K
                        << " which is not valid";
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(__xpu__softmax_topk,
                 paddle::lite::operators::XPUSoftmaxTopkOp);
