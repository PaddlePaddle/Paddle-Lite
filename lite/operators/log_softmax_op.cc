// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/operators/log_softmax_op.h"

#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool LogSoftmaxOp::CheckShape() const {
  CHECK_OR_FALSE(param_.x);
  CHECK_OR_FALSE(param_.output);
  auto x_dims = param_.x->dims();
  auto x_rank = x_dims.size();
  CHECK_OR_FALSE(param_.axis >= -static_cast<int>(x_rank) &&
                 param_.axis < static_cast<int>(x_rank));
  return true;
}

bool LogSoftmaxOp::InferShapeImpl() const {
  auto x_dims = param_.x->dims();
  param_.output->Resize(param_.x->dims());
  return true;
}

bool LogSoftmaxOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  auto x_name = opdesc.Input("X").front();
  auto out_name = opdesc.Output("Out").front();
  param_.x = GetVar<lite::Tensor>(scope, x_name);
  param_.output = GetMutableVar<Tensor>(scope, out_name);

  if (opdesc.HasAttr("axis")) {
    param_.axis = opdesc.GetAttr<int>("axis");
  } else {
    param_.axis = -1;
  }

  CHECK(param_.x);
  CHECK(param_.output);
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(log_softmax, paddle::lite::operators::LogSoftmaxOp);
