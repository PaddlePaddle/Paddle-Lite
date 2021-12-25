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

#include "lite/operators/where_op.h"
#include "lite/core/op_registry.h"
namespace paddle {
namespace lite {
namespace operators {

bool WhereOp::CheckShape() const {
  CHECK_OR_FALSE(param_.x);
  CHECK_OR_FALSE(param_.y);
  CHECK_OR_FALSE(param_.condition);
  CHECK_OR_FALSE(param_.out);
  return true;
}

bool WhereOp::InferShapeImpl() const {
  auto x_dims = param_.x->dims();
  auto y_dims = param_.y->dims();
  auto cond_dims = param_.condition->dims();
  CHECK_EQ(x_dims, y_dims)
      << "The dims of Inputs(X) and Inputs(Y) should be same. "
         "But received X's shape is "
      << x_dims << ", Y's shape is [%s]" << y_dims;
  CHECK_EQ(x_dims, cond_dims)
      << "The dims of Inputs(Condition) and Inputs(X) should be same. "
      << "But received Condition's shape is" << cond_dims << ", X's shape is "
      << x_dims;
  param_.out->Resize(x_dims);
  return true;
}

bool WhereOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  auto x = opdesc.Input("X").front();
  auto y = opdesc.Input("Y").front();
  auto condition = opdesc.Input("Condition").front();
  auto out = opdesc.Output("Out").front();
  param_.x = scope->FindVar(x)->GetMutable<lite::Tensor>();
  param_.y = scope->FindVar(y)->GetMutable<lite::Tensor>();
  param_.condition = scope->FindVar(condition)->GetMutable<lite::Tensor>();
  param_.out = scope->FindVar(out)->GetMutable<lite::Tensor>();
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(where, paddle::lite::operators::WhereOp);
