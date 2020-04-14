// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/operators/mul_grad_op.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace operators {

bool MulGradOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.x);
  CHECK_OR_FALSE(param_.y);
  CHECK_OR_FALSE(param_.output_grad);
  CHECK_OR_FALSE(param_.x_grad || param_.y_grad);
  CHECK_OR_FALSE(param_.x_num_col_dims);
  CHECK_OR_FALSE(param_.y_num_col_dims);

  const auto x_dims = param_.x->dims();
  const auto y_dims = param_.y->dims();
  const auto out_dims = param_.output_grad->dims();

  CHECK_GT_OR_FALSE(x_dims.size(), static_cast<size_t>(param_.x_num_col_dims));
  CHECK_GT_OR_FALSE(y_dims.size(), static_cast<size_t>(param_.y_num_col_dims));

  auto x_flatten_dims = flatten_2d(x_dims, param_.x_num_col_dims);
  auto y_flatten_dims = flatten_2d(y_dims, param_.y_num_col_dims);
  auto out_flatten_dims = flatten_2d(out_dims, param_.x_num_col_dims);

  // Out = X * Y;
  CHECK_EQ_OR_FALSE(x_flatten_dims[1], y_flatten_dims[0]);
  CHECK_EQ_OR_FALSE(x_flatten_dims[0], out_flatten_dims[0]);
  CHECK_EQ_OR_FALSE(y_flatten_dims[1], out_flatten_dims[1]);
  return true;
}

bool MulGradOpLite::InferShapeImpl() const {
  const auto x_dims = param_.x->dims();
  const auto y_dims = param_.y->dims();
  if (param_.x_grad) {
    param_.x_grad->Resize(x_dims);
    param_.x_grad->set_lod(param_.x->lod());
  }
  if (param_.y_grad) {
    param_.y_grad->Resize(y_dims);
    param_.y_grad->set_lod(param_.y->lod());
  }
}

bool MulGradOpLite::AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) {
  CHECK(!op_desc.Input("X").empty());
  CHECK(!op_desc.Input("Y").empty());
  CHECK(!op_desc.Input("Out@GRAD").empty());
  CHECK(!op_desc.Output("X@GRAD").empty() || !op_desc.Output("Y@GRAD").empty())
      << "at least one of 'X@GRAD' and 'Y@GRAD' is not empty";

  auto *x_var = scope->FindVar(op_desc.Input("X").front());
  CHECK(x_var);
  param_.x = &x_var->Get<Tensor>();

  auto *y_var = scope->FindVar(op_desc.Input("Y").front());
  CHECK(y_var);
  param_.y = &y_var->Get<Tensor>();

  auto *out_grad_var = scope->FindVar(op_desc.Input("Out@GRAD").front());
  CHECK(out_grad_var);
  param_.output_grad = &out_grad_var->Get<Tensor>();

  if (!op_desc.Output("X@GRAD").empty()) {
    auto *x_grad_var = scope->FindVar(op_desc.Output("X@GRAD").front());
    CHECK(x_grad_var);
    param_.x_grad = x_grad_var->GetMutable<Tensor>();
  }

  if (!op_desc.Output("Y@GRAD").empty()) {
    auto *y_grad_var = scope->FindVar(op_desc.Output("Y@GRAD").front());
    CHECK(y_grad_var);
    param_.y_grad = y_grad_var->GetMutable<Tensor>();
  }
  param_.x_num_col_dims = op_desc.GetAttr<int>("x_num_col_dims");
  param_.y_num_col_dims = op_desc.GetAttr<int>("y_num_col_dims");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(mul_grad, paddle::lite::operators::MulGradOpLite);
