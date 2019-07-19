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

#include "lite/operators/axpy_op.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool AxpyOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.scale);
  CHECK_OR_FALSE(param_.x);
  CHECK_OR_FALSE(param_.bias);
  CHECK_OR_FALSE(param_.output);

  const auto scale_dims = param_.scale->dims();
  const auto x_dims = param_.x->dims();
  CHECK_OR_FALSE(scale_dims[0] == x_dims[0] && scale_dims[1] == x_dims[1]);
  CHECK_OR_FALSE(x_dims == param_.output->dims());
  CHECK_OR_FALSE(x_dims = param_.bias->dims());

  return true;
}

bool AxpyOpLite::InferShape() const {
  auto dims = param_.bias->dims();

  // Set output dims
  param_.output->Resize(lite::DDim(dims));
  return true;
}
// TODO(Superjomn) replace framework::OpDesc with a lite one.
bool AxpyOpLite::AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) {
  auto scale = op_desc.Input("Scale").front();
  auto x = op_desc.Input("X").front();
  auto bias = op_desc.Input("Bias").front();
  auto output = op_desc.Output("Out").front();

  param_.scale = scope->FindVar(scale)->GetMutable<lite::Tensor>();
  param_.x = scope->FindVar(x)->GetMutable<lite::Tensor>();
  param_.bias = scope->FindVar(bias)->GetMutable<lite::Tensor>();
  param_.output = scope->FindVar(output)->GetMutable<lite::Tensor>();

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(axpy, paddle::lite::operators::AxpyOpLite);
