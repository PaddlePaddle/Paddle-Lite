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
  CHECK_OR_FALSE(param_.Scale);
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Bias);
  CHECK_OR_FALSE(param_.Out);

  const auto scale_dims = param_.Scale->dims();
  const auto x_dims = param_.X->dims();
  CHECK_OR_FALSE(scale_dims[0] == x_dims[0] && scale_dims[1] == x_dims[1]);
  CHECK_OR_FALSE(x_dims == param_.Bias->dims());

  return true;
}

bool AxpyOpLite::InferShapeImpl() const {
  auto dims = param_.Bias->dims();

  // Set output dims
  param_.Out->Resize(lite::DDim(dims));
  return true;
}
// TODO(Superjomn) replace framework::OpDesc with a lite one.
bool AxpyOpLite::AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) {
  auto scale = op_desc.Input("Scale").front();
  auto x = op_desc.Input("X").front();
  auto bias = op_desc.Input("Bias").front();
  auto output = op_desc.Output("Out").front();

  param_.Scale = scope->FindVar(scale)->GetMutable<lite::Tensor>();
  param_.X = scope->FindVar(x)->GetMutable<lite::Tensor>();
  param_.Bias = scope->FindVar(bias)->GetMutable<lite::Tensor>();
  param_.Out = scope->FindVar(output)->GetMutable<lite::Tensor>();

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(axpy, paddle::lite::operators::AxpyOpLite);
