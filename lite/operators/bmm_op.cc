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

#include "lite/operators/bmm_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool BmmOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Y);
  CHECK_OR_FALSE(param_.Out);

  const auto x_dims = param_.X->dims();
  const auto y_dims = param_.Y->dims();

  CHECK_EQ(x_dims.size(), 3);
  CHECK_EQ(y_dims.size(), 3);
  CHECK_EQ(x_dims[0], y_dims[0]);
  CHECK_EQ(x_dims[2], y_dims[1]);
  return true;
}

bool BmmOpLite::InferShapeImpl() const {
  const auto x_dims = param_.X->dims();
  const auto y_dims = param_.Y->dims();
  DDim dim_out(std::vector<int64_t>({x_dims[0], x_dims[1], y_dims[2]}));
  param_.Out->Resize(dim_out);
  return true;
}

bool BmmOpLite::AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) {
  CHECK(!op_desc.Input("X").empty());
  CHECK(!op_desc.Input("Y").empty());
  CHECK(!op_desc.Output("Out").empty());

  auto X = op_desc.Input("X").front();
  auto Y = op_desc.Input("Y").front();
  auto Out = op_desc.Output("Out").front();

  param_.X = GetVar<lite::Tensor>(scope, X);
  param_.Y = GetVar<lite::Tensor>(scope, Y);
  param_.Out = GetMutableVar<lite::Tensor>(scope, Out);
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(bmm, paddle::lite::operators::BmmOpLite);
