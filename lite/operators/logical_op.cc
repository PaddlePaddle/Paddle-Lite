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

#include "lite/operators/logical_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool BinaryLogicalOp::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Y);
  CHECK_OR_FALSE(param_.Out);
  return true;
}

bool BinaryLogicalOp::InferShapeImpl() const {
  CHECK_OR_FALSE(param_.Out);
  // TODO(Superjomn) Enable data sharing.
  auto input_dims = param_.X->dims();
  param_.Out->Resize(input_dims);
  return true;
}

bool BinaryLogicalOp::AttachImpl(const cpp::OpDesc &opdesc,
                                 lite::Scope *scope) {
  param_.X =
      scope->FindVar(opdesc.Input("X").front())->GetMutable<lite::Tensor>();
  param_.Y =
      scope->FindVar(opdesc.Input("Y").front())->GetMutable<lite::Tensor>();
  param_.Out =
      scope->FindVar(opdesc.Output("Out").front())->GetMutable<lite::Tensor>();
  CHECK(param_.X);
  CHECK(param_.Y);
  CHECK(param_.Out);
  return true;
}
bool UnaryLogicalOp::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Out);
  return true;
}

bool UnaryLogicalOp::InferShapeImpl() const {
  CHECK_OR_FALSE(param_.Out);
  // TODO(Superjomn) Enable data sharing.
  auto input_dims = param_.X->dims();
  param_.Out->Resize(input_dims);
  return true;
}

bool UnaryLogicalOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  param_.X =
      scope->FindVar(opdesc.Input("X").front())->GetMutable<lite::Tensor>();
  param_.Out =
      scope->FindVar(opdesc.Output("Out").front())->GetMutable<lite::Tensor>();
  CHECK(param_.X);
  CHECK(param_.Out);
  return true;
}
}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(logical_xor, paddle::lite::operators::BinaryLogicalOp);
REGISTER_LITE_OP(logical_and, paddle::lite::operators::BinaryLogicalOp);
REGISTER_LITE_OP(logical_or, paddle::lite::operators::BinaryLogicalOp);
REGISTER_LITE_OP(logical_not, paddle::lite::operators::UnaryLogicalOp);
