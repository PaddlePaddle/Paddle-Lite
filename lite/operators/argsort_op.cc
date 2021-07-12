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

#include "lite/operators/argsort_op.h"

#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool ArgsortOpLite::CheckShape() const {
  CHECK(param_.X);
  CHECK(param_.Out);
  CHECK(param_.Indices);

  auto in_dims = param_.X->dims();
  int axis = param_.axis;

  int num_dims = static_cast<int>(in_dims.size());
  CHECK_GE(axis, -num_dims) << "axis'(" << axis
                            << ") must be greater than or equal to - num_dims("
                            << -num_dims << ").";

  CHECK_LT(axis, num_dims) << "axis'(" << axis
                           << ") must be less than num_dims(" << num_dims
                           << ").";

  return true;
}

bool ArgsortOpLite::InferShapeImpl() const {
  auto in_dims = param_.X->dims();
  param_.Out->Resize(in_dims);
  param_.Indices->Resize(in_dims);

  param_.Out->set_lod(param_.X->lod());
  param_.Indices->set_lod(param_.X->lod());
  return true;
}

bool ArgsortOpLite::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  param_.X = scope->FindTensor(opdesc.Input("X").front());
  param_.Out = scope->FindMutableTensor(opdesc.Output("Out").front());
  param_.Indices = scope->FindMutableTensor(opdesc.Output("Indices").front());
  if (opdesc.HasAttr("axis")) {
    param_.axis = opdesc.GetAttr<int>("axis");
  }

  if (opdesc.HasAttr("descending")) {
    param_.descending = opdesc.GetAttr<bool>("descending");
  }
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(argsort, paddle::lite::operators::ArgsortOpLite);
