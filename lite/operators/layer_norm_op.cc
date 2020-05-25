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

#include "lite/operators/layer_norm_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool LayerNormOp::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Y);
  CHECK_OR_FALSE(param_.Mean);
  CHECK_OR_FALSE(param_.Variance);
  return true;
}

bool LayerNormOp::InferShapeImpl() const {
  auto out_dims = param_.X->dims();
  param_.Y->Resize(out_dims);
  auto inner_size = out_dims.Flatten2D(param_.begin_norm_axis)[0];
  param_.Mean->Resize(std::vector<int64_t>({inner_size}));
  param_.Variance->Resize(std::vector<int64_t>({inner_size}));

  auto out_lod = param_.Y->mutable_lod();
  *out_lod = param_.X->lod();
  return true;
}

bool LayerNormOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  param_.X =
      scope->FindVar(opdesc.Input("X").front())->GetMutable<lite::Tensor>();
  param_.Y =
      scope->FindVar(opdesc.Output("Y").front())->GetMutable<lite::Tensor>();
  param_.Mean =
      scope->FindVar(opdesc.Output("Mean").front())->GetMutable<lite::Tensor>();
  param_.Variance = scope->FindVar(opdesc.Output("Variance").front())
                        ->GetMutable<lite::Tensor>();
  CHECK(param_.X);
  CHECK(param_.Y);
  CHECK(param_.Mean);
  CHECK(param_.Variance);
  if (opdesc.HasInput("Scale")) {
    param_.Scale = scope->FindVar(opdesc.Input("Scale").front())
                       ->GetMutable<lite::Tensor>();
  }
  if (opdesc.HasInput("Bias")) {
    param_.Bias = scope->FindVar(opdesc.Input("Bias").front())
                      ->GetMutable<lite::Tensor>();
  }
  param_.begin_norm_axis = opdesc.GetAttr<int>("begin_norm_axis");
  param_.epsilon = opdesc.GetAttr<float>("epsilon");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(layer_norm, paddle::lite::operators::LayerNormOp);
