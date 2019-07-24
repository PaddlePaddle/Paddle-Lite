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

#include "lite/operators/normalize_op.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

// check输入是否符合规格
bool NormalizeOpLite::CheckShape() const {
  // lite::Tensor类型的检查
  CHECK_OR_FALSE(param_.X);
  // auto x_dims = param_.X->dims();
  /*  if (param_.has_scale) {
      CHECK_OR_FALSE(param_.scale);
      auto scale_dims = param_.scale->dims();
      CHECK_EQ(scale_dims.size(), 1UL) << "Input Scale must have 1 dimensions.";
    }
    if (param_.has_bias) {
      CHECK_OR_FALSE(param_.bias);
      auto bias_dims = param_.bias->dims();
      CHECK_EQ(bias_dims.size(), 1UL) << "Input Bias must have 1 dimensions.";
    }*/
  CHECK(param_.p == 1 || param_.p == 2) << "Input p must be 1 or 2.";
  return true;
}

bool NormalizeOpLite::InferShape() const {
  LOG(INFO) << "into infer shape_op;";
  param_.Out->Resize(param_.X->dims());
  return true;
}

// TODO(Superjomn) replace framework::OpDesc with a lite one.
bool NormalizeOpLite::AttachImpl(const cpp::OpDesc &op_desc,
                                 lite::Scope *scope) {
  LOG(INFO) << "into Attach IMPL OP";
  //////////////////////////
  ///////  lite::Tensor类型（输入输出类型）的写入。-----先写入输入类型
  param_.X = scope->FindVar(op_desc.Input("X").front())->GetMutable<Tensor>();
  //输出类型设置
  param_.Out =
      scope->FindVar(op_desc.Output("Out").front())->GetMutable<Tensor>();
  /*auto out = op_desc.Output("Out");
  for (auto var : out) {
    param_.Out.push_back(scope->FindVar(var)->GetMutable<lite::Tensor>());
  }*/
  // bool型变量输入
  param_.across_spatial = op_desc.GetAttr<bool>("across_spatial");
  // param_.has_scale = op_desc.GetAttr<bool>("has_scale");
  //  param_.channel_shared = op_desc.GetAttr<bool>("channel_shared");
  // param_.has_bias = op_desc.GetAttr<bool>("has_bias");
  // int型变量输入
  param_.p = op_desc.GetAttr<int>("p");
  //  param_.group = op_desc.GetAttr<int>("group");
  // float型变量输入
  param_.eps = op_desc.GetAttr<float>("eps");
  //根据类型写入变量
  /* if (param_.has_bias) {
     param_.bias =
         scope->FindVar(op_desc.Input("Bias").front())->GetMutable<Tensor>();
   }
   if (param_.has_scale) {
     param_.scale =
         scope->FindVar(op_desc.Input("Scale").front())->GetMutable<Tensor>();
   }*/
  LOG(INFO) << "acro:" << param_.across_spatial << "  P:" << param_.p
            << "  eps:" << param_.eps;
  return true;
  ////////////////////////////
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(normalize, paddle::lite::operators::NormalizeOpLite);
