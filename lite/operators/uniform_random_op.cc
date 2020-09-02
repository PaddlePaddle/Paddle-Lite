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

#include "lite/operators/uniform_random_op.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool UniformRandomOpLite::CheckShape() const { return true; }

bool UniformRandomOpLite::InferShapeImpl() const {
  if (param_.X) {
    if (param_.X->precision() == PrecisionType::kInt64){
      auto* new_data = param_.X->data<int64_t>();
      std::vector<int64_t> new_shape(new_data, new_data + param_.X->numel());
      param_.Out->Resize(new_shape);
    } else if (param_.X->precision() == PrecisionType::kInt32) {
      std::vector<int64_t> new_shape;
      auto* new_data = param_.X->data<int32_t>();
      for (int i = 0; i < param_.X->numel(); ++i) {
        new_shape.push_back(static_cast<int64_t>(*(new_data + i)));
      }
      param_.Out->Resize(new_shape);
    } else {
      LOG(ERROR) << "The dtype of shape tensor must be int32 or int64.";
    }
  } else {
    auto new_shape = param_.shape;
    param_.Out->Resize(new_shape);
  }
  return true;
}

bool UniformRandomOpLite::AttachImpl(const cpp::OpDesc& opdesc,
                                     lite::Scope* scope) {
  param_.shape = opdesc.GetAttr<std::vector<int64_t>>("shape");
  param_.min = opdesc.GetAttr<float>("min");
  param_.max = opdesc.GetAttr<float>("max");
  param_.seed = opdesc.GetAttr<int>("seed");
  param_.dtype = opdesc.GetAttr<int>("dtype");
  if (opdesc.HasInput("ShapeTensor")) {
    auto X = opdesc.Input("ShapeTensor").front();
    param_.X = scope->FindVar(X)->GetMutable<lite::Tensor>();
  }
  param_.Out = GetMutableVar<Tensor>(scope, opdesc.Output("Out").front());
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(uniform_random, paddle::lite::operators::UniformRandomOpLite);
