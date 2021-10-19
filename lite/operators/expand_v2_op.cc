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

#include "lite/operators/expand_v2_op.h"
#include "lite/core/op_registry.h"
namespace paddle {
namespace lite {
namespace operators {

bool ExpandV2OpLite::CheckShape() const {
  CHECK(param_.X);
  CHECK(param_.Out);
  return true;
}

bool ExpandV2OpLite::InferShapeImpl() const {
  std::vector<int> expand_shape;
  if (param_.Shape != nullptr) {
    auto shape_data = param_.Shape->template data<int>();
    for (int64_t i = 0; i < param_.Shape->numel(); i++) {
      expand_shape.push_back(shape_data[i]);
    }
  } else if (!param_.expand_shapes_tensor.empty()) {
    for (size_t i = 0; i < param_.expand_shapes_tensor.size(); i++) {
      expand_shape.push_back(
          param_.expand_shapes_tensor[i]->template data<int>()[0]);
    }
  } else {
    expand_shape = param_.shape;
  }

  std::vector<int64_t> x_shape = param_.X->dims().Vectorize();
  CHECK_GE(expand_shape.size(), x_shape.size());
  x_shape.insert(x_shape.begin(), expand_shape.size() - x_shape.size(), 1);
  for (size_t i = 0; i < expand_shape.size(); i++) {
    if (expand_shape[i] == -1) {
      expand_shape[i] = x_shape[i];
    }
    CHECK_GE(expand_shape[i], x_shape[i]);
  }

  std::vector<int64_t> out_shape(expand_shape.begin(), expand_shape.end());
  param_.Out->Resize(out_shape);
  return true;
}

bool ExpandV2OpLite::AttachImpl(const cpp::OpDesc& opdesc, lite::Scope* scope) {
  auto X_name = opdesc.Input("X").front();
  auto Out_name = opdesc.Output("Out").front();
  param_.X = GetVar<lite::Tensor>(scope, X_name);
  param_.Out = GetMutableVar<lite::Tensor>(scope, Out_name);

  if (opdesc.HasInput("Shape") && !opdesc.Input("Shape").empty()) {
    auto shape_tensor_name = opdesc.Input("Shape").front();
    param_.Shape = GetMutableVar<lite::Tensor>(scope, shape_tensor_name);
  }
  param_.expand_shapes_tensor.clear();  // Avoid errors caused by repeated calls
  if (opdesc.HasInput("expand_shapes_tensor") &&
      !opdesc.Input("expand_shapes_tensor").empty()) {
    for (auto expand_shapes_tensor_name :
         opdesc.Input("expand_shapes_tensor")) {
      param_.expand_shapes_tensor.push_back(
          GetMutableVar<lite::Tensor>(scope, expand_shapes_tensor_name));
    }
  }

  param_.shape = opdesc.GetAttr<std::vector<int>>("shape");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(expand_v2, paddle::lite::operators::ExpandV2OpLite);
