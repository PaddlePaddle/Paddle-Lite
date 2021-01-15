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
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Out);

  int x_dims_size = param_.X->dims().size();
  CHECK_GE(x_dims_size, 1u)
      << "The rank of Input(X) must be greater than or equal to 1.";
  CHECK_LE(x_dims_size, 6u)
      << "The rank of Input(X) must not be greater than 6.";
  std::vector<int> expand_shape;
  if (param_.Shape != nullptr) {
    auto Shape_data = param_.Shape->template data<int>();
    for (int64_t i = 0; i < param_.Shape->numel(); i++) {
      expand_shape.push_back(Shape_data[i]);
    }
  } else if (!param_.expand_shapes_tensor.empty()) {
    for (size_t i = 0; i < param_.expand_shapes_tensor.size(); i++) {
      expand_shape.push_back(
          param_.expand_shapes_tensor[i]->template data<int>()[0]);
    }
  } else {
    expand_shape = param_.shape;
  }
  auto shape_size = expand_shape.size();
  CHECK_GE(shape_size, x_dims_size) << "The size of shape for expand_v2 op "
                                       "must be greater than or equal to the "
                                       "size of the input.";
  const auto* x = param_.X;
  DDim in_shape = x->dims();
  std::vector<int64_t> vec_in_dims;
  for (int i = 0; i < in_shape.size(); ++i) vec_in_dims.push_back(in_shape[i]);
  auto diff = expand_shape.size() - vec_in_dims.size();
  vec_in_dims.insert(vec_in_dims.begin(), diff, 1);
  repeat_times_.resize(vec_in_dims.size());
  new_in_dims_.ConstructFrom(vec_in_dims);
  for (size_t i = 0; i < vec_in_dims.size(); ++i) {
    CHECK_NE(expand_shape[i], 0) << "The expanded size cannot be zero.";
    if (i < diff) {
      CHECK_GT(expand_shape[i], 0) << "The expanded size for non-existing "
                                      "dimensions must be positive for "
                                      "expand_v2 op.";
      repeat_times_[i] = expand_shape[i];
    } else if (expand_shape[i] > 0) {
      if (vec_in_dims[i] != 1) {
        CHECK_EQ(vec_in_dims[i], expand_shape[i])
            << "The value of the non-singleton dimension must match the "
               "corresponding value in shape for expand_v2 op.";
        repeat_times_[i] = 1;
      } else {
        repeat_times_[i] = expand_shape[i];
      }
    } else {
      CHECK_EQ(expand_shape[i], -1) << "When the value in shape is negative "
                                       "for expand_v2 op, only -1 is supported";
      repeat_times_[i] = 1;
    }
  }
  return true;
}

bool ExpandV2OpLite::InferShapeImpl() const {
  DDim out_dims(new_in_dims_);
  for (size_t i = 0; i < repeat_times_.size(); ++i) {
    out_dims[i] *= repeat_times_[i];
  }
  param_.Out->Resize(out_dims);

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
