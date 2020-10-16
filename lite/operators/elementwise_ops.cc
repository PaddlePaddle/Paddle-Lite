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

#include "lite/operators/elementwise_ops.h"
#include <algorithm>
#include <cmath>
#include "lite/core/op_registry.h"
namespace paddle {
namespace lite {
namespace operators {

bool ElementwiseOp::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Y);
  CHECK_OR_FALSE(param_.Out);
  return true;
}

bool ElementwiseOp::InferShapeImpl() const {
  auto x_dim = param_.X->dims();
  auto y_dim = param_.Y->dims();
  if (x_dim == y_dim) {
    param_.Out->Resize(x_dim);
    auto out_lod = param_.Out->mutable_lod();
    *out_lod = param_.X->lod();
  } else {
    size_t max_dim =
        (x_dim.size() > y_dim.size() ? x_dim.size() : y_dim.size());
    int axis = param_.axis;
    axis = (axis == -1 ? std::abs(static_cast<int>(x_dim.size() - y_dim.size()))
                       : axis);
    std::vector<int64_t> x_dims_array(max_dim);
    std::vector<int64_t> y_dims_array(max_dim);
    std::vector<int64_t> out_dims_array(max_dim);

    if (x_dim.size() > y_dim.size()) {
      for (int i = 0; i < axis; ++i) {
        y_dims_array[i] = 1;
      }
      if (axis + y_dim.size() < max_dim) {
        for (size_t i = axis + y_dim.size(); i < max_dim; ++i) {
          y_dims_array[i] = 1;
        }
      }
      x_dims_array = x_dim.Vectorize();
      for (size_t i = 0; i < y_dim.size(); ++i) {
        y_dims_array[i + axis] = y_dim[i];
      }
    } else {
      for (int i = 0; i < axis; ++i) {
        x_dims_array[i] = 1;
      }
      if (axis + x_dim.size() < max_dim) {
        for (size_t i = axis + x_dim.size(); i < max_dim; ++i) {
          x_dims_array[i] = 1;
        }
      }
      y_dims_array = y_dim.Vectorize();
      for (size_t i = 0; i < x_dim.size(); ++i) {
        x_dims_array[i + axis] = x_dim[i];
      }
    }
    for (size_t i = 0; i < max_dim; i++) {
      if (x_dims_array[i] == -1 || y_dims_array[i] == -1) {
        out_dims_array[i] = -1;
      } else {
        out_dims_array[i] = (std::max)(x_dims_array[i], y_dims_array[i]);
      }
    }
    param_.Out->Resize(DDim(out_dims_array));
    auto out_lod = param_.Out->mutable_lod();
    *out_lod = param_.X->lod();
  }

  return true;
}

bool ElementwiseOp::AttachImpl(const cpp::OpDesc& opdesc, lite::Scope* scope) {
  AttachParam(&param_);

  auto X_name = opdesc.Input("X").front();
  auto Y_name = opdesc.Input("Y").front();
  auto Out_name = opdesc.Output("Out").front();

  param_.X = GetVar<lite::Tensor>(scope, X_name);
  param_.Y = GetVar<lite::Tensor>(scope, Y_name);
  param_.Out = GetMutableVar<lite::Tensor>(scope, Out_name);
  param_.axis = opdesc.GetAttr<int>("axis");
  return true;
}

// #ifdef LITE_WITH_TRAIN
// bool ElementwiseGradExplicitOp::CheckShape() const {
//  CHECK_OR_FALSE(param_.Y);
//  CHECK_OR_FALSE(param_.X_grad);
//  CHECK_OR_FALSE(param_.Out_grad);
//  return true;
//}

// bool ElementwiseGradExplicitOp::InferShapeImpl() const {
//   param_.X_grad->Resize(param_.Out_grad->dims());
//   if (param_.Y_grad) param_.Y_grad->Resize(param_.Y->dims());
//   return true;
// }

// bool ElementwiseGradExplicitOp::AttachImpl(const cpp::OpDesc& opdesc,
//                                            lite::Scope* scope) {
//   CHECK_EQ(opdesc.InputArgumentNames().size(), 2UL);
//   auto Y_name = opdesc.Input("Y").front();
//   auto Out_name = opdesc.Input(framework::GradVarName("Out")).front();
//   auto X_grad = opdesc.Output(framework::GradVarName("X")).front();

//   if (opdesc.Output(framework::GradVarName("Y")).size() > 0) {
//     auto Y_grad = opdesc.Output(framework::GradVarName("Y")).front();
//     param_.Y_grad = GetMutableVar<Tensor>(scope, Y_grad);
//   }
//   param_.Y = GetVar<lite::Tensor>(scope, Y_name);
//   param_.Out_grad = GetVar<lite::Tensor>(scope, Out_name);
//   param_.X_grad = GetMutableVar<lite::Tensor>(scope, X_grad);
//   param_.axis = opdesc.GetAttr<int>("axis");

//   return true;
// }
// #endif

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(elementwise_sub, paddle::lite::operators::ElementwiseOp);
REGISTER_LITE_OP(elementwise_add, paddle::lite::operators::ElementwiseOp);

REGISTER_LITE_OP(elementwise_mul, paddle::lite::operators::ElementwiseOp);
REGISTER_LITE_OP(elementwise_max, paddle::lite::operators::ElementwiseOp);
REGISTER_LITE_OP(elementwise_div, paddle::lite::operators::ElementwiseOp);
REGISTER_LITE_OP(elementwise_mod, paddle::lite::operators::ElementwiseOp);
REGISTER_LITE_OP(elementwise_pow, paddle::lite::operators::ElementwiseOp);

// #ifdef LITE_WITH_TRAIN
// REGISTER_LITE_OP(elementwise_sub_grad,
//                  paddle::lite::operators::ElementwiseGradExplicitOp);
// REGISTER_LITE_OP(elementwise_add_grad,
//                  paddle::lite::operators::ElementwiseGradExplicitOp);
// #endif
