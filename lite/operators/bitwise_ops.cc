// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/operators/bitwise_ops.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool BitwiseOp::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  if (param_.bitwise_type_ != "bitwise_not") {
    CHECK_OR_FALSE(param_.Y);
  }
  CHECK_OR_FALSE(param_.Out);
  return true;
}

bool BitwiseOp::InferShapeImpl() const {
  if (param_.bitwise_type_ != "bitwise_not") {
    auto x_dim = param_.X->dims();
    auto y_dim = param_.Y->dims();
    if (x_dim == y_dim) {
      param_.Out->Resize(x_dim);
    } else {
      size_t max_dim =
          (x_dim.size() > y_dim.size() ? x_dim.size() : y_dim.size());
      int axis = param_.axis_;
      axis =
          (axis == -1 ? std::abs(static_cast<int>(x_dim.size() - y_dim.size()))
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
          out_dims_array[i] = 1;
        } else {
          out_dims_array[i] = (std::max)(x_dims_array[i], y_dims_array[i]);
        }
      }
      param_.Out->Resize(DDim(out_dims_array));
    }
  } else {
    lite::DDim x_dims = param_.X->dims();
    param_.Out->Resize(x_dims);
  }
  auto out_lod = param_.Out->mutable_lod();
  *out_lod = param_.X->lod();

  return true;
}

bool BitwiseOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  VLOG(4) << "opdesc.Type():" << opdesc.Type();
  auto X_name = opdesc.Input("X").front();
  auto Out_name = opdesc.Output("Out").front();

  param_.bitwise_type_ = opdesc.Type();
  param_.X = GetMutableVar<lite::Tensor>(scope, X_name);
  if (opdesc.Type() != "bitwise_not") {
    auto Y_name = opdesc.Input("Y").front();
    param_.Y = GetMutableVar<lite::Tensor>(scope, Y_name);
  }
  param_.Out = GetMutableVar<lite::Tensor>(scope, Out_name);

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

// Baisc activation ops
REGISTER_LITE_OP(bitwise_and, paddle::lite::operators::BitwiseOp);
REGISTER_LITE_OP(bitwise_or, paddle::lite::operators::BitwiseOp);
REGISTER_LITE_OP(bitwise_xor, paddle::lite::operators::BitwiseOp);
REGISTER_LITE_OP(bitwise_not, paddle::lite::operators::BitwiseOp);
