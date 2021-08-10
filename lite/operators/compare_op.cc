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

#include "lite/operators/compare_op.h"
#include <algorithm>
#include <cmath>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

static void GetBroadcastDimsArrays(const DDim &x_dims,
                                   const DDim &y_dims,
                                   int64_t *x_dims_array,
                                   int64_t *y_dims_array,
                                   int64_t *out_dims_array,
                                   int max_dim,
                                   int axis) {
  auto copy_data = [](const DDim &dims, int start, int end, int64_t *dest) {
    for (int i = start; i < end; i++) {
      dest[i] = dims[i];
    }
  };

  CHECK_GE(axis, 0);
  CHECK_LT(axis, max_dim);
  if (x_dims.size() > y_dims.size()) {
    std::fill(y_dims_array, y_dims_array + axis, 1);
    if (axis + y_dims.size() < max_dim) {
      std::fill(y_dims_array + axis + y_dims.size(), y_dims_array + max_dim, 1);
    }
    copy_data(x_dims, 0, x_dims.size(), x_dims_array);
    copy_data(y_dims, 0, y_dims.size(), y_dims_array + axis);
  } else {
    std::fill(x_dims_array, x_dims_array + axis, 1);
    if (axis + x_dims.size() < max_dim) {
      std::fill(x_dims_array + axis + x_dims.size(), x_dims_array + max_dim, 1);
    }
    copy_data(x_dims, 0, x_dims.size(), x_dims_array + axis);
    copy_data(y_dims, 0, y_dims.size(), y_dims_array);
  }

  for (int i = 0; i < max_dim; i++) {
    CHECK(x_dims_array[i] == y_dims_array[i] || x_dims_array[i] <= 1 ||
          y_dims_array[i] <= 1);
    if ((x_dims_array[i] > 1 || y_dims_array[i] > 1) ||
        (x_dims_array[i] == 1 && y_dims_array[i] == 1)) {
      out_dims_array[i] = (std::max)(x_dims_array[i], y_dims_array[i]);
    } else {
      out_dims_array[i] = -1;
    }
  }
}

bool CompareOp::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Y);
  CHECK_OR_FALSE(param_.Out);
  return true;
}

bool CompareOp::InferShapeImpl() const {
  CHECK_OR_FALSE(param_.Out);
  auto dim_x = param_.X->dims();
  auto dim_y = param_.Y->dims();
  if (dim_x == dim_y) {
    param_.Out->Resize(dim_x);
  } else {
    int max_dim = (std::max)(dim_x.size(), dim_y.size());
    int axis = std::abs(static_cast<int>(dim_x.size() - dim_y.size()));
    std::vector<int64_t> x_dims_array(max_dim);
    std::vector<int64_t> y_dims_array(max_dim);
    std::vector<int64_t> out_dims_array(max_dim);
    GetBroadcastDimsArrays(dim_x,
                           dim_y,
                           x_dims_array.data(),
                           y_dims_array.data(),
                           out_dims_array.data(),
                           max_dim,
                           axis);
    param_.Out->Resize(out_dims_array);
  }
  param_.Out->set_lod(param_.X->lod());
  return true;
}

bool CompareOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  param_.X =
      scope->FindVar(opdesc.Input("X").front())->GetMutable<lite::Tensor>();
  param_.Y =
      scope->FindVar(opdesc.Input("Y").front())->GetMutable<lite::Tensor>();
  param_.axis = opdesc.GetAttr<int>("axis");
  param_.force_cpu = opdesc.GetAttr<bool>("force_cpu");
  if (opdesc.HasAttr("fuse_greater_than")) {
    param_.fuse_greater_than = opdesc.GetAttr<bool>("fuse_greater_than");
  }
  param_.Out =
      scope->FindVar(opdesc.Output("Out").front())->GetMutable<lite::Tensor>();
  CHECK(param_.X);
  CHECK(param_.Y);
  CHECK(param_.Out);
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(equal, paddle::lite::operators::CompareOp);
REGISTER_LITE_OP(not_equal, paddle::lite::operators::CompareOp);
REGISTER_LITE_OP(less_than, paddle::lite::operators::CompareOp);
REGISTER_LITE_OP(less_equal, paddle::lite::operators::CompareOp);
REGISTER_LITE_OP(greater_than, paddle::lite::operators::CompareOp);
REGISTER_LITE_OP(greater_equal, paddle::lite::operators::CompareOp);
