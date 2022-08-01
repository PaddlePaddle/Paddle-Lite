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

#include "lite/operators/matmul_v2_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool MatMulV2OpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Y);
  CHECK_OR_FALSE(param_.Out);

  const auto x_dims = param_.X->dims();
  const auto y_dims = param_.Y->dims();
  bool x_transpose = param_.transpose_X;
  bool y_transpose = param_.transpose_Y;

  if (x_dims.size() > 1 && y_dims.size() > 1) {
    if (!x_transpose && !y_transpose) {
      CHECK_EQ(x_dims[x_dims.size() - 1], y_dims[y_dims.size() - 2])
          << "not supported x_dims(" << x_dims << ") and y_dims(" << y_dims
          << ")";
    } else if (!x_transpose && y_transpose) {
      CHECK_EQ(x_dims[x_dims.size() - 1], y_dims[y_dims.size() - 1])
          << "not supported x_dims(" << x_dims << ") and y_dims(" << y_dims
          << ")";
    } else if (x_transpose && !y_transpose) {
      CHECK_EQ(x_dims[x_dims.size() - 2], y_dims[y_dims.size() - 2])
          << "not supported x_dims(" << x_dims << ") and y_dims(" << y_dims
          << ")";
    } else {
      CHECK_EQ(x_dims[x_dims.size() - 2], y_dims[y_dims.size() - 1])
          << "not supported x_dims(" << x_dims << ") and y_dims(" << y_dims
          << ")";
    }
  } else if (x_dims.size() > 2 && y_dims.size() == 1) {
    CHECK_EQ(x_dims[x_dims.size() - 1], y_dims[0])
        << "not supported x_dims(" << x_dims << ") and y_dims(" << y_dims
        << ")";
  } else if (y_dims.size() > 2 && x_dims.size() == 1) {
    CHECK_EQ(x_dims[y_dims.size() - 1], x_dims[0])
        << "not supported x_dims(" << x_dims << ") and y_dims(" << y_dims
        << ")";
  } else if (x_dims.size() == 1 && y_dims.size() == 1) {
    CHECK_EQ(x_dims[0], y_dims[0]) << "not supported x_dims(" << x_dims
                                   << ") and y_dims(" << y_dims << ")";
  }
  return true;
}

bool MatMulV2OpLite::InferShapeImpl() const {
  const auto x_dims = param_.X->dims();
  const auto y_dims = param_.Y->dims();
  bool trans_x = param_.transpose_X;
  bool trans_y = param_.transpose_Y;
  std::vector<int64_t> dim_out_vec;
  bool x_broadcasted = false;
  bool y_broadcasted = false;
  std::vector<int64_t> dims_x = x_dims.data();
  std::vector<int64_t> dims_y = y_dims.data();
  int ndims_x = dims_x.size();
  int ndims_y = dims_y.size();
  if (ndims_x == 1) {
    dims_x.insert(dims_x.begin(), 1);
    ndims_x = 2;
    x_broadcasted = true;
  }
  if (ndims_y == 1) {
    dims_y.push_back(1);
    ndims_y = 2;
    y_broadcasted = true;
  }
  int64_t M = 1;
  int64_t N = 1;
  if (trans_x) {
    M = dims_x[ndims_x - 1];
  } else {
    M = dims_x[ndims_x - 2];
  }

  if (trans_y) {
    N = dims_y[ndims_y - 2];
  } else {
    N = dims_y[ndims_y - 1];
  }
  std::vector<int64_t> new_dims;
  if (ndims_x >= ndims_y) {
    dim_out_vec.assign(dims_x.begin(), dims_x.end() - 2);
  } else {
    dim_out_vec.assign(dims_y.begin(), dims_y.end() - 2);
  }
  if (!x_broadcasted) {
    dim_out_vec.push_back(M);
  }
  if (!y_broadcasted) {
    dim_out_vec.push_back(N);
  }
  if (x_broadcasted && y_broadcasted) {
    dim_out_vec.push_back(1);
  }
  DDim dim_out(dim_out_vec);
  param_.Out->Resize(dim_out);

  return true;
}

bool MatMulV2OpLite::AttachImpl(const cpp::OpDesc &op_desc,
                                lite::Scope *scope) {
  CHECK(!op_desc.Input("X").empty());
  CHECK(!op_desc.Input("Y").empty());
  CHECK(!op_desc.Output("Out").empty());

  auto X = op_desc.Input("X").front();
  auto Y = op_desc.Input("Y").front();
  auto Out = op_desc.Output("Out").front();

  param_.X = GetVar<lite::Tensor>(scope, X);
  param_.Y = GetVar<lite::Tensor>(scope, Y);
  param_.Out = GetMutableVar<lite::Tensor>(scope, Out);
  param_.transpose_X = op_desc.GetAttr<bool>("trans_x");
  param_.transpose_Y = op_desc.GetAttr<bool>("trans_y");
  if (op_desc.HasAttr("alpha")) {
    param_.alpha = op_desc.GetAttr<float>("alpha");
  }
  input_tensor_ptrs_cache_.push_back(param_.X);
  input_tensor_ptrs_cache_.push_back(param_.Y);
  output_tensor_ptrs_cache_.push_back(param_.Out);
  const OpInfo *op_info = static_cast<const OpInfo *>(&op_desc);
  if (op_info != nullptr && op_info->HasAttr("enable_int8")) {
    param_.enable_int8 = op_info->GetAttr<bool>("enable_int8");
    auto input_scale_name = "X0_scale";
    auto weight_scale_name = "Y0_scale";
    auto out_scale_name = "Out0_scale";
    if (op_info->HasInputScale(input_scale_name, true))
      param_.input_scale = op_info->GetInputScale(input_scale_name, true)[0];
    if (op_info->HasInputScale(weight_scale_name, true))
      param_.weight_scale = op_info->GetInputScale(weight_scale_name, true);
    if (op_info->HasOutputScale(out_scale_name, true))
      param_.output_scale = op_info->GetOutputScale(out_scale_name, true)[0];
  }
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(matmul_v2, paddle::lite::operators::MatMulV2OpLite);
