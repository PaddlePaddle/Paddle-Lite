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

#include "lite/operators/matmul_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool MatMulOpLite::CheckShape() const {
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
  }
  return true;
}

bool MatMulOpLite::InferShapeImpl() const {
  const auto x_dims = param_.X->dims();
  const auto y_dims = param_.Y->dims();
  bool x_transpose = param_.transpose_X;
  bool y_transpose = param_.transpose_Y;
  std::vector<int64_t> dim_out_vec;

  if ((x_dims.size() >= 2 && y_dims.size() >= 2) &&
      (x_dims.size() != 2 || y_dims.size() != 2)) {
    // x: [B, ..., M, K], y: [B, ..., K, N], out: [B, ..., M, N]
    // x: [B, M, K], y: [K, N], out: [B, M, N]
    // or
    // x: [M, K], y: [B, ..., K, N], out: [B, ..., M, N]
    // x: [M, K], y: [B, K, N], out: [B, M, N]
    DDim dims = x_dims.size() >= y_dims.size() ? x_dims : y_dims;
    dim_out_vec.resize(dims.size());
    for (size_t i = 0; i < dims.size() - 2; ++i) {
      dim_out_vec[i] = dims[i];
    }
    if (!x_transpose && !y_transpose) {
      dim_out_vec[dims.size() - 2] = x_dims[x_dims.size() - 2];
      dim_out_vec[dims.size() - 1] = y_dims[y_dims.size() - 1];
    } else if (!x_transpose && y_transpose) {
      dim_out_vec[dims.size() - 2] = x_dims[x_dims.size() - 2];
      dim_out_vec[dims.size() - 1] = y_dims[y_dims.size() - 2];
    } else if (x_transpose && !y_transpose) {
      dim_out_vec[dims.size() - 2] = x_dims[x_dims.size() - 1];
      dim_out_vec[dims.size() - 1] = y_dims[y_dims.size() - 1];
    } else {
      dim_out_vec[dims.size() - 2] = x_dims[x_dims.size() - 1];
      dim_out_vec[dims.size() - 1] = y_dims[y_dims.size() - 2];
    }
  } else if (x_dims.size() == 2 && y_dims.size() == 2) {
    // x: [M, K], y: [K, N], out: [M, N]
    // x: [M, K], y: [K, N], out: [M, N]
    dim_out_vec.resize(x_dims.size());
    if (x_transpose) {
      dim_out_vec[0] = x_dims[1];
    } else {
      dim_out_vec[0] = x_dims[0];
    }
    if (y_transpose) {
      dim_out_vec[1] = y_dims[0];
    } else {
      dim_out_vec[1] = y_dims[1];
    }
  } else if (x_dims.size() >= 2 && y_dims.size() == 1) {
    // x: [B, M, K], y: [K], out: [B, M] or
    // x: [B, K], y: [K], out [B]
    dim_out_vec.resize(x_dims.size() - 1);
    for (size_t i = 0; i < dim_out_vec.size(); ++i) {
      dim_out_vec[i] = x_dims[i];
    }
  } else if (x_dims.size() == 1 && y_dims.size() == 1) {
    // x: [K], y: [K], out: [1]
    if (x_dims[0] == y_dims[0] && x_transpose == false &&
        y_transpose == false) {
      dim_out_vec.resize(1);
      dim_out_vec[0] = 1;
    } else if (x_transpose == true && y_transpose == true) {
      dim_out_vec.resize(2);
      dim_out_vec[0] = x_dims[0];
      dim_out_vec[1] = y_dims[0];
    } else {
      LOG(FATAL) << "not supported x_dims.(" << x_dims << ") and y_dims("
                 << y_dims << ")"
                 << ", and x_transpose: " << x_transpose
                 << ", y_transpose: " << y_transpose;
    }
  } else {
    LOG(FATAL) << "not supported x_dims(" << x_dims << ") and y_dims(" << y_dims
               << ")";
  }

  DDim dim_out(dim_out_vec);
  param_.Out->Resize(dim_out);

  return true;
}

bool MatMulOpLite::AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) {
  CHECK(!op_desc.Input("X").empty());
  CHECK(!op_desc.Input("Y").empty());
  CHECK(!op_desc.Output("Out").empty());

  auto X = op_desc.Input("X").front();
  auto Y = op_desc.Input("Y").front();
  auto Out = op_desc.Output("Out").front();

  param_.X = GetVar<lite::Tensor>(scope, X);
  param_.Y = GetVar<lite::Tensor>(scope, Y);
  param_.Out = GetMutableVar<lite::Tensor>(scope, Out);
  param_.transpose_X = op_desc.GetAttr<bool>("transpose_X");
  param_.transpose_Y = op_desc.GetAttr<bool>("transpose_Y");
  param_.alpha = op_desc.GetAttr<float>("alpha");
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

REGISTER_LITE_OP(matmul, paddle::lite::operators::MatMulOpLite);
