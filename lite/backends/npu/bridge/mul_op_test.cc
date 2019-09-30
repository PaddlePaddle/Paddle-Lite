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

#include "lite/operators/mul_op.h"
#include <gtest/gtest.h>
#include "lite/backends/npu/bridge/registry.h"
#include "lite/backends/npu/bridge/test_helper.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace npu {
namespace bridge {

void mul_ref(const std::shared_ptr<operators::MulOpLite> op) {
  Scope* scope = op->scope();
  const OpInfo* op_info = op->op_info();
  auto x = scope->FindVar(op_info->Input("X").front())->GetMutable<Tensor>();
  auto y = scope->FindVar(op_info->Input("Y").front())->GetMutable<Tensor>();
  auto out =
      scope->FindVar(op_info->Output("Out").front())->GetMutable<Tensor>();
  int32_t x_num_col_dims = op_info->GetAttr<int32_t>("x_num_col_dims");
  int32_t y_num_col_dims = op_info->GetAttr<int32_t>("y_num_col_dims");
  auto x_data = x->mutable_data<float>();
  auto y_data = y->mutable_data<float>();
  auto out_data = out->mutable_data<float>();
  auto x_mat_dims = x->dims().Flatten2D(x_num_col_dims);
  auto y_mat_dims = y->dims().Flatten2D(y_num_col_dims);
  CHECK_EQ(x_mat_dims[1], y_mat_dims[0]);
  const int M = x_mat_dims[0];
  const int K = x_mat_dims[1];
  const int N = y_mat_dims[1];
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      out_data[m * N + n] = 0;
      for (int k = 0; k < K; ++k) {
        out_data[m * N + n] += x_data[m * K + k] * y_data[k * N + n];
      }
    }
  }
}

void test_mul(const std::vector<int64_t>& x_shape,
              const std::vector<int64_t>& y_shape,
              int x_num_col_dims,
              int y_num_col_dims) {
  const auto& bridges = lite::npu::bridge::Factory::Instance();
  const auto& supported_lists = bridges.AllFunctions();
  CHECK(bridges.HasType("mul"));

  Scope scope;
  std::string x_var_name("X");
  std::string y_var_name("Y");
  std::string out_var_name("Out");
  std::string out_ref_var_name("out_ref");
  auto* x = scope.Var(x_var_name)->GetMutable<Tensor>();
  auto* y = scope.Var(y_var_name)->GetMutable<Tensor>();
  auto* out = scope.Var(out_var_name)->GetMutable<Tensor>();
  auto* out_ref = scope.Var(out_ref_var_name)->GetMutable<Tensor>();
  x->Resize(x_shape);

  // get y shape
  auto x_mat_dims = x->dims().Flatten2D(x_num_col_dims);
  std::vector<int64_t> y_shape;
  for (int i = 0; i < y_num_col_dims - 1; i++) {
    y_shape.push_back(1);
  }
  y_shape.push_back(x_mat_dims[1]);
  y_shape.push_back(o);
  y->Resize(y_shape);

  FillTensor<float, int>(x);
  FillTensor<float, int>(y);

  // create mul op
  cpp::OpDesc mul_op_desc;
  mul_op_desc.SetType("mul");
  mul_op_desc.SetInput("X", {x_var_name});
  mul_op_desc.SetInput("Y", {y_var_name});
  mul_op_desc.SetOutput("Out", {out_var_name});
  mul_op_desc.SetAttr("x_num_col_dims", static_cast<int>(x_num_col_dims));
  mul_op_desc.SetAttr("y_num_col_dims", static_cast<int>(y_num_col_dims));

  auto mul_op = CreateOp<operators::MulOpLite>(mul_op_desc, &scope);
  LauchOp(mul_op, {x_var_name}, {out_var_name});
  out_ref->CopyDataFrom(*out);

  mul_ref(mul_op);

  // compare results
  auto* out_data = out->mutable_data<float>();
  auto* out_ref_data = out_ref->mutable_data<float>();
  for (int i = 0; i < out->dims().production(); i++) {
    EXPECT_NEAR(out_data[i], out_ref_data[i], 1e-5);
  }

  // model release
  npu::OpList::Global().clear();
  npu::DeviceInfo::Global().Clear();
}

TEST(NPUBridges, mul) {
  test_mul({1, 8, 8, 1}, {1, 8, 2, 2}, 2, 2);
  test_mul({1, 5, 5, 1}, {1, 5, 7, 7}, 2, 2);
  test_mul({1, 4, 1, 1}, {4, 8}, 1, 1);
}

}  // namespace bridge
}  // namespace npu
}  // namespace lite
}  // namespace paddle

USE_LITE_OP(mul);
USE_NPU_BRIDGE(mul);
