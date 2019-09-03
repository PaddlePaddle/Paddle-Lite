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

#include "lite/operators/fc_op.h"
#include <gtest/gtest.h>
#include "lite/backends/npu/bridge/registry.h"
#include "lite/backends/npu/bridge/test_helper.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace npu {
namespace bridge {

void fc_ref(const std::shared_ptr<operators::FcOpLite> op) {
  Scope* scope = op->scope();
  const OpInfo* op_info = op->op_info();
  auto input =
      scope->FindVar(op_info->Input("Input").front())->GetMutable<Tensor>();
  auto w = scope->FindVar(op_info->Input("W").front())->GetMutable<Tensor>();
  auto out =
      scope->FindVar(op_info->Output("Out").front())->GetMutable<Tensor>();
  int32_t in_num_col_dims = op_info->GetAttr<int32_t>("in_num_col_dims");
  Tensor* bias = nullptr;
  float* bias_data = nullptr;
  if (op_info->HasInput("Bias")) {
    auto bias_var_names = op_info->Input("Bias");
    if (bias_var_names.size() > 0) {
      auto bias_var_name = bias_var_names.front();
      bias = scope->FindVar(bias_var_name)->GetMutable<lite::Tensor>();
      bias_data = bias->mutable_data<float>();
    }
  }
  auto input_data = input->data<float>();
  auto w_data = w->mutable_data<float>();
  auto out_data = out->mutable_data<float>();
  auto in_mat_dims = input->dims().Flatten2D(in_num_col_dims);
  int out_num_classes = w->dims()[1];
  const int M = in_mat_dims[0];
  const int K = in_mat_dims[1];
  const int N = out_num_classes;
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      out_data[m * N + n] = 0;
      for (int k = 0; k < K; ++k) {
        out_data[m * N + n] += input_data[m * K + k] * w_data[k * N + n];
      }
    }
  }
  if (bias_data != nullptr) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        out_data[m * N + n] += bias_data[n];
      }
    }
  }
}

void test_fc(const std::vector<int64_t>& x_shape,
             const std::vector<int64_t>& w_shape,
             int in_num_col_dims,
             bool has_bias) {
  CHECK_EQ(w_shape.size(), 2UL);

  const auto& bridges = lite::npu::bridge::Factory::Instance();
  const auto& supported_lists = bridges.AllFunctions();
  CHECK(bridges.HasType("fc"));

  Scope scope;
  std::string x_var_name("Input");
  std::string w_var_name("W");
  std::string bias_var_name("Bias");
  std::string out_var_name("Out");
  std::string out_ref_var_name("out_ref");
  auto* x = scope.Var(x_var_name)->GetMutable<Tensor>();
  auto* w = scope.Var(w_var_name)->GetMutable<Tensor>();
  auto* out = scope.Var(out_var_name)->GetMutable<Tensor>();
  auto* out_ref = scope.Var(out_ref_var_name)->GetMutable<Tensor>();
  x->Resize(x_shape);
  input->Resize({bs, ic, ih, iw});

  // get w shape
  auto in_mat_dims = input->dims().Flatten2D(in_num_col_dims);
  std::vector<int64_t> w_shape = {in_mat_dims[1], out_num_classes};
  w->Resize(w_shape);

  FillTensor<float, int>(x);
  FillTensor<float, int>(w);

  // create fc op
  cpp::OpDesc fc_op_desc;
  fc_op_desc.SetType("fc");
  fc_op_desc.SetInput("Input", {x_var_name});
  fc_op_desc.SetInput("W", {w_var_name});
  fc_op_desc.SetOutput("Out", {out_var_name});
  fc_op_desc.SetAttr("in_num_col_dims", static_cast<int>(in_num_col_dims));
  if (has_bias) {
    auto* bias = scope.Var(bias_var_name)->GetMutable<Tensor>();
    bias->Resize({w_shape[1]});
    FillTensor<float, int>(bias);
    fc_op_desc.SetInput("Bias", {bias_var_name});
  }

  auto fc_op = CreateOp<operators::FcOpLite>(fc_op_desc, &scope);
  LauchOp(fc_op, {x_var_name}, {out_var_name});
  out_ref->CopyDataFrom(*out);

  // compare results
  fc_ref(fc_op);
  auto* out_data = out->mutable_data<float>();
  auto* out_ref_data = out_ref->mutable_data<float>();
  for (int i = 0; i < out->dims().production(); i++) {
    EXPECT_NEAR(out_data[i], out_ref_data[i], 1e-5);
  }

  // model release
  npu::OpList::Global().clear();
  npu::DeviceInfo::Global().Clear();
}

TEST(NPUBridges, fc) {
  for (bool use_bias : {true, false}) {
    test_fc({1, 8, 8, 1}, {8, 4}, 2, use_bias);
    test_fc({1, 5, 5, 1}, {5, 7}, 2, use_bias);
    test_fc({1, 4, 1, 1}, {4, 8}, 1, use_bias);
  }
}

}  // namespace bridge
}  // namespace npu
}  // namespace lite
}  // namespace paddle

USE_LITE_OP(fc);
USE_NPU_BRIDGE(fc);
