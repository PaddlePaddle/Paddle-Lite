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
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"
#include "lite/core/subgraph_bridge_registry.h"
#include "lite/kernels/mlu/bridges/test_helper.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace mlu {

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

void test_fc(const std::vector<int64_t>& input_shape,
             const std::vector<int64_t>& w_shape,
             int in_num_col_dims,
             bool has_bias) {
  CHECK_EQ(w_shape.size(), 2UL);

  Scope scope;
  std::string input_var_name("Input");
  std::string w_var_name("W");
  std::string w_int_var_name("W_int");
  std::string bias_var_name("Bias");
  std::string out_var_name("Out");
  std::string out_ref_var_name("out_ref");
  auto* input = scope.Var(input_var_name)->GetMutable<Tensor>();
  auto* w = scope.Var(w_var_name)->GetMutable<Tensor>();
  auto* w_int = scope.Var(w_int_var_name)->GetMutable<Tensor>();
  auto* out = scope.Var(out_var_name)->GetMutable<Tensor>();
  auto* out_ref = scope.Var(out_ref_var_name)->GetMutable<Tensor>();
  input->Resize(input_shape);
  w->Resize(w_shape);
  w_int->Resize(w_shape);

  FillTensor<int8_t, int8_t>(w_int, -127, 127);
  float w_scale = 1. / 1024;
  float input_scale = 1. / 8;

  Tensor input_int;
  input_int.Resize(input_shape);
  FillTensor<int8_t, int8_t>(&input_int, -127, 127);
  for (size_t i = 0; i < input->data_size(); i++) {
    input->mutable_data<float>()[i] = input_int.data<int8_t>()[i] * input_scale;
  }

  for (size_t i = 0; i < w->data_size(); i++) {
    w->mutable_data<float>()[i] = w_int->data<int8_t>()[i] * w_scale;
  }

  // create fc op
  cpp::OpDesc fc_op_desc;
  fc_op_desc.SetType("fc");
  fc_op_desc.SetInput("Input", {input_var_name});
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
  fc_ref(fc_op);
  out_ref->CopyDataFrom(*out);

  // create fc imlu op
  cpp::OpDesc fc_op_desc_mlu;
  fc_op_desc_mlu.SetType("fc");
  fc_op_desc_mlu.SetInput("Input", {input_var_name});
  fc_op_desc_mlu.SetInput("W", {w_int_var_name});
  fc_op_desc_mlu.SetOutput("Out", {out_var_name});
  fc_op_desc_mlu.SetAttr("in_num_col_dims", static_cast<int>(in_num_col_dims));

  OpInfo op_info(fc_op_desc_mlu);
  op_info.SetInputScale(w_int_var_name,
                        std::vector<float>(w_shape[1], w_scale));
  op_info.SetInputScale(input_var_name, {input_scale});
  if (has_bias) {
    op_info.SetInput("Bias", {bias_var_name});
  }

  auto fc_op_mlu = CreateOp<operators::FcOpLite>(op_info, &scope);

  Tensor input_tmp, out_tmp;
  input_tmp.Resize(input_shape);
  transpose(input->mutable_data<float>(),
            input_tmp.mutable_data<float>(),
            {static_cast<int>(input_shape[0]),
             static_cast<int>(input_shape[1]),
             static_cast<int>(input_shape[2]),
             static_cast<int>(input_shape[3])},
            {0, 2, 3, 1});
  input->CopyDataFrom(input_tmp);

  LaunchOp(fc_op_mlu, {input_var_name}, {out_var_name});

  auto os = out->dims();
  out_tmp.Resize(os);
  auto* out_data = out->mutable_data<float>();
  //  transpose(out_data,
  //            out_tmp.mutable_data<float>(),
  //            {static_cast<int>(os[0]),
  //             static_cast<int>(os[2]),
  //             static_cast<int>(os[3]),
  //             static_cast<int>(os[1])},
  //            {0, 3, 1, 2});
  //
  //  out_data = out_tmp.mutable_data<float>();

  // compare results
  auto* out_ref_data = out_ref->mutable_data<float>();
  for (int i = 0; i < out->dims().production(); i++) {
    EXPECT_NEAR(out_data[i], out_ref_data[i], 1e-5);
  }
}

TEST(MLUBridges, fc) {
  for (bool use_bias : {true, false}) {
    test_fc({1, 8, 8, 1}, {64, 4}, 1, use_bias);
    test_fc({1, 5, 5, 1}, {25, 7}, 1, use_bias);
    test_fc({1, 4, 1, 1}, {4, 8}, 1, use_bias);
    test_fc({1, 1024, 1, 1}, {1024, 32}, 1, use_bias);
  }
}

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

USE_SUBGRAPH_BRIDGE(fc, kMLU);
