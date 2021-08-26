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

#include "lite/operators/concat_op.h"
#include <gtest/gtest.h>
#include <random>
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"
#include "lite/core/subgraph/subgraph_bridge_registry.h"
#include "lite/kernels/mlu/bridges/test_helper.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace mlu {

void concat_ref(const std::shared_ptr<operators::ConcatOpLite> op) {
  Scope* scope = op->scope();
  const OpInfo* op_info = op->op_info();
  auto x = op_info->Input("X");
  std::vector<lite::Tensor*> inputs;
  for (auto var : x) {
    inputs.push_back(scope->FindVar(var)->GetMutable<lite::Tensor>());
  }
  auto out =
      scope->FindVar(op_info->Output("Out").front())->GetMutable<Tensor>();
  int axis = op_info->GetAttr<int>("axis");
  std::vector<lite::Tensor*> inputs_concat(inputs.size());
  for (size_t j = 0; j < inputs.size(); ++j) {
    inputs_concat[j] = inputs[j];
  }
  size_t num = inputs.size();
  int rows = 1;
  auto dim_0 = inputs[0]->dims();
  for (int i = 0; i < axis; ++i) {
    rows *= dim_0[i];
  }
  int out_rows = rows, out_cols = 0;
  std::vector<int64_t> inputs_cols(inputs.size());
  for (size_t i = 0; i < num; ++i) {
    int t_cols = inputs[i]->numel() / rows;
    out_cols += t_cols;
    inputs_cols[i] = t_cols;
  }
  for (int k = 0; k < out_rows; ++k) {
    float* dst_ptr = out->mutable_data<float>() + k * out_cols;
    int col_idx = 0;
    for (size_t j = 0; j < num; ++j) {
      int col_len = inputs_cols[j];
      const float* src_prt = inputs[j]->data<float>() + k * col_len;
      std::memcpy(dst_ptr + col_idx, src_prt, sizeof(float) * col_len);
      col_idx += col_len;
    }
  }
}

void test_concat(std::vector<std::vector<int64_t>> input, int axis) {
  std::string x_var_name = "x";
  std::string y_var_name = "y";
  std::string out_var_name = "out";
  std::string out_ref_var_name = "out_ref";

  // prepare input&output variables
  Scope scope;
  auto* x = scope.Var(x_var_name)->GetMutable<Tensor>();
  auto* y = scope.Var(y_var_name)->GetMutable<Tensor>();
  x->Resize(DDim(input[0]));
  y->Resize(DDim(input[1]));
  auto* out = scope.Var(out_var_name)->GetMutable<Tensor>();
  auto* out_ref = scope.Var(out_ref_var_name)->GetMutable<Tensor>();
  CHECK_EQ(out->dims(), out_ref->dims());

  // initialize input&output data
  FillTensor<float>(x);
  FillTensor<float>(y);

  // initialize op desc
  cpp::OpDesc opdesc;
  opdesc.SetType("concat");
  opdesc.SetInput("X", {x_var_name, y_var_name});
  opdesc.SetOutput("Out", {out_var_name});
  opdesc.SetAttr("axis", axis);

  auto op = CreateOp<operators::ConcatOpLite>(opdesc, &scope);
  concat_ref(op);
  out_ref->CopyDataFrom(*out);

  Tensor input_x, input_y;
  input_x.Resize(DDim(input[0]));
  input_y.Resize(DDim(input[1]));
  transpose(x->mutable_data<float>(),
            input_x.mutable_data<float>(),
            {static_cast<int>(input[0][0]),
             static_cast<int>(input[0][1]),
             static_cast<int>(input[0][2]),
             static_cast<int>(input[0][3])},
            {0, 2, 3, 1});
  transpose(y->mutable_data<float>(),
            input_y.mutable_data<float>(),
            {static_cast<int>(input[1][0]),
             static_cast<int>(input[1][1]),
             static_cast<int>(input[1][2]),
             static_cast<int>(input[1][3])},
            {0, 2, 3, 1});
  x->CopyDataFrom(input_x);
  y->CopyDataFrom(input_y);

  LaunchOp(op, {x_var_name, y_var_name}, {out_var_name});

  auto* out_data = out->mutable_data<float>();
  auto* out_ref_data = out_ref->mutable_data<float>();

  Tensor output_trans;
  output_trans.Resize(out->dims());
  auto os = out->dims();
  transpose(out_data,
            output_trans.mutable_data<float>(),
            {static_cast<int>(os[0]),
             static_cast<int>(os[2]),
             static_cast<int>(os[3]),
             static_cast<int>(os[1])},
            {0, 3, 1, 2});
  out_data = output_trans.mutable_data<float>();

  for (int i = 0; i < out->dims().production(); i++) {
    VLOG(5) << i;
    EXPECT_NEAR(out_data[i], out_ref_data[i], 5e-4);
  }
}

TEST(MLUBridges, concat) {
  test_concat({{3, 3, 5, 2}, {2, 3, 5, 2}}, 0);
  test_concat({{3, 5, 5, 2}, {3, 1, 5, 2}}, 1);
  test_concat({{3, 3, 2, 2}, {3, 3, 4, 2}}, 2);
  test_concat({{3, 3, 5, 2}, {3, 3, 5, 6}}, 3);
}

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

USE_SUBGRAPH_BRIDGE(concat, kMLU);
