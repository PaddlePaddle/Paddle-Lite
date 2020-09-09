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

#include "lite/operators/transpose_op.h"
#include <gtest/gtest.h>
#include "lite/core/op_registry.h"
#include "lite/core/subgraph_bridge_registry.h"
#include "lite/kernels/mlu/bridges/test_helper.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace mlu {

int data_index(std::vector<int> pos, DDimLite dims) {
  int d1 = dims[1];
  int d2 = dims[2];
  int d3 = dims[3];
  return pos[3] + pos[2] * d3 + pos[1] * d3 * d2 + pos[0] * d3 * d2 * d1;
}

std::vector<int> pos_trans(std::vector<int> in_pos, std::vector<int> axis) {
  std::vector<int> out_pos(in_pos.size());
  for (size_t i = 0; i < axis.size(); i++) {
    out_pos[axis[i]] = in_pos[i];
  }
  return out_pos;
}

template <typename dtype>
void transpose_ref(const std::shared_ptr<operators::TransposeOp> op) {
  Scope* scope = op->scope();
  const OpInfo* op_info = op->op_info();

  auto input =
      scope->FindVar(op_info->Input("X").front())->GetMutable<Tensor>();
  auto output =
      scope->FindVar(op_info->Output("Out").front())->GetMutable<Tensor>();
  auto x_dims = input->dims();
  auto y_dims = output->dims();
  auto axis = op_info->GetAttr<std::vector<int>>("axis");

  // auto input_data = input->data<dtype>();
  auto* input_data = input->mutable_data<dtype>();
  auto* output_data = output->mutable_data<dtype>();

  int input_n = x_dims[0];
  int input_c = x_dims[1];
  int input_h = x_dims[2];
  int input_w = x_dims[3];

  for (int n = 0; n < input_n; ++n) {
    for (int c = 0; c < input_c; ++c) {
      for (int h = 0; h < input_h; ++h) {
        for (int w = 0; w < input_w; ++w) {
          std::vector<int> in_pos{n, c, h, w};
          std::vector<int> out_pos = pos_trans(in_pos, axis);
          int in_index = data_index(in_pos, x_dims);
          int out_index = data_index(out_pos, y_dims);
          output_data[out_index] = input_data[in_index];
        }
      }
    }
  }
}

void test_transpose(const std::vector<int64_t>& input_shape,
                    std::vector<int> axis) {
  // prepare input&output variables
  Scope scope;
  std::string x_var_name = "x";
  std::string out_var_name = "out";
  std::string out_ref_var_name = "out_ref";
  auto* x = scope.Var(x_var_name)->GetMutable<Tensor>();
  auto* out = scope.Var(out_var_name)->GetMutable<Tensor>();
  auto* out_ref = scope.Var(out_ref_var_name)->GetMutable<Tensor>();
  x->Resize(input_shape);

  // initialize input&output data
  FillTensor<float>(x);

  // initialize op desc
  cpp::OpDesc opdesc;
  opdesc.SetType("transpose");
  opdesc.SetInput("X", {x_var_name});
  opdesc.SetOutput("Out", {out_var_name});
  opdesc.SetAttr("axis", axis);

  // create and convert op to MLU model, then run it on MLU
  auto op = CreateOp<operators::TransposeOp>(opdesc, &scope);

  // transpose_ref  must run befor LaunchOp
  // otherwise get Cannot access memory
  // execute reference implementation and save to output tensor
  transpose_ref<float>(op);
  out_ref->CopyDataFrom(*out);

  Tensor input_x;
  input_x.Resize(DDim(input_shape));
  transpose(x->mutable_data<float>(),
            input_x.mutable_data<float>(),
            {static_cast<int>(input_shape[0]),
             static_cast<int>(input_shape[1]),
             static_cast<int>(input_shape[2]),
             static_cast<int>(input_shape[3])},
            {0, 2, 3, 1});
  x->CopyDataFrom(input_x);

  LaunchOp(op, {x_var_name}, {out_var_name});
  // compare results
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
    EXPECT_NEAR(out_data[i], out_ref_data[i], 1e-2);
  }
}

// TODO(pmshst): fix the transpose test
TEST(MLUBridges, transpose) {
  std::vector<int64_t> input_shape = {2, 3, 4, 5};
  test_transpose(input_shape, std::vector<int>{0, 1, 3, 2});
}

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

USE_SUBGRAPH_BRIDGE(transpose, kMLU);
USE_SUBGRAPH_BRIDGE(transpose2, kMLU);
