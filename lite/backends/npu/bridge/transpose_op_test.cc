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
#include "lite/backends/npu/bridge/registry.h"
#include "lite/backends/npu/bridge/test_helper.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace npu {
namespace bridge {

int data_index(std::vector<int> pos, DDimLite dims) {
  int d1 = dims[1];
  int d2 = dims[2];
  int d3 = dims[3];
  return pos[3] + pos[2] * d3 + pos[1] * d3 * d2 + pos[0] * d3 * d2 * d1;
}

std::vector<int> pos_trans(std::vector<int> in_pos, std::vector<int> axis) {
  std::vector<int> out_pos(in_pos.size());
  for (int i = 0; i < axis.size(); i++) {
    out_pos[axis[i]] = in_pos[i];
  }
  return out_pos;
}

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

  auto* input_data = input->data<float>();
  auto* output_data = output->mutable_data<float>();

  int input_n = x_dims[0];
  int input_c = x_dims[1];
  int input_h = x_dims[2];
  int input_w = x_dims[3];
  int output_n = y_dims[0];
  int output_c = y_dims[1];
  int output_h = y_dims[2];
  int output_w = y_dims[3];

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

void test_transpose(int bs, int ic, int ih, int iw, std::vector<int> axis) {
  // prepare input&output variables
  Scope scope;
  std::string x_var_name = "x";
  std::string out_var_name = "out";
  std::string out_ref_var_name = "out_ref";
  auto* x = scope.Var(x_var_name)->GetMutable<Tensor>();
  auto* out = scope.Var(out_var_name)->GetMutable<Tensor>();
  auto* out_ref = scope.Var(out_ref_var_name)->GetMutable<Tensor>();
  x->Resize({bs, ic, ih, iw});

  // initialize input&output data
  FillTensor<float>(x);

  // initialize op desc
  cpp::OpDesc opdesc;
  opdesc.SetType("transpose");
  opdesc.SetInput("X", {x_var_name});
  opdesc.SetOutput("Out", {out_var_name});
  opdesc.SetAttr("axis", axis);

  // create and convert op to NPU model, then run it on NPU
  auto op = CreateOp<operators::TransposeOp>(opdesc, &scope);
  LauchOp(op, {x_var_name}, {out_var_name});
  out_ref->CopyDataFrom(*out);

  // execute reference implementation and save to output tensor
  transpose_ref(op);

  // compare results
  auto* out_data = out->mutable_data<float>();
  auto* out_ref_data = out_ref->mutable_data<float>();
  for (int i = 0; i < out->dims().production(); i++) {
    EXPECT_NEAR(out_data[i], out_ref_data[i], 1e-2);
  }
}

TEST(NPUBridges, transpose) {
#if 0
  for (auto bs : {1, 4, 7}) {
    for (auto ic : {1, 4, 7}) {
      for (auto ih : {1, 4, 7}) {
        for (auto iw : {1, 4, 7}) {
          for (auto axis : {std::vector<int>{0, 1, 2, 3},
                            std::vector<int>{0, 1, 3, 2},
                            std::vector<int>{0, 3, 1, 2},
                            std::vector<int>{1, 2, 3, 0},
                            std::vector<int>{3, 2, 1, 0},
                            std::vector<int>{2, 3, 1, 0}}) {
            test_transpose(bs, ic, ih, iw, axis);
          }
        }
      }
    }
  }
#endif
  test_transpose(2, 3, 4, 5, std::vector<int>{0, 1, 3, 2});
  // test_transpose(2, 3, 4, 5, std::vector<int>{0, 1, 2, 3});
  // test_transpose(2, 2, 2, 2, std::vector<int>{0,1,3,2});
  // test_transpose(1, 1, 2, 2, std::vector<int>{0,1,3,2});
  // test_transpose(1, 1, 1, 2, std::vector<int>{0,1,2,3});
}

}  // namespace bridge
}  // namespace npu
}  // namespace lite
}  // namespace paddle

USE_LITE_OP(transpose);
USE_NPU_BRIDGE(transpose);

USE_LITE_OP(transpose2);
USE_NPU_BRIDGE(transpose2);
