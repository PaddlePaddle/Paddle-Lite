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

#include <gtest/gtest.h>
#include <random>
#include "lite/backends/npu/bridge/registry.h"
#include "lite/backends/npu/bridge/test_helper.h"
#include "lite/core/op_registry.h"
#include "lite/operators/relu_op.h"

namespace paddle {
namespace lite {
namespace npu {
namespace bridge {

void relu_ref(const std::shared_ptr<operators::ReluOp> op) {
  Scope* scope = op->scope();
  const OpInfo* op_info = op->op_info();
  auto x = scope->FindVar(op_info->Input("X").front())->GetMutable<Tensor>();
  auto out =
      scope->FindVar(op_info->Output("Out").front())->GetMutable<Tensor>();
  auto x_data = x->data<float>();
  auto out_data = out->mutable_data<float>();
  DDim x_dims = x->dims();
  DDim out_dims = out->dims();
  CHECK_EQ(x_dims.production(), out_dims.production());
  for (int i = 0; i < out_dims.production(); i++) {
    out_data[i] = std::max(0.f, x_data[i]);
  }
}

void test_relu(int bs, int ic, int ih, int iw) {
  // prepare input&output variables
  Scope scope;
  std::string x_var_name("x");
  std::string out_var_name("out");
  std::string out_ref_var_name("out_ref");
  auto* x = scope.Var(x_var_name)->GetMutable<Tensor>();
  auto* out = scope.Var(out_var_name)->GetMutable<Tensor>();
  auto* out_ref = scope.Var(out_ref_var_name)->GetMutable<Tensor>();
  x->Resize({bs, ic, ih, iw});

  // initialize input&output data
  FillTensor<float, int>(x);

  // initialize op desc
  cpp::OpDesc opdesc;
  opdesc.SetType("relu");
  opdesc.SetInput("X", {x_var_name});
  opdesc.SetOutput("Out", {out_var_name});

  // create and convert op to NPU model, then run it on NPU
  auto op = CreateOp<operators::ReluOp>(opdesc, &scope);
  LauchOp(op, {x_var_name}, {out_var_name});
  out_ref->CopyDataFrom(*out);

  // execute reference implementation and save to output tensor
  relu_ref(op);

  // compare results
  auto* out_data = out->mutable_data<float>();
  auto* out_ref_data = out_ref->mutable_data<float>();
  for (int i = 0; i < out->dims().production(); i++) {
    VLOG(5) << i;
    EXPECT_NEAR(out_data[i], out_ref_data[i], 1e-5);
  }
}

TEST(NPUBridges, relu) {
  for (auto bs : {1, 3}) {
    for (auto ic : {3, 4}) {
      for (auto ih : {2, 5}) {
        for (auto iw : {5, 9}) {
          VLOG(3) << "bs: " << bs << " ic: " << ic << " ih: " << ih
                  << " iw: " << iw;
          test_relu(bs, ic, ih, iw);
        }
      }
    }
  }
}

}  // namespace bridge
}  // namespace npu
}  // namespace lite
}  // namespace paddle

USE_LITE_OP(relu);
USE_NPU_BRIDGE(relu);
