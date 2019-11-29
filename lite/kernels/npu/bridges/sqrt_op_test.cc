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
#include <cmath>
#include "lite/core/op_registry.h"
#include "lite/kernels/npu/bridges/registry.h"
#include "lite/kernels/npu/bridges/test_helper.h"
#include "lite/operators/activation_ops.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace npu {
namespace bridges {

template <typename dtype>
void sqrt_ref(const std::shared_ptr<operators::ActivationOp> op) {
  Scope* scope = op->scope();
  const OpInfo* op_info = op->op_info();

  auto x = scope->FindTensor("x");
  auto out = scope->FindMutableTensor("out_ref");
  out->Resize(x->dims());
  auto x_data = x->data<dtype>();
  auto out_data = out->mutable_data<dtype>();

  for (size_t i = 0; i < x->numel(); i++) {
    out_data[i] = std::sqrtf(x_data[i]);
  }
}

void test_sqrt(const std::vector<int64_t>& input_shape) {
  // prepare input&output variables
  Scope scope;
  std::string x_var_name = "x";
  std::string out_var_name = "out";
  std::string out_ref_var_name = "out_ref";
  auto* x = scope.NewTensor(x_var_name);
  auto* out = scope.NewTensor(out_var_name);
  auto* out_ref = scope.NewTensor(out_ref_var_name);
  x->Resize(input_shape);

  // initialize input&output data
  FillTensor<float>(x, 0, 5);

  // initialize op desc
  cpp::OpDesc opdesc;
  opdesc.SetType("sqrt");
  opdesc.SetInput("X", {x_var_name});
  opdesc.SetOutput("Out", {out_var_name});

  // create and convert op to NPU model, then run it on NPU
  auto op = CreateOp<operators::ActivationOp>(opdesc, &scope);
  LauchOp(op, {x_var_name}, {out_var_name});

  // execute reference implementation and save to output tensor
  sqrt_ref<float>(op);

  // compare results
  auto* out_data = out->mutable_data<float>();
  auto* out_ref_data = out_ref->mutable_data<float>();
  for (int i = 0; i < out->dims().production(); i++) {
    EXPECT_NEAR(out_data[i], out_ref_data[i], 1e-2);
  }
}

TEST(NPUBridges, sqrt) {
  test_sqrt({2});
  test_sqrt({2, 3});
  test_sqrt({1, 2, 3, 4});
  test_sqrt({5, 6, 7, 8});
}

}  // namespace bridges
}  // namespace npu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_OP(sqrt);
USE_NPU_BRIDGE(sqrt);
