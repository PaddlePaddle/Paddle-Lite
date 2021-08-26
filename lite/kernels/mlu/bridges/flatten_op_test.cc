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

#include "lite/operators/flatten_op.h"

#include <gtest/gtest.h>

#include <random>

#include "lite/core/op_registry.h"
#include "lite/core/subgraph/subgraph_bridge_registry.h"
#include "lite/kernels/mlu/bridges/test_helper.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace mlu {

void test_flatten(std::vector<int64_t> input_shape, int axis) {
  // prepare input&output variables
  Scope scope;
  std::string x_var_name("x");
  std::string out_var_name("out");
  auto* x = scope.Var(x_var_name)->GetMutable<Tensor>();
  auto* out = scope.Var(out_var_name)->GetMutable<Tensor>();
  x->Resize(input_shape);
  Tensor x_cpu;

  // initialize input&output data
  FillTensor<float, int>(x);
  x_cpu.CopyDataFrom(*x);

  Tensor input_trans;
  input_trans.Resize(input_shape);
  transpose(x->mutable_data<float>(),
            input_trans.mutable_data<float>(),
            {static_cast<int>(input_shape[0]),
             static_cast<int>(input_shape[1]),
             static_cast<int>(input_shape[2]),
             static_cast<int>(input_shape[3])},
            {0, 2, 3, 1});
  x->CopyDataFrom(input_trans);

  // initialize op desc
  cpp::OpDesc opdesc;
  opdesc.SetType("flatten2");
  opdesc.SetInput("X", {x_var_name});
  opdesc.SetOutput("Out", {out_var_name});
  opdesc.SetAttr<int>("axis", axis);
  auto op = CreateOp<operators::FlattenOp>(opdesc, &scope);

  LaunchOp(op, {x_var_name}, {out_var_name});
  // compare results
  auto* out_data = out->mutable_data<float>();
  for (int i = 0; i < out->dims().production(); i++) {
    EXPECT_NEAR(out_data[i], x_cpu.mutable_data<float>()[i], 1e-5);
  }
}

TEST(MLUBridges, flatten) { test_flatten({1, 2, 4, 4}, 2); }
}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

USE_SUBGRAPH_BRIDGE(flatten, kMLU);
USE_SUBGRAPH_BRIDGE(flatten2, kMLU);
