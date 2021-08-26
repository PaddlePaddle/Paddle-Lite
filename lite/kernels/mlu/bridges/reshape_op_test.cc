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

#include "lite/operators/reshape_op.h"

#include <gtest/gtest.h>

#include <random>

#include "lite/core/op_registry.h"
#include "lite/core/subgraph/subgraph_bridge_registry.h"
#include "lite/kernels/mlu/bridges/test_helper.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace mlu {

void test_reshape(std::vector<int64_t> input_shape,
                  std::vector<int64_t> out_shape) {
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
  opdesc.SetType("reshape2");
  opdesc.SetInput("X", {x_var_name});
  opdesc.SetOutput("Out", {out_var_name});
  std::vector<int> shape_attr;
  shape_attr.resize(out_shape.size());
  for (size_t i = 0; i < out_shape.size(); i++) {
    shape_attr[i] = static_cast<int>(out_shape[i]);
  }

  opdesc.SetAttr<std::vector<int>>("shape", shape_attr);
  auto op = CreateOp<operators::ReshapeOp>(opdesc, &scope);

  auto os = out->dims();
  out->Resize(out_shape);
  LaunchOp(op, {x_var_name}, {out_var_name});

  Tensor out_trans;
  out_trans.Resize(out_shape);
  transpose(out->mutable_data<float>(),
            out_trans.mutable_data<float>(),
            {static_cast<int>(out_shape[0]),
             static_cast<int>(out_shape[1]),
             static_cast<int>(out_shape[2]),
             static_cast<int>(out_shape[3])},
            {0, 3, 1, 2});
  out->CopyDataFrom(out_trans);
  // compare results
  auto* out_data = out->mutable_data<float>();
  for (int i = 0; i < out->dims().production(); i++) {
    EXPECT_NEAR(out_data[i], x_cpu.mutable_data<float>()[i], 1e-5);
  }
}

TEST(MLUBridges, reshape) { test_reshape({1, 2, 4, 4}, {1, 4, 2, 4}); }
}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

USE_SUBGRAPH_BRIDGE(reshape, kMLU);
USE_SUBGRAPH_BRIDGE(reshape2, kMLU);
