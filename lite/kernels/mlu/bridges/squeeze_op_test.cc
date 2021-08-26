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

#include "lite/operators/squeeze_op.h"
#include <gtest/gtest.h>
#include <memory>
#include <utility>
#include <vector>
#include "lite/core/op_registry.h"
#include "lite/core/subgraph/subgraph_bridge_registry.h"
#include "lite/kernels/mlu/bridges/test_helper.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace mlu {

// squeeze
TEST(MLUBridges, squeeze) {
  Scope scope;
  std::string x_var_name("x");
  std::string out_var_name("out");
  std::string ref_var_name("ref");
  auto* x = scope.Var(x_var_name)->GetMutable<Tensor>();
  auto* out = scope.Var(out_var_name)->GetMutable<Tensor>();
  auto* out_ref = scope.Var(ref_var_name)->GetMutable<Tensor>();
  std::vector<int64_t> x_shape({1, 3, 1, 5});
  x->Resize(x_shape);
  out_ref->Resize(x_shape);
  std::vector<int64_t> out_shape({3, 5});
  out->Resize(out_shape);

  FillTensor<float>(x, 0, 10);
  out_ref->CopyDataFrom(*x);

  // SqueezeCompute squeeze;
  cpp::OpDesc opdesc;
  opdesc.SetType("squeeze");
  opdesc.SetInput("X", {x_var_name});
  opdesc.SetOutput("Out", {out_var_name});

  std::vector<int> axes{0, -2};
  opdesc.SetAttr("axes", axes);
  // create and convert op to MLU model, then run it on MLU
  auto op = CreateOp<operators::SqueezeOp>(opdesc, &scope);
  LaunchOp(op, {x_var_name}, {out_var_name});

  auto x_data = out_ref->data<float>();
  auto out_data = out->data<float>();
  for (int j = 0; j < out->numel(); ++j) {
    EXPECT_NEAR(out_data[j], x_data[j], 1e-5);
  }
}

// squeeze2
TEST(MLUBridges, squeeze2) {
  Scope scope;
  std::string x_var_name("x");
  std::string out_var_name("out");
  std::string xshape_var_name("xshape");
  std::string ref_var_name("ref");
  auto* x = scope.Var(x_var_name)->GetMutable<Tensor>();
  auto* out = scope.Var(out_var_name)->GetMutable<Tensor>();
  auto* xshape = scope.Var(xshape_var_name)->GetMutable<Tensor>();
  auto* out_ref = scope.Var(ref_var_name)->GetMutable<Tensor>();
  std::vector<int64_t> x_shape({1, 3, 1, 5});
  x->Resize(x_shape);
  out_ref->Resize(x_shape);
  std::vector<int64_t> out_shape({3, 5});
  out->Resize(out_shape);
  std::vector<int64_t> xshape_shape({1, 3, 1, 5});
  xshape->Resize(xshape_shape);

  FillTensor<float>(x, 0, 10);
  out_ref->CopyDataFrom(*x);

  // Squeeze2Compute squeeze2;
  cpp::OpDesc opdesc;
  opdesc.SetType("squeeze2");
  opdesc.SetInput("X", {x_var_name});
  opdesc.SetOutput("Out", {out_var_name});
  opdesc.SetOutput("XShape", {xshape_var_name});

  std::vector<int> axes({0, -2});
  opdesc.SetAttr("axes", axes);
  // create and convert op to MLU model, then run it on MLU
  auto op = CreateOp<operators::SqueezeOp>(opdesc, &scope);
  LaunchOp(op, {x_var_name}, {out_var_name, xshape_var_name});

  auto x_data = out_ref->mutable_data<float>();
  auto out_data = out->mutable_data<float>();
  auto xshape_data = xshape->mutable_data<float>();
  for (int j = 0; j < out->numel(); ++j) {
    EXPECT_NEAR(out_data[j], x_data[j], 1e-5);
    EXPECT_NEAR(xshape_data[j], x_data[j], 1e-5);
  }
}

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

USE_SUBGRAPH_BRIDGE(squeeze, kMLU);
USE_SUBGRAPH_BRIDGE(squeeze2, kMLU);
