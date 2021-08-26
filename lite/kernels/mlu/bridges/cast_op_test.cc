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

#include "lite/operators/cast_op.h"
#include <gtest/gtest.h>
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"
#include "lite/core/subgraph/subgraph_bridge_registry.h"
#include "lite/kernels/mlu/bridges/test_helper.h"
#include "lite/kernels/mlu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace mlu {

void test_cast_FP16_to_FP32(std::vector<int64_t> shape) {
  // prepare input&output variables
  std::string x_var_name = "x";
  std::string out_var_name = "out";

  Scope scope;
  auto* x = scope.Var(x_var_name)->GetMutable<Tensor>();
  auto* out = scope.Var(out_var_name)->GetMutable<Tensor>();
  x->Resize(DDim(shape));
  auto* x_data = x->mutable_data<paddle::lite::fluid::float16>();

  // initialize input&output data
  for (int i = 0; i < x->dims().production(); i++) {
    x_data[i] = static_cast<paddle::lite::fluid::float16>(i);
  }
  // initialize op desc
  int in_dtype = 4, out_dtype = 5;
  cpp::OpDesc opdesc;
  opdesc.SetType("cast");
  opdesc.SetInput("X", {x_var_name});
  opdesc.SetOutput("Out", {out_var_name});
  opdesc.SetAttr("in_dtype", in_dtype);
  opdesc.SetAttr("out_dtype", out_dtype);

  auto op = CreateOp<operators::CastOp>(opdesc, &scope);

  Tensor data;
  data.Resize(DDim(shape));
  auto* copy_data = data.mutable_data<paddle::lite::fluid::float16>();
  data.CopyDataFrom(*x);
  x->set_precision(paddle::lite_api::PrecisionType::kFP16);
  LaunchOp(op, {x_var_name}, {out_var_name});

  // compare results
  auto* out_data = out->mutable_data<float>();
  for (int i = 0; i < out->dims().production(); i++) {
    VLOG(5) << i;
    EXPECT_NEAR(out_data[i], static_cast<double>(copy_data[i]), 5e-4);
  }
}

void test_cast_FP32_to_FP16(std::vector<int64_t> shape) {
  // prepare input&output variables
  std::string x_var_name = "x";
  std::string out_var_name = "out";

  Scope scope;
  auto* x = scope.Var(x_var_name)->GetMutable<Tensor>();
  auto* out = scope.Var(out_var_name)->GetMutable<Tensor>();
  x->Resize(DDim(shape));
  auto* x_data = x->mutable_data<float>();

  // initialize input&output data
  for (int i = 0; i < x->dims().production(); i++) {
    x_data[i] = static_cast<float>(i);
  }
  // initialize op desc
  int in_dtype = 5, out_dtype = 4;
  cpp::OpDesc opdesc;
  opdesc.SetType("cast");
  opdesc.SetInput("X", {x_var_name});
  opdesc.SetOutput("Out", {out_var_name});
  opdesc.SetAttr("in_dtype", in_dtype);
  opdesc.SetAttr("out_dtype", out_dtype);

  auto op = CreateOp<operators::CastOp>(opdesc, &scope);

  Tensor data;
  data.Resize(DDim(shape));
  auto* copy_data = data.mutable_data<float>();
  data.CopyDataFrom(*x);
  x->set_precision(paddle::lite_api::PrecisionType::kFloat);
  LaunchOp(op, {x_var_name}, {out_var_name});

  // compare results
  auto* out_data = out->mutable_data<paddle::lite::fluid::float16>();
  for (int i = 0; i < out->dims().production(); i++) {
    VLOG(5) << i;
    EXPECT_NEAR(static_cast<double>(out_data[i]), copy_data[i], 5e-4);
  }
}

TEST(MLUBridges, cast) {
  test_cast_FP16_to_FP32({2, 3, 4, 5});
  test_cast_FP16_to_FP32({6, 3, 2, 5});
  test_cast_FP32_to_FP16({2, 3, 4, 5});
  test_cast_FP32_to_FP16({6, 3, 2, 5});
}

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

USE_SUBGRAPH_BRIDGE(cast, kMLU);
