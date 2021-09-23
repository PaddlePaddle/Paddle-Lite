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

#include "lite/operators/layout_op.h"
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

void test_layout_NHWC2NCHW(std::vector<int64_t> input_shape) {
  // prepare input&output variables
  std::string x_var_name = "input";
  std::string out_var_name = "out";

  Scope scope;
  auto* x = scope.Var(x_var_name)->GetMutable<Tensor>();
  auto* out = scope.Var(out_var_name)->GetMutable<Tensor>();
  x->Resize(DDim(input_shape));
  // initialize input&output data
  FillTensor<float>(x);

  // initialize op desc
  cpp::OpDesc opdesc;
  opdesc.SetType("layout");
  opdesc.SetInput("Input", {x_var_name});
  opdesc.SetOutput("Out", {out_var_name});

  auto op = CreateOp<operators::LayoutOp>(opdesc, &scope);

  // execute reference implementation and save to output tensor
  Tensor input;
  input.Resize(DDim(input_shape));
  switch (input_shape.size()) {
    case 2:
      transpose<float>(
          x->mutable_data<float>(),
          input.mutable_data<float>(),
          {static_cast<int>(input_shape[0]), static_cast<int>(input_shape[1])},
          {0, 1});
      break;
    case 3:
      transpose<float>(x->mutable_data<float>(),
                       input.mutable_data<float>(),
                       {static_cast<int>(input_shape[0]),
                        static_cast<int>(input_shape[2]),
                        static_cast<int>(input_shape[1])},
                       {0, 2, 1});
      break;
    case 4:
      transpose<float>(x->mutable_data<float>(),
                       input.mutable_data<float>(),
                       {static_cast<int>(input_shape[0]),
                        static_cast<int>(input_shape[2]),
                        static_cast<int>(input_shape[3]),
                        static_cast<int>(input_shape[1])},
                       {0, 3, 1, 2});
      break;
    case 5:
      transpose<float>(x->mutable_data<float>(),
                       input.mutable_data<float>(),
                       {static_cast<int>(input_shape[0]),
                        static_cast<int>(input_shape[2]),
                        static_cast<int>(input_shape[3]),
                        static_cast<int>(input_shape[4]),
                        static_cast<int>(input_shape[1])},
                       {0, 4, 1, 2, 3});
      break;
    default:
      CHECK(0) << "Unsupport";
  }
  auto* x_data = input.mutable_data<float>();
  LaunchOp(op, {x_var_name}, {out_var_name});

  // compare results
  auto* out_data = out->mutable_data<float>();

  for (int i = 0; i < out->dims().production(); i++) {
    VLOG(5) << i;
    EXPECT_NEAR(out_data[i], x_data[i], 5e-4);
  }
}

void test_layout_NCHW2NHWC(std::vector<int64_t> input_shape) {
  // prepare input&output variables
  std::string x_var_name = "input";
  std::string out_var_name = "out";

  Scope scope;
  auto* x = scope.Var(x_var_name)->GetMutable<Tensor>();
  auto* out = scope.Var(out_var_name)->GetMutable<Tensor>();
  x->Resize(DDim(input_shape));
  // initialize input&output data
  FillTensor<float>(x);

  // initialize op desc
  cpp::OpDesc opdesc;
  opdesc.SetType("layout");
  opdesc.SetInput("Input", {x_var_name});
  opdesc.SetOutput("Out", {out_var_name});

  auto op = CreateOp<operators::LayoutOp>(opdesc, &scope);

  // execute reference implementation and save to output tensor
  Tensor input;
  input.Resize(DDim(input_shape));
  switch (input_shape.size()) {
    case 2:
      transpose<float>(
          x->mutable_data<float>(),
          input.mutable_data<float>(),
          {static_cast<int>(input_shape[0]), static_cast<int>(input_shape[1])},
          {0, 1});
      break;
    case 3:
      transpose<float>(x->mutable_data<float>(),
                       input.mutable_data<float>(),
                       {static_cast<int>(input_shape[0]),
                        static_cast<int>(input_shape[1]),
                        static_cast<int>(input_shape[2])},
                       {0, 2, 1});
      break;
    case 4:
      transpose<float>(x->mutable_data<float>(),
                       input.mutable_data<float>(),
                       {static_cast<int>(input_shape[0]),
                        static_cast<int>(input_shape[1]),
                        static_cast<int>(input_shape[2]),
                        static_cast<int>(input_shape[3])},
                       {0, 2, 3, 1});
      break;
    case 5:
      transpose<float>(x->mutable_data<float>(),
                       input.mutable_data<float>(),
                       {static_cast<int>(input_shape[0]),
                        static_cast<int>(input_shape[1]),
                        static_cast<int>(input_shape[2]),
                        static_cast<int>(input_shape[3]),
                        static_cast<int>(input_shape[4])},
                       {0, 2, 3, 4, 1});
      break;
    default:
      CHECK(0) << "Unsupport";
  }
  auto* x_data = input.mutable_data<float>();
  LaunchOp(op, {x_var_name}, {out_var_name}, CNML_NCHW);

  // compare results
  auto* out_data = out->mutable_data<float>();

  for (int i = 0; i < out->dims().production(); i++) {
    VLOG(5) << i;
    EXPECT_NEAR(out_data[i], x_data[i], 5e-4);
  }
}

TEST(MLUBridges, layout) {
  test_layout_NHWC2NCHW({12, 32, 4});
  test_layout_NHWC2NCHW({12, 32, 44, 3});
  test_layout_NHWC2NCHW({12, 32, 44, 3, 6});
  test_layout_NCHW2NHWC({12, 32, 55});
  test_layout_NCHW2NHWC({12, 32, 44, 3});
  test_layout_NCHW2NHWC({12, 32, 44, 3, 8});
  test_layout_NHWC2NCHW({12, 32});
  test_layout_NCHW2NHWC({12, 32});
}

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

USE_SUBGRAPH_BRIDGE(layout, kMLU);
