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

#include "lite/operators/norm_op.h"

#include <gtest/gtest.h>

#include <cmath>
#include <iostream>

#include "lite/core/op_registry.h"
#include "lite/core/subgraph_bridge_registry.h"
#include "lite/kernels/mlu/bridges/test_helper.h"
#include "lite/kernels/mlu/bridges/utility.h"
namespace paddle {
namespace lite {
namespace subgraph {
namespace mlu {

// void ToFile(std::string file_name, Tensor* tensor) {
//   int count = tensor->dims().production();
//   auto data = tensor->mutable_data<float>();
//   std::ostringstream outs;
//   for (size_t i = 0; i < count; i++) {
//     outs << data[i] << std::endl;
//   }
//   std::ofstream of;
//   of.open(file_name, std::ios::out);
//   of << outs.str();
//   of.close();
// }

void norm_ref(const std::shared_ptr<operators::NormOp> op) {
  Scope* scope = op->scope();
  const OpInfo* op_info = op->op_info();
  auto x = scope->FindVar(op_info->Input("X").front())->GetMutable<Tensor>();
  auto out =
      scope->FindVar(op_info->Output("Out").front())->GetMutable<Tensor>();
  int axis = op_info->GetAttr<int>("axis");
  int epsilon = op_info->GetAttr<float>("epsilon");
  auto x_dims = x->dims();
  if (axis < 0) {
    axis += x_dims.size();
  }
  out->Resize(x_dims.Vectorize());
  auto* out_data = out->mutable_data<float>();

  const auto* x_data = x->data<float>();
  int pre_n = x_dims.count(0, axis);
  int n = x_dims[axis];
  int post_n = x_dims.count(axis + 1, x_dims.size());
  for (int i = 0; i < pre_n; i++) {
    for (int k = 0; k < post_n; k++) {
      float sum = epsilon;
      const float* in_tmp = x_data + i * n * post_n + k;
      for (int j = 0; j < n; j++) {
        sum += in_tmp[j * post_n] * in_tmp[j * post_n];
      }
      sum = std::sqrt(sum);
      float* out_tmp = out_data + i * n * post_n + k;
      for (int j = 0; j < n; j++) {
        out_tmp[j * post_n] = in_tmp[j * post_n] / sum;
      }
    }
  }
}

void test_norm(const std::vector<int64_t>& input_shape, int axis) {
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
  FillTensor<float, float>(x, -9, 9);
  // initialize op desc
  cpp::OpDesc opdesc;
  float epsilon = 1e-9f;
  opdesc.SetType("norm");
  opdesc.SetInput("X", {x_var_name});
  opdesc.SetOutput("Out", {out_var_name});
  opdesc.SetAttr("axis", static_cast<int>(axis));
  opdesc.SetAttr("epsilon", static_cast<float>(epsilon));

  // create and convert op to MLU model, then run it on MLU
  auto op = CreateOp<operators::NormOp>(opdesc, &scope);
  norm_ref(op);
  out_ref->CopyDataFrom(*out);
  Tensor input_x;
  input_x.Resize(DDim(input_shape));
  // change input layout from NCHW to NHWC
  transpose<float>(x->mutable_data<float>(),
                   input_x.mutable_data<float>(),
                   {static_cast<int>(input_shape[0]),
                    static_cast<int>(input_shape[1]),
                    static_cast<int>(input_shape[2]),
                    static_cast<int>(input_shape[3])},
                   {0, 2, 3, 1});
  x->CopyDataFrom(input_x);

  LaunchOp(op, {x_var_name}, {out_var_name});
  auto* out_data = out->mutable_data<float>();
  auto* out_ref_data = out_ref->mutable_data<float>();
  std::vector<int64_t> out_shape = input_shape;
  Tensor output_trans;
  output_trans.Resize(out_shape);
  // Change output layout from NHWC to NCHW
  transpose<float>(out_data,
                   output_trans.mutable_data<float>(),
                   {static_cast<int>(out_shape[0]),
                    static_cast<int>(out_shape[2]),
                    static_cast<int>(out_shape[3]),
                    static_cast<int>(out_shape[1])},
                   {0, 3, 1, 2});
  out_data = output_trans.mutable_data<float>();

  for (int i = 0; i < out->dims().production(); i++) {
    EXPECT_NEAR(out_data[i], out_ref_data[i], 1e-2);
  }
}

TEST(MLUBridges, norm) {
  test_norm({1, 2, 3, 4}, 1);
  test_norm({1, 2, 3, 4}, 2);
  test_norm({1, 2, 3, 4}, 3);
}

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

USE_SUBGRAPH_BRIDGE(norm, kMLU);
