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

#include "lite/operators/argmax_op.h"
#include <gtest/gtest.h>
#include <cmath>
#include "lite/core/op_registry.h"
#include "lite/kernels/npu/bridges/registry.h"
#include "lite/kernels/npu/bridges/test_helper.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace npu {
namespace bridges {

template <typename dtype>
void argmax_ref(const std::shared_ptr<operators::ArgmaxOpLite> op) {
  Scope* scope = op->scope();
  const OpInfo* op_info = op->op_info();

  auto x = scope->FindTensor("x");
  auto out = scope->FindMutableTensor("out_ref");
  int axis = op_info->GetAttr<int64_t>("axis");
  auto x_dims = x->dims();
  if (axis < 0) {
    axis += x_dims.size();
  }
  auto y_shape = x_dims.Vectorize();
  y_shape.erase(y_shape.begin() + axis);
  out->Resize(y_shape);
  auto out_dims = out->dims();

  auto x_data = x->data<dtype>();
  auto out_data = out->mutable_data<dtype>();

  const int size = x_dims[axis];
  const int in_channel = x_dims.count(axis, x_dims.size());
  const int out_channel = out_dims.count(axis, out_dims.size());
  const int in_stride = x_dims.count(axis + 1, x_dims.size());
  const int out_stride = x_dims.count(0, axis);

  for (int n = 0; n < out_stride; n++) {
    for (int k = 0; k < in_stride; k++) {
      const float* in_ptr = x_data + n * in_channel + k;
      std::vector<std::pair<float, int>> vec;
      vec.resize(size);
      for (int i = 0; i < size; i++) {
        vec[i] = std::make_pair(in_ptr[i * in_stride], i);
      }
      // sort
      std::partial_sort(vec.begin(),
                        vec.begin() + 1,
                        vec.end(),
                        std::greater<std::pair<float, int>>());

      // out
      dtype* out_ptr = out_data + n * out_channel + k;
      *out_ptr = vec[0].second;
    }
  }
}

void test_argmax(const std::vector<int64_t>& input_shape, int axis) {
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
  FillTensor<float>(x);

  // initialize op desc
  cpp::OpDesc opdesc;
  opdesc.SetType("arg_max");
  opdesc.SetInput("X", {x_var_name});
  opdesc.SetOutput("Out", {out_var_name});
  opdesc.SetAttr("axis", static_cast<int64_t>(axis));

  // create and convert op to NPU model, then run it on NPU
  auto op = CreateOp<operators::ArgmaxOpLite>(opdesc, &scope);
  LauchOp(op, {x_var_name}, {out_var_name});

  // execute reference implementation and save to output tensor
  argmax_ref<float>(op);

  // compare results
  auto* out_data = out->mutable_data<int>();
  auto* out_ref_data = out_ref->mutable_data<float>();
  for (int i = 0; i < out->dims().production(); i++) {
    EXPECT_NEAR(out_data[i], out_ref_data[i], 1e-2);
  }
}

TEST(NPUBridges, argmax) {
  test_argmax({1, 2, 3, 4}, 1);
  test_argmax({1, 2, 3, 4}, 2);
  test_argmax({1, 2, 3, 4}, 3);
}

}  // namespace bridges
}  // namespace npu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_OP(arg_max);
USE_NPU_BRIDGE(arg_max);
