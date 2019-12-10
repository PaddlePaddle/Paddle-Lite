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

#include "lite/operators/unsqueeze_op.h"
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

static DDim GetOutputShape(const std::vector<int>& unsqz_dims,
                           const DDim& in_dims) {
  int output_size = in_dims.size() + static_cast<int>(unsqz_dims.size());
  int cur_output_size = in_dims.size();
  std::vector<int64_t> output_shape(output_size, 0);

  // Validate Check: rank range.
  CHECK_LE(output_size, 6) << "The output tensor's rank should be less than 6.";

  for (int axis : unsqz_dims) {
    int cur = axis < 0 ? axis + cur_output_size + 1 : axis;
    // Validate Check: the axis bound
    CHECK((cur >= 0) && (cur <= cur_output_size))
        << "The unsqueeze dims must be within range of current rank.";
    // Move old axis, and insert new axis
    for (int i = cur_output_size; i >= cur; --i) {
      if (output_shape[i] == 1) {
        // Move axis
        output_shape[i + 1] = 1;
        output_shape[i] = 0;
      }
    }

    output_shape[cur] = 1;
    // Add the output size.
    cur_output_size++;
  }

  // Make output shape
  for (int in_idx = 0, out_idx = 0; out_idx < output_size; ++out_idx) {
    if (output_shape[out_idx] == 0) {
      output_shape[out_idx] = in_dims[in_idx++];
    }
  }

  return DDim(output_shape);
}

template <typename dtype>
void unsqueeze_ref(const std::shared_ptr<operators::UnsqueezeOp> op) {
  Scope* scope = op->scope();
  const OpInfo* op_info = op->op_info();

  auto x = scope->FindTensor("x");
  auto out = scope->FindMutableTensor("out_ref");
  auto axes = op_info->GetAttr<std::vector<int>>("axes");
  auto y_dims = GetOutputShape(axes, x->dims());
  out->Resize(y_dims);

  auto x_data = x->data<dtype>();
  auto out_data = out->mutable_data<dtype>();

  memcpy(out_data, x_data, x->numel() * sizeof(float));
}

void test_unsqueeze(const std::vector<int64_t>& input_shape,
                    std::vector<int> axes) {
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
  opdesc.SetType("unsqueeze");
  opdesc.SetInput("X", {x_var_name});
  opdesc.SetOutput("Out", {out_var_name});
  opdesc.SetAttr("axes", axes);

  // create and convert op to NPU model, then run it on NPU
  auto op = CreateOp<operators::UnsqueezeOp>(opdesc, &scope);
  LauchOp(op, {x_var_name}, {out_var_name});

  // execute reference implementation and save to output tensor
  unsqueeze_ref<float>(op);

  // compare results
  CHECK_EQ(out->dims().size(), out_ref->dims().size());
  for (int i = 0; i < out->dims().size(); i++) {
    CHECK_EQ(out->dims()[i], out_ref->dims()[i]);
  }

  auto* out_data = out->mutable_data<float>();
  auto* out_ref_data = out_ref->mutable_data<float>();
  for (int i = 0; i < out->dims().production(); i++) {
    EXPECT_NEAR(out_data[i], out_ref_data[i], 1e-2);
  }
}

TEST(NPUBridges, unsqueeze) {
  test_unsqueeze({2}, {0, 2});
  test_unsqueeze({2, 3}, {1, 3});
  test_unsqueeze({1, 2, 3}, {3});
  test_unsqueeze({5, 6, 7}, {1});
}

}  // namespace bridges
}  // namespace npu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_OP(unsqueeze);
USE_NPU_BRIDGE(unsqueeze);
