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

#include "lite/kernels/x86/stack_compute.h"
#include <gtest/gtest.h>
#include <memory>
#include <utility>
#include <vector>
#include "lite/core/op_registry.h"
namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

// stack
TEST(stack_x86, retrive_op) {
  auto stack =
      KernelRegistry::Global().Create<TARGET(kX86), PRECISION(kFloat)>("stack");
  ASSERT_FALSE(stack.empty());
  ASSERT_TRUE(stack.front());
}

TEST(stack_x86, init) {
  lite::kernels::x86::StackCompute<float> stack;
  ASSERT_EQ(stack.precision(), PRECISION(kFloat));
  ASSERT_EQ(stack.target(), TARGET(kX86));
}

TEST(stack_x86, run_test) {
  lite::Tensor x;
  lite::Tensor out;
  int num_input = 5;

  std::vector<int64_t> x_shape({10, 20, 10});
  x.Resize(lite::DDim(x_shape));

  std::vector<int64_t> out_shape({5, 10, 20, 10});
  out.Resize(lite::DDim(out_shape));

  auto x_data = x.mutable_data<float>();
  auto out_data = out.mutable_data<float>();

  for (int64_t i = 0; i < x.dims().production(); ++i) {
    x_data[i] = static_cast<float>(i);
  }
  std::vector<lite::Tensor*> input;
  for (int i = 0; i < num_input; ++i) {
    input.emplace_back(&x);
  }

  // StackCompute stack;
  StackCompute<float> stack;
  operators::StackParam param;

  param.X = input;
  param.Out = &out;
  int axis = 0;
  param.axis = axis;
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<X86Context>();
  stack.SetContext(std::move(ctx));
  stack.SetParam(param);
  stack.Run();

  int ref_data = 0;
  for (int j = 0; j < out.dims().production(); ++j) {
    EXPECT_NEAR(out_data[j], ref_data, 1e-5);
    ref_data++;
    ref_data = (ref_data >= 2000) ? (ref_data - 2000) : ref_data;
  }
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(stack, kX86, kFloat, kNCHW, def);
