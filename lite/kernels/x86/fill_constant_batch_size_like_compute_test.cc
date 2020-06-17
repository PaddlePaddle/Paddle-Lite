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

#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "lite/core/op_registry.h"
#include "lite/kernels/x86/fill_constant_batch_size_like_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

TEST(fill_constant_batch_size_like_x86, retrive_op) {
  auto fill_constant_batch_size_like =
      KernelRegistry::Global().Create("fill_constant_batch_size_like");
  ASSERT_FALSE(fill_constant_batch_size_like.empty());
  ASSERT_TRUE(fill_constant_batch_size_like.front());
}

TEST(fill_constant_batch_size_like_x86, init) {
  lite::kernels::x86::FillConstantBatchSizeLikeCompute<float>
      fill_constant_batch_size_like;
  ASSERT_EQ(fill_constant_batch_size_like.precision(), PRECISION(kFloat));
  ASSERT_EQ(fill_constant_batch_size_like.target(), TARGET(kX86));
}

TEST(fill_constant_batch_size_like_x86, run_test) {
  lite::Tensor input;
  lite::Tensor out;
  std::vector<int64_t> input_shape{219, 232};
  input.Resize(input_shape);
  std::vector<int64_t> out_shape{219, 132, 7};
  out.Resize(out_shape);

  auto input_data = input.mutable_data<float>();
  auto out_data = out.mutable_data<float>();

  for (int64_t i = 0; i < input.dims().production(); ++i) {
    input_data[i] = static_cast<float>(i);
  }

  FillConstantBatchSizeLikeCompute<float> fill_constant_batch_size_like;
  operators::FillConstantBatchSizeLikeParam param;
  param.input = &input;
  param.out = &out;
  std::vector<int> shape{-1, 132, 7};
  float value = 3.5;
  param.shape = shape;
  param.value = value;

  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<X86Context>();
  fill_constant_batch_size_like.SetContext(std::move(ctx));
  fill_constant_batch_size_like.SetParam(param);
  fill_constant_batch_size_like.Run();

  std::vector<float> ref_results{
      3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5};
  for (size_t i = 0; i < ref_results.size(); i++) {
    EXPECT_NEAR(out_data[i], ref_results[i], 1e-3);
  }
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(fill_constant_batch_size_like, kX86, kFloat, kNCHW, def);
