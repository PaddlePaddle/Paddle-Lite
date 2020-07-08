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
#include "lite/kernels/x86/pool_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

TEST(pool_x86, retrive_op) {
  auto pool2d = KernelRegistry::Global().Create("pool2d");
  ASSERT_FALSE(pool2d.empty());
  ASSERT_TRUE(pool2d.front());
}

TEST(pool2d_x86, init) {
  PoolCompute<float> pool2d;
  ASSERT_EQ(pool2d.precision(), PRECISION(kFloat));
  ASSERT_EQ(pool2d.target(), TARGET(kX86));
}

TEST(pool2d_x86, run_test) {
  lite::Tensor x, out;
  constexpr int batch_size = 1;
  std::vector<int64_t> x_shape{batch_size, 3, 4, 4};
  x.Resize(lite::DDim(x_shape));
  std::vector<int64_t> out_shape{batch_size, 3, 2, 2};
  out.Resize(lite::DDim(out_shape));

  auto x_data = x.mutable_data<float>();
  auto out_data = out.mutable_data<float>();

  for (int64_t i = 0; i < x.dims().production(); i++) {
    x_data[i] = static_cast<float>(i);
  }

  PoolCompute<float> pool2d;
  operators::PoolParam param;

  param.x = &x;
  param.output = &out;
  param.strides = {2, 2};
  std::vector<int> paddings = {0, 0, 0, 0};
  param.paddings = std::make_shared<std::vector<int>>(paddings);
  param.ksize = {2, 2};
  param.pooling_type = "max";
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<X86Context>();
  pool2d.SetContext(std::move(ctx));
  pool2d.SetParam(param);
  pool2d.Run();

  LOG(INFO) << "output: ";
  float ref_result[12] = {
      5., 7., 13., 15., 21., 23., 29., 31., 37., 39., 45., 47.};
  for (int i = 0; i < out.dims().production(); i++) {
    LOG(INFO) << out_data[i];
    EXPECT_NEAR(out_data[i], ref_result[i], 1e-5);
  }
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(pool2d, kX86, kFloat, kNCHW, def);
