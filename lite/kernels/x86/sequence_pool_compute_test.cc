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

#include <memory>
#include <utility>
#include <vector>

#include "lite/core/op_registry.h"
#include "lite/kernels/x86/sequence_pool_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

TEST(sequence_pool_x86, retrive_op) {
  auto sequence_pool = KernelRegistry::Global().Create("sequence_pool");
  ASSERT_FALSE(sequence_pool.empty());
  ASSERT_TRUE(sequence_pool.front());
}

TEST(sequence_pool_x86, init) {
  SequencePoolCompute<float> sequence_pool;
  ASSERT_EQ(sequence_pool.precision(), PRECISION(kFloat));
  ASSERT_EQ(sequence_pool.target(), TARGET(kX86));
}

TEST(sequence_pool_x86, run_test) {
  lite::Tensor x, out;
  lite::LoD lod;
  lod.push_back(std::vector<uint64_t>{0, 10});

  x.set_lod(lod);
  const size_t second_dim = 8u;
  std::vector<int64_t> input_shape{static_cast<int64_t>(lod[0].back()),
                                   static_cast<int64_t>(second_dim)};
  lite::DDim in_dims(input_shape);
  x.Resize(in_dims);

  const size_t out_first_dim = lod[0].size() - 1;
  std::vector<int64_t> output_shape{static_cast<int64_t>(out_first_dim),
                                    static_cast<int64_t>(second_dim)};
  lite::DDim out_dims(output_shape);
  out.Resize(out_dims);

  auto x_data = x.mutable_data<float>();
  auto out_data = out.mutable_data<float>();

  for (int64_t i = 0; i < x.dims().production(); i++) {
    x_data[i] = 1.1f * i;
  }

  SequencePoolCompute<float> sequence_pool;
  operators::SequencePoolParam param;
  param.X = &x;
  param.Out = &out;

  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<X86Context>();
  sequence_pool.SetContext(std::move(ctx));
  sequence_pool.SetParam(param);
  sequence_pool.Run();

  std::vector<float> ref_results = {
      39.6f, 40.7f, 41.8f, 42.9f, 44.f, 45.1f, 46.2f, 47.3f};
  for (int i = 0; i < out.dims().production(); i++) {
    EXPECT_NEAR(out_data[i], ref_results[i], 1e-3);
  }
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(sequence_pool, kX86, kFloat, kNCHW, def);
