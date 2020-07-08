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
#include "lite/kernels/x86/softmax_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

TEST(softmax_x86, retrive_op) {
  auto softmax = KernelRegistry::Global().Create("softmax");
  ASSERT_FALSE(softmax.empty());
  ASSERT_TRUE(softmax.front());
}

TEST(softmax_x86, init) {
  SoftmaxCompute<float> softmax;
  ASSERT_EQ(softmax.precision(), PRECISION(kFloat));
  ASSERT_EQ(softmax.target(), TARGET(kX86));
}

TEST(softmax_x86, run_test) {
  lite::Tensor x, out;
  constexpr int batch_size = 1;
  std::vector<int64_t> x_shape{batch_size, 3, 3, 3};
  x.Resize(lite::DDim(x_shape));
  std::vector<int64_t> out_shape{batch_size, 3, 3, 3};
  out.Resize(lite::DDim(out_shape));

  auto x_data = x.mutable_data<float>();
  auto out_data = out.mutable_data<float>();

  for (int64_t i = 0; i < x.dims().production(); i++) {
    x_data[i] = static_cast<float>(i);
  }
  SoftmaxCompute<float> softmax;
  operators::SoftmaxParam param;

  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<X86Context>();
  softmax.SetContext(std::move(ctx));

  param.x = &x;
  param.output = &out;

  softmax.SetParam(param);
  softmax.Run();

  std::vector<float> ref_results = {
      0.0900306f, 0.244728f, 0.665241f, 0.0900306f, 0.244728f, 0.665241f,
      0.0900306f, 0.244728f, 0.665241f, 0.0900306f, 0.244728f, 0.665241f,
      0.0900306f, 0.244728f, 0.665241f, 0.0900306f, 0.244728f, 0.665241f,
      0.0900306f, 0.244728f, 0.665241f, 0.0900306f, 0.244728f, 0.665241f,
      0.0900306f, 0.244728f, 0.665241f};
  for (int i = 0; i < out.dims().production(); i++) {
    EXPECT_NEAR(out_data[i], ref_results[i], 1e-3);
  }
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(softmax, kX86, kFloat, kNCHW, def);
