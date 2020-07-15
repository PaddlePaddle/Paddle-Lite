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
#include "lite/kernels/x86/match_matrix_tensor_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

TEST(match_matrix_tensor_x86, retrive_op) {
  auto kernel = KernelRegistry::Global().Create("match_matrix_tensor");
  ASSERT_FALSE(kernel.empty());
  ASSERT_TRUE(kernel.front());
}

TEST(match_matrix_tensor_x86, init) {
  MatchMatrixTensorCompute<float> mmtc;
  ASSERT_EQ(mmtc.precision(), PRECISION(kFloat));
  ASSERT_EQ(mmtc.target(), TARGET(kX86));
}

TEST(match_matrix_tensor_x86, run_test) {
  int ix = 5, iy = 4, h = 2, dim_t = 2;
  lite::Tensor x, w, y, out, tmp;
  x.Resize({ix, h});
  w.Resize({h, dim_t, h});
  y.Resize({iy, h});
  out.Resize({18, 1});
  tmp.Resize({20, 1});

  LoD x_lod{};
  x_lod.push_back({0, 2, 5});
  x.set_lod(x_lod);
  LoD y_lod{};
  y_lod.push_back({0, 3, 4});
  y.set_lod(y_lod);

  auto* x_data = x.mutable_data<float>();
  for (int64_t i = 0; i < x.numel(); i++) {
    x_data[i] = static_cast<float>(i);
  }
  auto* y_data = y.mutable_data<float>();
  for (int64_t i = 0; i < y.numel(); i++) {
    y_data[i] = static_cast<float>(i);
  }
  auto* w_data = w.mutable_data<float>();
  for (int64_t i = 0; i < w.numel(); i++) {
    w_data[i] = static_cast<float>(i);
  }

  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<X86Context>();
  MatchMatrixTensorCompute<float> mmtc;
  mmtc.SetContext(std::move(ctx));

  operators::MatchMatrixTensorParam param;
  param.x = &x;
  param.w = &w;
  param.y = &y;
  param.dim_t = dim_t;
  param.out = &out;
  param.tmp = &tmp;

  mmtc.SetParam(param);
  mmtc.Run();

  std::vector<float> ref_results = {5,
                                    23,
                                    41,
                                    17,
                                    75,
                                    133,
                                    7,
                                    33,
                                    59,
                                    27,
                                    125,
                                    223,
                                    323,
                                    455,
                                    587,
                                    557,
                                    793,
                                    1029};
  auto* out_data = out.mutable_data<float>();
  for (int i = 0; i < out.dims().production(); i++) {
    EXPECT_NEAR(out_data[i], ref_results[i], 1e-3);
    // LOG(INFO) << out_data[i];
  }
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(match_matrix_tensor, kX86, kFloat, kNCHW, def);
