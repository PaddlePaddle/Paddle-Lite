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
#include "lite/kernels/x86/search_grnn_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

TEST(search_grnn_x86, retrive_op) {
  auto kernel = KernelRegistry::Global().Create("search_grnn");
  ASSERT_FALSE(kernel.empty());
  ASSERT_TRUE(kernel.front());
}

TEST(search_grnn_x86, init) {
  SearchGrnnCompute<float> ssdc;
  ASSERT_EQ(ssdc.precision(), PRECISION(kFloat));
  ASSERT_EQ(ssdc.target(), TARGET(kX86));
}

TEST(search_grnn_x86, run_test) {
  int num_input = 128;
  int num_hidden = 128;
  int num_batch = 3;
  lite::Tensor x, wi, wh, out, idx_sorted_by_width, layout_input, tmp_buffer;
  x.Resize({num_batch, num_input});
  wi.Resize({3, num_hidden, num_input});
  wh.Resize({3, num_hidden, num_hidden});
  // out.Resize({num_batch, num_hidden});
  LoD x_lod{};
  x_lod.push_back({0, 1, 3});
  x.set_lod(x_lod);

  auto* x_data = x.mutable_data<float>();
  for (int64_t i = 0; i < x.numel(); i++) {
    x_data[i] = static_cast<float>(i);
  }
  auto* wi_data = wi.mutable_data<float>();
  for (int64_t i = 0; i < wi.numel(); i++) {
    wi_data[i] = static_cast<float>(i);
  }
  auto* wh_data = wh.mutable_data<float>();
  for (int64_t i = 0; i < wh.numel(); i++) {
    wh_data[i] = static_cast<float>(i);
  }

  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<X86Context>();

  operators::SearchGrnnParam param;
  param.x = &x;
  param.wi = &wi;
  param.wh = &wh;
  param.out = &out;
  param.idx_sorted_by_width = &idx_sorted_by_width;
  param.layout_input = &layout_input;
  param.tmp_buffer = &tmp_buffer;
  param.num_input = num_input;
  param.num_hidden = num_hidden;

  SearchGrnnCompute<float> sgc;
  sgc.SetContext(std::move(ctx));
  sgc.SetParam(param);
  sgc.Run();

  // std::vector<float> ref_results = {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19};
  auto* out_data = out.mutable_data<float>();
  LOG(INFO) << out.numel();
  for (int i = 0; i < out.numel(); i++) {
    // EXPECT_NEAR(out_data[i], ref_results[i], 1e-3);
    LOG(INFO) << out_data[i];
  }
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(search_grnn, kX86, kFloat, kNCHW, def);
