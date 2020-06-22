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
#include "lite/kernels/x86/search_seq_depadding_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

TEST(search_seq_depadding_x86, retrive_op) {
  auto kernel = KernelRegistry::Global().Create("search_seq_depadding");
  ASSERT_FALSE(kernel.empty());
  ASSERT_TRUE(kernel.front());
}

TEST(search_seq_depadding_x86, init) {
  SearchSeqDepaddingCompute<float> ssdc;
  ASSERT_EQ(ssdc.precision(), PRECISION(kFloat));
  ASSERT_EQ(ssdc.target(), TARGET(kX86));
}

TEST(search_seq_depadding_x86, run_test) {
  lite::Tensor pad, src, out;
  pad.Resize({2 * 3, 4});
  src.Resize({3, 1});
  out.Resize({3, 4});
  LoD pad_lod{};
  pad_lod.push_back({0, 4, 6});
  pad.set_lod(pad_lod);
  LoD src_lod{};
  src_lod.push_back({0, 2, 3});
  src.set_lod(src_lod);

  auto* pad_data = pad.mutable_data<float>();
  for (int64_t i = 0; i < pad.dims().production(); i++) {
    pad_data[i] = static_cast<float>(i);
  }
  SearchSeqDepaddingCompute<float> ssdc;
  operators::SearchSeqDepaddingParam param;

  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<X86Context>();
  ssdc.SetContext(std::move(ctx));

  param.pad = &pad;
  param.src = &src;
  param.out = &out;

  ssdc.SetParam(param);
  ssdc.Run();

  std::vector<float> ref_results = {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19};
  auto* out_data = out.mutable_data<float>();
  for (int i = 0; i < out.dims().production(); i++) {
    EXPECT_NEAR(out_data[i], ref_results[i], 1e-3);
  }
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(search_seq_depadding, kX86, kFloat, kNCHW, def);
