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
#include "lite/kernels/x86/search_group_padding_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

TEST(search_group_padding_x86, retrieve_op) {
  auto search_group_padding =
      KernelRegistry::Global().Create("search_group_padding");
  ASSERT_FALSE(search_group_padding.empty());
  ASSERT_TRUE(search_group_padding.front());
}

TEST(search_group_padding_x86, init) {
  SearchGroupPaddingCompute<float> search_group_padding;
  ASSERT_EQ(search_group_padding.precision(), PRECISION(kFloat));
  ASSERT_EQ(search_group_padding.target(), TARGET(kX86));
}

TEST(search_group_padding_x86, run_test) {
  lite::Tensor x, out_emb_padding, out_new, out_padding;
  x.Resize({2, 3});
  out_emb_padding.Resize({-1, 3});
  out_new.Resize({2, 1});
  out_padding.Resize({-1, 1});
  LoD x_lod{};
  x_lod.push_back({0, 1});
  x.set_lod(x_lod);

  auto* x_data = x.mutable_data<float>();
  for (int64_t i = 0; i < x.dims().production(); i++) {
    x_data[i] = static_cast<float>(i);
  }
  SearchGroupPaddingCompute<float> sgp_kernel;
  operators::SearchGroupPaddingParam param;

  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<X86Context>();
  sgp_kernel.SetContext(std::move(ctx));

  param.x = &x;
  param.out_emb_padding = &out_emb_padding;
  param.out_new = &out_new;
  param.out_padding = &out_padding;

  sgp_kernel.SetParam(param);
  sgp_kernel.Run();

  std::vector<float> out_emb_padding_ref = {0, 1, 2};
  std::vector<float> out_new_ref = {0, 0};
  std::vector<float> out_padding_ref = {0};
  auto* out_emb_padding_data = out_emb_padding.mutable_data<float>();
  auto* out_new_data = out_new.mutable_data<float>();
  auto* out_padding_data = out_padding.mutable_data<float>();
  for (int i = 0; i < out_emb_padding.dims().production(); i++) {
    EXPECT_NEAR(out_emb_padding_data[i], out_emb_padding_ref[i], 1e-5);
  }
  for (int i = 0; i < out_new.dims().production(); i++) {
    EXPECT_NEAR(out_new_data[i], out_new_ref[i], 1e-5);
  }
  for (int i = 0; i < out_padding.dims().production(); i++) {
    EXPECT_NEAR(out_padding_data[i], out_padding_ref[i], 1e-5);
  }
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(search_group_padding, kX86, kFloat, kNCHW, def);
