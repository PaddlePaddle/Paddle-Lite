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

#include "lite/kernels/x86/attention_padding_mask_compute.cc"
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

void attention_padding_mask_ref(
    const Tensor& x,
    const Tensor& y,
    Tensor* out,
    Tensor* pad_begin,
    const operators::AttentionPaddingMaskParam& param) {
  auto attn_offset = x.lod()[0];
  auto src_offset = y.lod()[0];
  int attn_seq_num = attn_offset.size() - 1;
  int src_seq_num = src_offset.size() - 1;
  int attn_seq_len = attn_offset[1];
  int src_seq_len = x.dims()[1];
  CHECK_EQ(attn_seq_num % src_seq_num, 0);

  auto count = x.numel();
  auto attn_data = x.data<float>();
  out->Resize(x.dims());
  out->set_lod(x.lod());
  auto out_data = out->mutable_data<float>();
  memcpy(out_data, attn_data, count * sizeof(float));

  for (int i = 0; i < attn_seq_num; ++i) {
    for (int j = 0; j < attn_seq_len; ++j) {
      auto tmp_out_data = out_data + src_seq_len * (attn_seq_len * i + j);
      int src_seq_idx = i % src_seq_num;
      int cur_len = src_offset[src_seq_idx + 1] - src_offset[src_seq_idx];
      for (int k = cur_len; k < src_seq_len; k++) {
        tmp_out_data[k] = param.mask;
      }
    }
  }
}

void prepare_input(Tensor* x, const LoD& lod, int64_t dim2rd) {
  std::vector<int64_t> x_dims{static_cast<int64_t>(lod[0].back()), dim2rd};
  x->Resize(x_dims);
  x->set_lod(lod);
  auto x_data = x->mutable_data<float>();
  auto x_num = x->numel();
  for (int i = 0; i < x_num; i++) {
    x_data[i] = (i - x_num) * 1.1;
  }
}

int get_max_len(const LoD& lod) {
  int max_len = 0;
  auto offset = lod[0];
  for (int i = 0; i < offset.size() - 1; i++) {
    int cur_len = offset[i + 1] - offset[i];
    max_len = max_len < cur_len ? cur_len : max_len;
  }
  return max_len;
}

TEST(attention_padding_mask_x86, retrive_op) {
  auto attention_padding_mask =
      KernelRegistry::Global().Create<TARGET(kX86), PRECISION(kFloat)>(
          "attention_padding_mask");
  ASSERT_FALSE(attention_padding_mask.empty());
  ASSERT_TRUE(attention_padding_mask.front());
}

TEST(attention_padding_mask_x86, init) {
  AttentionPaddingMaskCompute<float> attention_padding_mask;
  ASSERT_EQ(attention_padding_mask.precision(), PRECISION(kFloat));
  ASSERT_EQ(attention_padding_mask.target(), TARGET(kX86));
}

TEST(attention_padding_mask_x86, run_test) {
  lite::Tensor x, y;
  lite::Tensor out, pad_begin, out_ref, pad_begin_ref;

  LoD x_lod{{0, 3, 6, 9, 12}}, y_lod{{0, 4, 6}};
  prepare_input(&x, x_lod, get_max_len(y_lod));
  prepare_input(&y, y_lod, 1);

  operators::AttentionPaddingMaskParam param;
  param.X = &x;
  param.Y = &y;
  param.pad_id = 12800001;
  param.mask = -90000000.f;
  param.Out = &out;
  param.pad_begin = &pad_begin;

  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<X86Context>();
  AttentionPaddingMaskCompute<float> attention_padding_mask_kernel;
  attention_padding_mask_kernel.SetParam(param);
  attention_padding_mask_kernel.SetContext(std::move(ctx));
  attention_padding_mask_kernel.Run();

  attention_padding_mask_ref(x, y, &out_ref, &pad_begin_ref, param);
  auto out_data = out.data<float>();
  auto out_ref_data = out_ref.data<float>();
  for (int i = 0; i < out.numel(); i++) {
    EXPECT_NEAR(out_data[i], out_ref_data[i], 1e-5);
  }
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(search_attention_padding_mask, kX86, kFloat, kNCHW, def);
