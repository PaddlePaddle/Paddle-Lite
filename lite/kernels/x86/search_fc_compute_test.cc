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
#include "lite/kernels/x86/search_fc_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

void fc_cpu_base(const lite::Tensor* X,
                 const lite::Tensor* W,
                 const lite::Tensor* b,
                 int out_size,
                 lite::Tensor* Out) {
  const float* data_in = X->data<float>();
  const float* bias = b->data<float>();
  const float* weights = W->data<float>();
  float* data_out = Out->mutable_data<float>();
  int out_rows = X->dims()[0];
  int in_cols = X->numel() / out_rows;
  int out_cols = W->numel() / in_cols;
  int index_out;

  for (int i = 0; i < out_rows; i++) {
    for (int j = 0; j < out_cols; j++) {
      index_out = i * out_cols + j;
      data_out[index_out] = bias ? bias[j] : 0;

      for (int k = 0; k < in_cols; k++) {
        data_out[index_out] +=
            data_in[i * in_cols + k] * weights[j * in_cols + k];
      }
    }
  }
}

TEST(search_fc_x86, retrive_op) {
  auto search_fc = KernelRegistry::Global().Create("search_fc");
  ASSERT_FALSE(search_fc.empty());
  ASSERT_TRUE(search_fc.front());
}

TEST(search_fc_x86, init) {
  SearchFcCompute<float> search_fc;
  ASSERT_EQ(search_fc.precision(), PRECISION(kFloat));
  ASSERT_EQ(search_fc.target(), TARGET(kX86));
}

TEST(search_fc_x86, run_test) {
  lite::Tensor x, w, b, out;
  lite::Tensor out_ref;
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<X86Context>();
  std::vector<int64_t> x_shape{1, 4};
  x.Resize(lite::DDim(x_shape));
  std::vector<int64_t> w_shape{3, 4};
  w.Resize(lite::DDim(w_shape));
  std::vector<int64_t> b_shape{3};
  b.Resize(lite::DDim(b_shape));
  std::vector<int64_t> out_shape{1, 3};
  out.Resize(lite::DDim(out_shape));
  out_ref.Resize(lite::DDim(out_shape));
  auto x_data = x.mutable_data<float>();
  auto w_data = w.mutable_data<float>();
  auto b_data = b.mutable_data<float>();
  auto out_data = out.mutable_data<float>();
  auto out_data_ref = out_ref.mutable_data<float>();
  for (int64_t i = 0; i < x.dims().production(); i++) {
    x_data[i] = static_cast<float>(i);
  }
  for (int64_t i = 0; i < w.dims().production(); i++) {
    w_data[i] = static_cast<float>(i);
  }
  for (int64_t i = 0; i < b.dims().production(); i++) {
    b_data[i] = static_cast<float>(i);
  }

  fc_cpu_base(&x, &w, &b, 3, &out_ref);

  SearchFcCompute<float> fc;
  operators::SearchFcParam param;
  param.X = &x;
  param.W = &w;
  param.b = &b;
  param.Out = &out;
  param.out_size = 3;
  fc.SetParam(param);
  fc.SetContext(std::move(ctx));
  fc.Run();

  VLOG(3) << "output vs ref";
  for (int i = 0; i < out.dims().production(); i++) {
    EXPECT_NEAR(out_data[i], out_data_ref[i], 1e-5);
  }
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(search_fc, kX86, kFloat, kNCHW, def);
