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
#include "lite/kernels/x86/sequence_arithmetic_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

void sequence_arithmetic_compute_ref(const Tensor& x,
                                     const Tensor& y,
                                     Tensor* out,
                                     int op_type) {
  auto x_data = x.data<float>();
  auto y_data = y.data<float>();
  out->Resize(x.dims());
  out->set_lod(x.lod());
  auto out_data = out->mutable_data<float>();
  auto x_seq_offset = x.lod()[0];
  auto y_seq_offset = y.lod()[0];
  int seq_num = x_seq_offset.size() - 1;
  int inner_size = x.numel() / x.dims()[0];

  for (int i = 0; i < seq_num; i++) {
    int len_x = (x_seq_offset[i + 1] - x_seq_offset[i]) * inner_size;
    int len_y = (y_seq_offset[i + 1] - y_seq_offset[i]) * inner_size;
    auto input_x = x_data + x_seq_offset[i] * inner_size;
    auto input_y = y_data + y_seq_offset[i] * inner_size;
    auto t_out = out_data + x_seq_offset[i] * inner_size;
    int len = std::min(len_x, len_y);
    for (int j = 0; j < len; j++) {
      switch (op_type) {
        case 1:
          t_out[j] = input_x[j] + input_y[j];
          break;
        case 2:
          t_out[j] = input_x[j] - input_y[j];
          break;
        case 3:
          t_out[j] = input_x[j] * input_y[j];
          break;
        default:
          break;
      }
    }
    if (len_x > len) {
      memcpy(t_out + len, input_x + len, sizeof(float) * (len_x - len));
    }
  }
}

void prepare_input(Tensor* x, const LoD& x_lod) {
  x->Resize({static_cast<int64_t>(x_lod[0].back()), 3});
  x->set_lod(x_lod);
  auto x_data = x->mutable_data<float>();
  for (int i = 0; i < x->numel(); i++) {
    x_data[i] = (i - x->numel() / 2) * 1.1;
  }
}

TEST(sequence_arithmetic_x86, retrive_op) {
  auto sequence_arithmetic =
      KernelRegistry::Global().Create("sequence_arithmetic");
  ASSERT_FALSE(sequence_arithmetic.empty());
  ASSERT_TRUE(sequence_arithmetic.front());
}

TEST(sequence_arithmetic_x86, init) {
  SequenceArithmeticCompute<float> sequence_arithmetic;
  ASSERT_EQ(sequence_arithmetic.precision(), PRECISION(kFloat));
  ASSERT_EQ(sequence_arithmetic.target(), TARGET(kX86));
}

TEST(sequence_arithmetic_x86, run_test) {
  SequenceArithmeticCompute<float> sequence_arithmetic;
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<X86Context>();

  lite::Tensor x, y, out, out_ref;
  lite::LoD x_lod{{0, 2, 5, 9}}, y_lod{{0, 2, 5, 9}};
  prepare_input(&x, x_lod);
  prepare_input(&y, y_lod);

  operators::SequenceArithmeticParam param;
  param.X = &x;
  param.Y = &y;
  param.Out = &out;
  param.op_type = 1;

  sequence_arithmetic.SetContext(std::move(ctx));
  sequence_arithmetic.SetParam(param);
  sequence_arithmetic.Run();

  sequence_arithmetic_compute_ref(x, y, &out_ref, param.op_type);
  auto out_data = out.data<float>();
  auto out_ref_data = out_ref.data<float>();
  for (int i = 0; i < out.numel(); i++) {
    EXPECT_NEAR(out_data[i], out_ref_data[i], 1e-3);
  }
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(sequence_arithmetic, kX86, kFloat, kNCHW, def);
