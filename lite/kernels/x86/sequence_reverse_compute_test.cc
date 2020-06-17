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
#include "lite/kernels/x86/sequence_reverse_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

namespace {
static void sequence_reverse_ref(const lite::Tensor* x, lite::Tensor* y) {
  const auto* x_data = x->data<float>();
  auto seq_offset = x->lod()[x->lod().size() - 1];
  int width = x->numel() / x->dims()[0];
  auto* y_data = y->mutable_data<float>();
  for (size_t i = 0; i < seq_offset.size() - 1; ++i) {
    auto start_pos = seq_offset[i];
    auto end_pos = seq_offset[i + 1];
    for (auto pos = start_pos; pos < end_pos; ++pos) {
      auto cur_pos = end_pos - pos - 1 + start_pos;
      std::memcpy(y_data + pos * width,
                  x_data + cur_pos * width,
                  width * sizeof(float));
    }
  }
}
}  // namespace

TEST(sequence_reverse_x86, retrive_op) {
  auto sequence_reverse = KernelRegistry::Global().Create("sequence_reverse");
  ASSERT_FALSE(sequence_reverse.empty());
  ASSERT_TRUE(sequence_reverse.front());
}

TEST(sequence_reverse_x86, init) {
  SequenceReverseCompute<float, PRECISION(kFloat)> sequence_reverse;
  ASSERT_EQ(sequence_reverse.precision(), PRECISION(kFloat));
  ASSERT_EQ(sequence_reverse.target(), TARGET(kX86));
}

TEST(sequence_reverse_x86, run_test) {
  SequenceReverseCompute<float, PRECISION(kFloat)> seq_kernel;
  std::unique_ptr<KernelContext> ctx(new KernelContext);

  operators::SequenceReverseParam param;
  lite::Tensor x, x_ref;
  lite::Tensor y, y_ref;

  int32_t lod_len = 10, feature_len = 4;
  LoD lod_info{{0, 2, 4}, {0, 3, 5, 6, 10}};

  x.Resize({lod_len, feature_len});
  x_ref.Resize({lod_len, feature_len});
  y.Resize({lod_len, feature_len});
  y_ref.Resize({lod_len, feature_len});
  x.set_lod(lod_info);
  x_ref.set_lod(lod_info);
  y.set_lod(lod_info);
  y_ref.set_lod(lod_info);

  auto* y_data = y.mutable_data<float>();
  float* x_data = x.mutable_data<float>();
  float* x_ref_data = x_ref.mutable_data<float>();
  float* y_ref_data = y_ref.mutable_data<float>();

  for (int i = 0; i < x.numel(); ++i) {
    x_ref_data[i] = (i - 2.0) * 1.0;
    x_data[i] = (i - 2.0) * 1.0;
  }

  param.X = &x;
  param.Out = &y;
  seq_kernel.SetParam(param);

  seq_kernel.SetContext(std::move(ctx));
  seq_kernel.Run();

  sequence_reverse_ref(&x_ref, &y_ref);
  for (int i = 0; i < y.numel(); i++) {
    EXPECT_NEAR(y_data[i], y_ref_data[i], 1e-5);
  }
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(sequence_reverse, kX86, kFloat, kNCHW, def);
