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

#include "lite/kernels/x86/sequence_softmax_compute.h"
#include <gtest/gtest.h>
#include <memory>
#include <utility>
#include <vector>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

TEST(sequence_softmax_x86, retrive_op) {
  auto sequence_softmax =
      KernelRegistry::Global().Create<TARGET(kX86), PRECISION(kFloat)>(
          "sequence_softmax");
  ASSERT_FALSE(sequence_softmax.empty());
  ASSERT_TRUE(sequence_softmax.front());
}

TEST(sequence_softmax_x86, init) {
  SequenceSoftmaxCompute<float> sequence_softmax;
  ASSERT_EQ(sequence_softmax.precision(), PRECISION(kFloat));
  ASSERT_EQ(sequence_softmax.target(), TARGET(kX86));
}

TEST(sequence_softmax_x86, run_test) {
  lite::Tensor x, out;
  lite::LoD lod;
  lod.push_back(std::vector<uint64_t>{0, 3, 5, 9, 10, 12, 15});

  x.set_lod(lod);
  std::vector<int64_t> input_shape{static_cast<int64_t>(lod[0].back())};
  lite::DDim in_dims(input_shape);
  x.Resize(in_dims);
  out.Resize(in_dims);

  auto x_data = x.mutable_data<float>();
  auto out_data = out.mutable_data<float>();

  std::vector<float> data_t{
      0.7, 1, 0.6, 1.5, 1.1, 1.2, 0.2, 0.6, 1.9, 3.1, 2.5, 0.8, 0.1, 2.4, 1.3};
  for (int i = 0; i < lod[0].back(); i++) {
    x_data[i] = data_t[i];
  }

  SequenceSoftmaxCompute<float> sequence_softmax;
  operators::SequenceSoftmaxParam param;
  param.X = &x;
  param.Out = &out;

  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<X86Context>();
  sequence_softmax.SetContext(std::move(ctx));
  sequence_softmax.SetParam(param);
  sequence_softmax.Run();

  std::vector<float> ref_results = {0.30724832,
                                    0.41474187,
                                    0.2780098,
                                    0.59868765,
                                    0.40131235,
                                    0.2544242,
                                    0.09359743,
                                    0.13963096,
                                    0.5123474,
                                    1.,
                                    0.84553474,
                                    0.15446526,
                                    0.06995796,
                                    0.69777346,
                                    0.23226859};
  for (int i = 0; i < out.dims().production(); i++) {
    EXPECT_NEAR(out_data[i], ref_results[i], 1e-3);
  }
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(sequence_softmax, kX86, kFloat, kNCHW, def);
