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
#include "lite/kernels/x86/gru_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

TEST(gru_x86, retrive_op) {
  auto gru = KernelRegistry::Global().Create("gru");
  ASSERT_FALSE(gru.empty());
  ASSERT_TRUE(gru.front());
}

TEST(gru_x86, init) {
  GRUCompute<float> gru;
  ASSERT_EQ(gru.precision(), PRECISION(kFloat));
  ASSERT_EQ(gru.target(), TARGET(kX86));
}

TEST(gru_x86, run_test) {
  lite::Tensor input, h0, weight, bias;
  lite::Tensor batch_gate, batch_reset_hidden_prev, batch_hidden, hidden;
  constexpr int batch_size = 9;
  std::vector<int64_t> input_shape{batch_size, 15};
  input.Resize(lite::DDim(input_shape));
  std::vector<int64_t> weight_shape{5, 15};
  weight.Resize(lite::DDim(weight_shape));
  std::vector<int64_t> h0_shape{3, 5};
  h0.Resize(lite::DDim(h0_shape));
  std::vector<int64_t> bias_shape{1, 15};
  bias.Resize(lite::DDim(bias_shape));
  std::vector<int64_t> batch_gate_shape{batch_size, 15};
  batch_gate.Resize(lite::DDim(batch_gate_shape));
  std::vector<int64_t> batch_reset_hidden_prev_shape{batch_size, 5};
  batch_reset_hidden_prev.Resize(lite::DDim(batch_reset_hidden_prev_shape));
  std::vector<int64_t> batch_hidden_shape{batch_size, 5};
  batch_hidden.Resize(lite::DDim(batch_hidden_shape));
  std::vector<int64_t> hidden_shape{batch_size, 5};
  hidden.Resize(lite::DDim(hidden_shape));

  std::vector<std::vector<uint64_t>> lod{{0, 2, 6, 9}};
  input.set_lod(lod);

  auto input_data = input.mutable_data<float>();
  auto weight_data = weight.mutable_data<float>();
  auto h0_data = h0.mutable_data<float>();
  auto bias_data = bias.mutable_data<float>();

  for (int64_t i = 0; i < input.dims().production(); i++) {
    input_data[i] = static_cast<float>(0);
  }
  for (int64_t i = 0; i < weight.dims().production(); i++) {
    weight_data[i] = static_cast<float>(0);
  }
  for (int64_t i = 0; i < h0.dims().production(); i++) {
    h0_data[i] = static_cast<float>(0);
  }
  for (int64_t i = 0; i < bias.dims().production(); i++) {
    bias_data[i] = static_cast<float>(0);
  }
  // ReluCompute relu;
  GRUCompute<float> gru;
  operators::GRUParam param;

  param.input = &input;
  param.h0 = &h0;
  param.weight = &weight;
  param.bias = &bias;
  param.batch_gate = &batch_gate;
  param.batch_reset_hidden_prev = &batch_reset_hidden_prev;
  param.batch_hidden = &batch_hidden;
  param.hidden = &hidden;
  param.gate_activation = "sigmoid";
  param.activation = "tanh";
  param.is_reverse = false;
  param.origin_mode = false;

  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<X86Context>();
  gru.SetContext(std::move(ctx));
  gru.SetParam(param);
  gru.Run();

  auto batch_gate_data = batch_gate.mutable_data<float>();
  auto batch_reset_hidden_prev_data =
      batch_reset_hidden_prev.mutable_data<float>();
  auto batch_hidden_data = batch_hidden.mutable_data<float>();
  auto hidden_data = hidden.mutable_data<float>();
  std::vector<float> batch_gate_out{
      0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0,
      0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0,
      0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0,
      0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0,
      0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0,
      0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0,
      0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0,
      0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0,
      0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0};
  std::vector<float> batch_reset_hidden_prev_out{
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> batch_hidden_out{
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> hidden_out{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  LOG(INFO) << "output: ";
  for (int i = 0; i < batch_gate.dims().production(); i++) {
    LOG(INFO) << batch_gate_data[i];
    EXPECT_NEAR(batch_gate_data[i], batch_gate_out[i], 1e-3);
  }
  for (int i = 0; i < batch_reset_hidden_prev.dims().production(); i++) {
    LOG(INFO) << batch_reset_hidden_prev_data[i];
    EXPECT_NEAR(
        batch_reset_hidden_prev_data[i], batch_reset_hidden_prev_out[i], 1e-3);
  }
  for (int i = 0; i < batch_hidden.dims().production(); i++) {
    LOG(INFO) << batch_hidden_data[i];
    EXPECT_NEAR(batch_hidden_data[i], batch_hidden_out[i], 1e-3);
  }
  for (int i = 0; i < hidden.dims().production(); i++) {
    LOG(INFO) << hidden_data[i];
    EXPECT_NEAR(hidden_data[i], hidden_out[i], 1e-3);
  }
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(gru, kX86, kFloat, kNCHW, def);
