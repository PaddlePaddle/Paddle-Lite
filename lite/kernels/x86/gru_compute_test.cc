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

#include "lite/kernels/x86/gru_compute.h"
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

TEST(gru_x86, retrive_op) {
  auto gru =
      KernelRegistry::Global().Create<TARGET(kX86), PRECISION(kFloat)>("gru");
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

  std::vector<std::vector<uint64_t> > lod{{0, 2, 6, 9}};
  input.set_lod(lod);

  auto input_data = input.mutable_data<float>();
  auto weight_data = weight.mutable_data<float>();
  auto h0_data = h0.mutable_data<float>();
  auto bias_data = bias.mutable_data<float>();

  for (int64_t i = 0; i < input.dims().production(); i++) {
    input_data[i] = static_cast<float>(i);
  }
  for (int64_t i = 0; i < weight.dims().production(); i++) {
    weight_data[i] = static_cast<float>(i);
  }
  for (int64_t i = 0; i < h0.dims().production(); i++) {
    h0_data[i] = static_cast<float>(i);
  }
  for (int64_t i = 0; i < bias.dims().production(); i++) {
    bias_data[i] = static_cast<float>(i);
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
  auto batch_reset_hidden_prev_data = batch_reset_hidden_prev.mutable_data<float>();
  auto batch_hidden_data = batch_hidden.mutable_data<float>();
  auto hidden_data = hidden.mutable_data<float>();
  LOG(INFO) << "output: ";
  for (int i = 0; i < batch_gate.dims().production(); i++) {
    LOG(INFO) << batch_gate_data[i];
  }
  for (int i = 0; i < batch_reset_hidden_prev.dims().production(); i++) {
    LOG(INFO) << batch_reset_hidden_prev_data[i];
  }
  for (int i = 0; i < batch_hidden.dims().production(); i++) {
    LOG(INFO) << batch_hidden_data[i];
  }
  for (int i = 0; i < hidden.dims().production(); i++) {
    LOG(INFO) << hidden_data[i];
  }
}

TEST(gru_grad_x86, retrive_op) {
  auto gru_grad =
      KernelRegistry::Global().Create<TARGET(kX86), PRECISION(kFloat)>("gru_grad");
  ASSERT_FALSE(gru_grad.empty());
  ASSERT_TRUE(gru_grad.front());
}

TEST(gru_grad_x86, init) {
  GRUCompute<float> gru_grad;
  ASSERT_EQ(gru_grad.precision(), PRECISION(kFloat));
  ASSERT_EQ(gru_grad.target(), TARGET(kX86));
}

TEST(gru_grad_x86, run_test) {
  lite::Tensor input, h0, weight, bias;
  lite::Tensor batch_gate, batch_reset_hidden_prev, batch_hidden, hidden;
  lite::Tensor hidden_grad, input_grad, h0_grad, weight_grad, bias_grad;
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
  std::vector<int64_t> hidden_grad_shape{batch_size, 5};
  hidden_grad.Resize(lite::DDim(hidden_grad_shape));
  std::vector<int64_t> input_grad_shape{batch_size, 15};
  input_grad.Resize(lite::DDim(input_grad_shape));
  std::vector<int64_t> weight_grad_shape{5, 15};
  weight_grad.Resize(lite::DDim(weight_grad_shape));
  std::vector<int64_t> h0_grad_shape{3, 5};
  h0_grad.Resize(lite::DDim(h0_grad_shape));
  std::vector<int64_t> bias_grad_shape{1, 15};
  bias_grad.Resize(lite::DDim(bias_grad_shape));

  std::vector<std::vector<uint64_t> > lod{{0, 2, 6, 9}};
  input.set_lod(lod);
  hidden.set_lod(lod);
  hidden_grad.set_lod(lod);
  std::vector<std::vector<uint64_t> > out_lod{{0, 3, 6, 8, 9}, {2, 6, 0, 3, 7, 1, 4, 8, 5}, {1, 2, 0}};
  batch_gate.set_lod(out_lod);
  batch_hidden.set_lod(out_lod);

  auto input_data = input.mutable_data<float>();
  auto weight_data = weight.mutable_data<float>();
  auto h0_data = h0.mutable_data<float>();
  auto bias_data = bias.mutable_data<float>();
  auto batch_gate_data = batch_gate.mutable_data<float>();
  auto batch_reset_hidden_prev_data = batch_reset_hidden_prev.mutable_data<float>();
  auto batch_hidden_data = batch_hidden.mutable_data<float>();
  auto hidden_data = hidden.mutable_data<float>();
  auto hidden_grad_data = hidden_grad.mutable_data<float>();

  for (int64_t i = 0; i < input.dims().production(); i++) {
    input_data[i] = static_cast<float>(i);
  }
  for (int64_t i = 0; i < weight.dims().production(); i++) {
    weight_data[i] = static_cast<float>(i);
  }
  for (int64_t i = 0; i < h0.dims().production(); i++) {
    h0_data[i] = static_cast<float>(i);
  }
  for (int64_t i = 0; i < bias.dims().production(); i++) {
    bias_data[i] = static_cast<float>(i);
  }
  for (int64_t i = 0; i < batch_gate.dims().production(); i++) {
    batch_gate_data[i] = static_cast<float>(i);
  }
  for (int64_t i = 0; i < batch_reset_hidden_prev.dims().production(); i++) {
    batch_reset_hidden_prev_data[i] = static_cast<float>(i);
  }
  for (int64_t i = 0; i < hidden.dims().production(); i++) {
    hidden_data[i] = static_cast<float>(i);
  }
  for (int64_t i = 0; i < batch_hidden.dims().production(); i++) {
    batch_hidden_data[i] = static_cast<float>(i);
  }
  for (int64_t i = 0; i < hidden_grad.dims().production(); i++) {
    hidden_grad_data[i] = static_cast<float>(i);
  }
  // GRUGradCompute gru_grad;
  GRUGradCompute<float> gru_grad;
  operators::GRUGradParam param;

  param.input = &input;
  param.h0 = &h0;
  param.weight = &weight;
  param.bias = &bias;
  param.batch_gate = &batch_gate;
  param.batch_reset_hidden_prev = &batch_reset_hidden_prev;
  param.batch_hidden = &batch_hidden;
  param.hidden = &hidden;
  param.hidden_grad = &hidden_grad;
  param.input_grad = &input_grad;
  param.h0_grad = &h0_grad;
  param.weight_grad = &weight_grad;
  param.bias_grad = &bias_grad;

  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<X86Context>();
  gru_grad.SetContext(std::move(ctx));
  gru_grad.SetParam(param);
  gru_grad.Run();

  auto input_grad_data = input_grad.mutable_data<float>();
  auto h0_grad_data = h0_grad.mutable_data<float>();
  auto weight_grad_data = weight_grad.mutable_data<float>();
  auto bias_grad_data = bias_grad.mutable_data<float>();
  LOG(INFO) << "output: ";
  for (int i = 0; i < input_grad.dims().production(); i++) {
    LOG(INFO) << input_grad_data[i];
  }
  for (int i = 0; i < h0_grad.dims().production(); i++) {
    LOG(INFO) << h0_grad_data[i];
  }
  for (int i = 0; i < weight_grad.dims().production(); i++) {
    LOG(INFO) << weight_grad_data[i];
  }
  for (int i = 0; i < bias_grad.dims().production(); i++) {
    LOG(INFO) << bias_grad_data[i];
  }
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(gru, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(gru_grad, kX86, kFloat, kNCHW, def);
