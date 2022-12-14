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

#pragma once
#include <algorithm>
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "ft_ops/fusion_decoding_op.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

namespace xft = baidu::xpu::xft;

class FusionDecodingCompute
    : public KernelLite<TARGET(kXPU), PRECISION(kAny), DATALAYOUT(kAny)> {
 public:
  void PrepareForRun() override;

  void Run() override;

  virtual ~FusionDecodingCompute() = default;
 private:
  xft::xftTensor<float, 2> word_embedding;
  xft::xftTensor<float, 2> position_embedding;
  std::vector<xft::xftTensor<float, 1>> self_ln_weight;
  std::vector<xft::xftTensor<float, 1>> self_ln_bias;
  std::vector<xft::xftTensor<int16_t, 2>> self_q_weight;
  std::vector<xft::xftTensor<float, 1>> self_q_bias;
  std::vector<xft::xftTensor<int16_t, 2>> self_k_weight;
  std::vector<xft::xftTensor<float, 1>> self_k_bias;
  std::vector<xft::xftTensor<int16_t, 2>> self_v_weight;
  std::vector<xft::xftTensor<float, 1>> self_v_bias;
  std::vector<xft::xftTensor<int16_t, 2>> self_out_weight;
  std::vector<xft::xftTensor<float, 1>> self_out_bias;
  std::vector<xft::xftTensor<float, 1>> cross_ln_weight;
  std::vector<xft::xftTensor<float, 1>> cross_ln_bias;
  std::vector<xft::xftTensor<int16_t, 2>> cross_q_weight;
  std::vector<xft::xftTensor<float, 1>> cross_q_bias;
  std::vector<xft::xftTensor<int16_t, 2>> cross_k_weight;
  std::vector<xft::xftTensor<float, 1>> cross_k_bias;
  std::vector<xft::xftTensor<int16_t, 2>> cross_v_weight;
  std::vector<xft::xftTensor<float, 1>> cross_v_bias;
  std::vector<xft::xftTensor<int16_t, 2>> cross_out_weight;
  std::vector<xft::xftTensor<float, 1>> cross_out_bias;
  std::vector<xft::xftTensor<float, 1>> ffn_ln_weight;
  std::vector<xft::xftTensor<float, 1>> ffn_ln_bias;
  std::vector<xft::xftTensor<int16_t, 2>> ffn_inter_weight;
  std::vector<xft::xftTensor<float, 1>> ffn_inter_bias;
  std::vector<xft::xftTensor<int16_t, 2>> ffn_out_weight;
  std::vector<xft::xftTensor<float, 1>> ffn_out_bias;
  xft::xftTensor<float, 1> decoder_ln_weight;
  xft::xftTensor<float, 1> decoder_ln_bias;
  xft::xftTensor<int16_t, 2> emb_weight;
  xft::xftTensor<float, 1> emb_bias;
  xft::FdParam fd_param;
};

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
