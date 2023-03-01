// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include <vector>
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "models/fusion_decoding_op.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

namespace xft = baidu::xpu::xft;

class FusionDecodingCompute
    : public KernelLite<TARGET(kXPU), PRECISION(kFloat), DATALAYOUT(kAny)> {
 public:
  void PrepareForRun() override;

  void Run() override;

  virtual ~FusionDecodingCompute() = default;

 private:
  xft::xftMat<float> word_embedding;
  xft::xftMat<float> position_embedding;
  std::vector<xft::xftVec<float>> self_ln_weight;
  std::vector<xft::xftVec<float>> self_ln_bias;
  std::vector<xft::xftMat<int16_t>> self_q_weight;
  std::vector<xft::xftVec<float>> self_q_bias;
  std::vector<xft::xftMat<int16_t>> self_k_weight;
  std::vector<xft::xftVec<float>> self_k_bias;
  std::vector<xft::xftMat<int16_t>> self_v_weight;
  std::vector<xft::xftVec<float>> self_v_bias;
  std::vector<xft::xftMat<int16_t>> self_out_weight;
  std::vector<xft::xftVec<float>> self_out_bias;
  std::vector<xft::xftVec<float>> cross_ln_weight;
  std::vector<xft::xftVec<float>> cross_ln_bias;
  std::vector<xft::xftMat<int16_t>> cross_q_weight;
  std::vector<xft::xftVec<float>> cross_q_bias;
  std::vector<xft::xftMat<int16_t>> cross_k_weight;
  std::vector<xft::xftVec<float>> cross_k_bias;
  std::vector<xft::xftMat<int16_t>> cross_v_weight;
  std::vector<xft::xftVec<float>> cross_v_bias;
  std::vector<xft::xftMat<int16_t>> cross_out_weight;
  std::vector<xft::xftVec<float>> cross_out_bias;
  std::vector<xft::xftVec<float>> ffn_ln_weight;
  std::vector<xft::xftVec<float>> ffn_ln_bias;
  std::vector<xft::xftMat<int16_t>> ffn_inter_weight;
  std::vector<xft::xftVec<float>> ffn_inter_bias;
  std::vector<xft::xftMat<int16_t>> ffn_out_weight;
  std::vector<xft::xftVec<float>> ffn_out_bias;
  xft::xftVec<float> decoder_ln_weight;
  xft::xftVec<float> decoder_ln_bias;
  xft::xftMat<int16_t> emb_weight;
  xft::xftVec<float> emb_bias;
  xft::FdParam fd_param;
};

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
