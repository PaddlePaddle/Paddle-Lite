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
#include "lite/backends/xpu/target_wrapper.h"
#include "lite/core/kernel.h"
#include "models/fusion_unified_decoding_op.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {
namespace xft = baidu::xpu::xft;

class FusionUnifiedDecodingCompute
    : public KernelLite<TARGET(kXPU), PRECISION(kFloat), DATALAYOUT(kAny)> {
 public:
  using param_t = operators::FusionUnifiedDecodingParam;

  void PrepareForRun() override;

  void Run() override;

  ~FusionUnifiedDecodingCompute() override = default;

 private:
  xft::FudParam fud_param_;
  xft::xftMat<float> word_embedding_;
  xft::xftMat<float> positional_embedding_;
  xft::xftMat<float> type_embedding_;
  std::vector<xft::xftVec<float>> self_ln_weight_;
  std::vector<xft::xftVec<float>> self_ln_bias_;
  std::vector<xft::xftMat<int16_t>> self_q_weight_;
  std::vector<xft::xftVec<float>> self_q_bias_;
  std::vector<xft::xftMat<int16_t>> self_k_weight_;
  std::vector<xft::xftVec<float>> self_k_bias_;
  std::vector<xft::xftMat<int16_t>> self_v_weight_;
  std::vector<xft::xftVec<float>> self_v_bias_;
  std::vector<xft::xftMat<int16_t>> self_out_weight_;
  std::vector<xft::xftVec<float>> self_out_bias_;
  std::vector<xft::xftVec<float>> ffn_ln_weight_;
  std::vector<xft::xftVec<float>> ffn_ln_bias_;
  std::vector<xft::xftMat<int16_t>> ffn_inter_weight_;
  std::vector<xft::xftVec<float>> ffn_inter_bias_;
  std::vector<xft::xftMat<int16_t>> ffn_out_weight_;
  std::vector<xft::xftVec<float>> ffn_out_bias_;
  xft::xftVec<float> decoder_ln_weight_;
  xft::xftVec<float> decoder_ln_bias_;
  xft::xftMat<int16_t> trans_weight_;
  xft::xftVec<float> trans_bias_;
  xft::xftVec<float> lm_ln_weight_;
  xft::xftVec<float> lm_ln_bias_;
  xft::xftMat<int16_t> emb_weight_;
  xft::xftVec<float> emb_bias_;
};
}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
