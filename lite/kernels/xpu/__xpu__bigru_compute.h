// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>
#include "lite/core/kernel.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

class XPUBiGRUCompute : public KernelLite<TARGET(kXPU), PRECISION(kFloat)> {
 public:
  using param_t = operators::BiGRUParam;

  void PrepareForRun() override;

  void MulRun(bool forward);

  void GRURun(bool forward);

  void BiGRURun();

  virtual void Run();

  virtual ~XPUBiGRUCompute() = default;

 private:
  inline xdnn::Activation_t GetActType(const std::string& act_type) {
    if (act_type == "sigmoid") {
      return xdnn::Activation_t::SIGMOID;
    } else if (act_type == "tanh") {
      return xdnn::Activation_t::TANH;
    } else if (act_type == "relu") {
      return xdnn::Activation_t::RELU;
    } else {
      LOG(FATAL) << "unsupported activation type:" << act_type;
      return xdnn::Activation_t::LINEAR;
    }
  }

  Tensor fw_mul_out;
  Tensor bw_mul_out;

  void PrepareBiasForRun(bool forward);
  void PrepareMulWeightForRun(bool forward);
  void PrepareGRUWeightForRun(bool forward);

  XPUScratchPadGuard input_max_guard_;
  XPUScratchPadGuard mul_output_max_guard_;

  XPUScratchPadGuard fw_bias_guard_;
  XPUScratchPadGuard bw_bias_guard_;

  float fw_mul_weight_abs_max_;
  float bw_mul_weight_abs_max_;

  float fw_gru_weight_s1_abs_max_;
  float fw_gru_weight_s2_abs_max_;
  XPUScratchPadGuard fw_gru_weight_max_guard_;
  XPUScratchPadGuard fw_gru_quant_weight_guard_;

  float bw_gru_weight_s1_abs_max_;
  float bw_gru_weight_s2_abs_max_;
  XPUScratchPadGuard bw_gru_weight_max_guard_;
  XPUScratchPadGuard bw_gru_quant_weight_guard_;

  XPUQuantData fw_mul_quant_weight_;
  XPUQuantData bw_mul_quant_weight_;
};

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
