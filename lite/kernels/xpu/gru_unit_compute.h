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
// WIfloatHOUfloat WARRANfloatIES OR CONDIfloatIONS OF ANY KIND, either express
// or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

class GRUUnitCompute : public KernelLite<TARGET(kXPU), PRECISION(kFloat)> {
  float gate_weight_abs_max = -1;
  float state_weight_abs_max = -1;

 public:
  void Run() override {
    auto& ctx = this->ctx_->As<XPUContext>();
    auto& param = *param_.get_mutable<operators::GRUUnitParam>();
    // inputs
    auto input = param.input;
    auto hidden_prev = param.hidden_prev;
    auto weight = param.weight;
    auto bias = param.bias;
    // outputs
    auto hidden = param.hidden;
    // args
    int gate_activation = param.gate_activation;
    CHECK_EQ(gate_activation, 1)
        << "Only support gate_activation=1(sigmoid) but received "
        << gate_activation << " in XPU gru_unit kernel";
    int activation = param.activation;
    CHECK_EQ(activation, 2) << "Only support activation=2(tanh) but received "
                            << activation << " in XPU gru_unit kernel";
    bool origin_mode = param.origin_mode;

    int batch_size = input->dims()[0];
    int frame_size = hidden_prev->dims()[1];

    const float* input_ptr = input->data<float>();
    const float* hidden_prev_ptr = hidden_prev->data<float>();
    const float* weight_ptr = weight->data<float>();
    const float* bias_ptr = (bias == nullptr) ? nullptr : bias->data<float>();

    float* hidden_ptr = hidden->mutable_data<float>(TARGET(kXPU));

    int ret = xdnn::paddle_gru_unit_inference<float, float, int16_t>(
        ctx.GetRawContext(),
        input_ptr,
        hidden_prev_ptr,
        weight_ptr,
        bias_ptr,
        hidden_ptr,
        gate_weight_abs_max,
        state_weight_abs_max,
        batch_size,
        frame_size,
        origin_mode);
    CHECK_EQ(ret, 0) << "call xdnn::paddle_gru_unit failed!";
  }

  virtual ~GRUUnitCompute() = default;
};

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
