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

class GRUCompute : public KernelLite<TARGET(kXPU), PRECISION(kFloat)> {
  float weight_s1_abs_max = -1;
  float weight_s2_abs_max = -1;

 public:
  void Run() override {
    auto& ctx = this->ctx_->As<XPUContext>();
    auto& param = *param_.get_mutable<operators::GRUParam>();

    bool origin_mode = param.origin_mode;
    bool is_reverse = param.is_reverse;

    auto* input = param.input;
    const float* input_data = input->data<float>();
    auto* h0 = param.h0;
    CHECK_EQ((void*)h0, (void*)nullptr) << "h0 should be nullptr for XPU";

    auto* weight = param.weight;
    const float* weight_data = weight->data<float>();
    auto* bias = param.bias;
    const float* bias_data = bias->data<float>();

    auto* hidden = param.hidden;
    float* hidden_ptr = hidden->mutable_data<float>(TARGET(kXPU));
    const auto& hidden_dims = hidden->dims();
    int frame_size = hidden_dims[1];

    auto& input_lod = input->lod()[0];
    int batch_size = input_lod.size() - 1;
    for (int i = 0; i < batch_size; i++) {
      int cur_seq_len = input_lod[i + 1] - input_lod[i];
      int ret = xdnn::gru_unit_int16(
          ctx.GetRawContext(),              // Context *ctx,
          cur_seq_len,                      // int seq_len,
          frame_size,                       // int frame_size,
          is_reverse,                       // bool is_reverse,
          origin_mode,                      // bool origin_mode,
          const_cast<float*>(input_data),   // float *input, // [seq_len, 3D]
          const_cast<float*>(weight_data),  // float *weight, // [D, 3D]
          weight_s1_abs_max,  // float& weight_s1_abs_max, // [D, 2D]
          weight_s2_abs_max,  // float& weight_s2_abs_max, // [D, D]
          const_cast<float*>(bias_data),  // float *bias, // [1, 3D]
          hidden_ptr);                    // float *hidden // [seq_len, D]
      CHECK_EQ(ret, 0) << "call xdnn::gru_unit_int16 failed!";
      input_data += cur_seq_len * 3 * frame_size;
      hidden_ptr += cur_seq_len * frame_size;
    }
    // batch_gate, batch_reset_hidden_prev lod not set
    hidden->set_lod(input->lod());
  }

  virtual ~GRUCompute() = default;
};

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
