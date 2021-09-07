// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include <memory>
#include <string>
#include <vector>
#include "lite/api/paddle_place.h"
#include "lite/core/op_registry.h"
#include "lite/core/optimizer/mir/pass.h"
#include "lite/core/target_wrapper.h"

namespace paddle {
namespace lite {
namespace mir {
/*
 * Use x86_int8_attribute_pass method to update bias val in model.
 * if int8 op has bias, then bias - 128 * weight_scale * in_scale;
 * else int8 op add bias input, and val is (- 128 * weight * weight_scale *
 * in_scale).
 * x86 int8 compute need int8 val and uint8 val to compute
 * out = (in + 128) * weight * in_ scale * wei_scale + (bias - 128 * weight *
 * in_ scale * wei_scale)
 */
class X86Int8AttributePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override;
  // bias_d = conv_bias_d - 128 * weight_scale * input_scale * conv_weight_d
  inline void compute_new_bias(float* bias_d,
                               const int8_t* conv_weight_d,
                               float* conv_bias_d,
                               std::vector<float> weight_scale,
                               std::vector<float> input_scale,
                               int h,
                               int w) {
    for (int i = 0; i < h; i++) {
      auto bias_val = conv_bias_d ? conv_bias_d[i] : 0.f;
      float sum = 0.f;
      float scale = weight_scale[i] * input_scale[0] * 128;
      const int8_t* wei_ptr = conv_weight_d + i * w;
      for (int j = 0; j < w; j++) {
        sum += static_cast<float>(wei_ptr[j]) * scale;
      }
      bias_d[i] = bias_val - sum;
    }
  }

 private:
  std::vector<std::string> int8_ops_{
      "conv2d", "depthwise_conv2d", "conv2d_transpose", "fc"};
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
