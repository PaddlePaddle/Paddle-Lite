/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <algorithm>

#include "../pe.hpp"
#include "../pe_params.hpp"

namespace paddle_mobile {
namespace zynqmp {

class ConcatPE : public PE {
 public:
  bool init() {
    Tensor* output = param_.output;
    output->setAligned(true);
    return true;
  }

  void apply() {}

  bool dispatch() {
    Tensor* output = param_.output;
    Shape& output_shape = output->shape();
    float16* out_data = param_.output->data<float16>();

    int channel_sum = 0;
    int out_channel = output_shape.channel();
    float scale = 0;
    for (int n = 0; n < param_.inputs.size(); n++) {
      Tensor* input = param_.inputs[n];
      input->invalidate();
      scale = std::max(scale, input->scale()[0]);
      Shape& input_shape = input->shape();
      int wh = output_shape.width() * output_shape.height();
      for (int j = 0; j < wh; j++) {
        float16* src = input->data<float16>() + j * input_shape.channel();
        memcpy(out_data + j * out_channel + channel_sum, src,
               input_shape.channel() * sizeof(float16));
      }
      channel_sum += input_shape.channel();
    }
    output->scale()[0] = scale;
    output->scale()[1] = 1.0f / scale;
    output->flush();
    return true;
  }

  ConcatParam& param() { return param_; }

 private:
  ConcatParam param_;
};

}  // namespace zynqmp
}  // namespace paddle_mobile
