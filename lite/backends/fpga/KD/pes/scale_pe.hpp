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

#include "lite/backends/fpga/KD/pe.hpp"
#include "lite/backends/fpga/KD/pe_params.hpp"

namespace paddle {
namespace zynqmp {
class ScalePE : public PE {
 public:
  inline int gcd(int a, int b) {
    while (b) {
      int temp = a;
      a = b;
      b = temp % b;
    }
    return a;
  }

  inline int lcm(int a, int b) { return a * b / gcd(a, b); }
  bool init() {
    Tensor* output = param_.output;
    output->setAligned(true);
    output->setDataLocation(Device);
    return true;
  }

  void apply() {
    Tensor* input = param_.input;
    Tensor* output = param_.output;
    Shape& input_shape = input->shape();
    int channel = input_shape.channel();
    int repeat = 1;
    int alignment = 16;
    int length = channel;

    if (channel % alignment != 0 || channel < alignment) {
      int c_lcm = lcm(channel, alignment);
      repeat = c_lcm / (channel);
    }
    Shape shape(N, {channel * repeat});
    param_.alignedBias()->mutableData<float16>(FP16, shape);
    param_.alignedScale()->mutableData<float16>(FP16, shape);

    float16* bias_data = param_.alignedBias()->data<float16>();
    float16* scale_data = param_.alignedScale()->data<float16>();

    if (param_.bias != nullptr) {
      float* bias_data_float = param_.bias->data<float>();
      for (int i = 0; i < repeat; i++) {
        for (int j = 0; j < length; j++) {
          float16 value = float_to_half(bias_data_float[j]);
          bias_data[i * length + j] = value;
        }
      }
    } else {
      float16 zero = float_to_half(0.0f);
      for (int i = 0; i < repeat; i++) {
        for (int j = 0; j < length; j++) {
          bias_data[i * length + j] = zero;
        }
      }
    }

    float* scale_data_float = param_.scale->data<float>();
    for (int i = 0; i < repeat; i++) {
      for (int j = 0; j < length; j++) {
        float16 value = float_to_half(scale_data_float[j]);
        scale_data[i * length + j] = value;
      }
    }

    param_.alignedScale()->flush();
    param_.alignedBias()->flush();

    int wc = input_shape.width() * input_shape.channel();
    int wc_aligned = align_image(wc);

    ScaleArgs& args = param_.args;
    args.scale_address = param_.alignedScale()->data<void>();
    args.bias_address = param_.alignedBias()->data<void>();
    args.wc_alignment = wc_aligned;
    args.channel_alignment = channel * repeat;

    args.image.address = input->data<void>();
    args.image.scale_address = input->scale();
    args.image.channels = channel;
    args.image.height = input_shape.height();
    args.image.width = input_shape.width();
    args.image.pad_width = 0;
    args.image.pad_height = 0;
    args.output.address = output->data<void>();
    args.output.scale_address = output->scale();
  }

  bool dispatch() {
    param_.input->syncToDevice();
    return compute_fpga_scale(param_.args) == 0;
  }

  ScaleParam& param() { return param_; }

 private:
  ScaleParam param_;
};
}  // namespace zynqmp
}  // namespace paddle
