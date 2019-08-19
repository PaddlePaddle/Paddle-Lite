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

#include "../pe.hpp"
#include "../pe_params.hpp"

namespace paddle_mobile {
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

    if (param_.scale->dataType() == FP32) {
      if (param_.bias != nullptr) {
        float* bias_data_float = param_.bias->data<float>();
        for (int i = 0; i < repeat; i++) {
          for (int j = 0; j < length; j++) {
            float16 value = float_to_half(bias_data_float[j]);
            bias_data[i * length + j] = value;
            // bias_data[i * length + j] = float_to_half(1.0f);
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
    } else {
      if (param_.bias != nullptr) {
        float16* bias_data_float = param_.bias->data<float16>();
        for (int i = 0; i < repeat; i++) {
          for (int j = 0; j < length; j++) {
            float16 value = bias_data_float[j];
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

      float16* scale_data_float = param_.scale->data<float16>();
      for (int i = 0; i < repeat; i++) {
        for (int j = 0; j < length; j++) {
          float16 value = scale_data_float[j];
          scale_data[i * length + j] = value;
        }
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

  void cpu_compute() {
    Tensor* input = param_.input;
    Tensor* output = param_.output;
    Tensor float_input;
    float* image_addr = float_input.mutableData<float>(FP32, input->shape());
    input->syncToCPU();
    float_input.copyFrom(input);
    float16* data_out = output->data<float16>();

    float* scale_data = param_.scale->data<float>();

    int wh = input->shape().width() * input->shape().height();

    float max = 0;

    for (int i = 0; i < wh; i++) {
      for (int c = 0; c < input->shape().channel(); c++) {
        int index = i * input->shape().channel() + c;
        float value = image_addr[index] * scale_data[c];
        data_out[index] = float_to_half(value);

        if (value < 0) {
          value = -value;
        }
        if (value > max) {
          max = value;
        }
      }
    }
    output->flush();
    // std::cout << "max:" << max << std::endl;
    output->scale()[0] = max / 127.0f;
    output->scale()[1] = 127.0f / max;
  }

  bool dispatch() {
    // std::cout << "data_type:" << param_.scale->dataType() << std::endl;
    if (param_.scale->dataType() == FP16) {
      // param_.scale->saveToFile("ps.txt");
      apply();
    }
    // param_.scale->saveToFile("scale.txt");
    // cpu_compute();
    // return true;
    // param_.input->syncToDevice();
    return compute_fpga_scale(param_.args) == 0;
  }

  ScaleParam& param() { return param_; }

 private:
  ScaleParam param_;
};
}  // namespace zynqmp
}  // namespace paddle_mobile
