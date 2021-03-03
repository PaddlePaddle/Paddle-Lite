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
class ResizePE : public PE {
 public:
  bool init() {
    Tensor* output = param_.output;
    output->setAligned(true);
    output->setDataLocation(Device);
    return true;
  }

  void apply() {
    Tensor* input = param_.input;
    Tensor* output = param_.output;
    ResizeArgs& args = args_;

    int input_width = input->shape().width();
    int input_height = input->shape().height();
    int input_channel = input->shape().channel();

    int output_width = output->shape().width();
    int output_height = output->shape().height();

    args.input_width = input_width;
    args.input_height = input_height;
    args.image_channel = input_channel;
    args.output_width = output_width;
    args.output_height = output_height;
    float height_ratio = static_cast<float>(input_height) /
                         static_cast<float>(args.output_height);
    float width_ratio =
        static_cast<float>(input_width) / static_cast<float>(args.output_width);
    args.height_ratio = *reinterpret_cast<uint32_t*>(&height_ratio);
    args.width_ratio = *reinterpret_cast<uint32_t*>(&width_ratio);

    args.input_image_address = input->mutableData<void>();
    args.output_image_address = output->mutableData<void>();
    args.output_scale_address = reinterpret_cast<uint16_t*>(output->max());
  }

  void compute_scale(Tensor* src, float* scale) {
    float16* data = src->data<float16>();
    src->invalidate();
    float max = 0;
    for (int i = 0; i < src->shape().numel(); i++) {
      float value = half_to_float(data[i]);
      if (value < 0) {
        value = -value;
      }
      if (value > max) {
        max = value;
      }
    }
    scale[0] = max / 127.0;
    scale[1] = 127.0 / max;
  }
  void cpu_compute() {
    Shape& in_shape = param_.input->shape();
    Shape& out_shape = param_.output->shape();
    int channel = in_shape.channel();
    int in_height = in_shape.height();
    int in_width = in_shape.width();
    int out_width = out_shape.width();
    int factor = out_shape.width() / in_shape.width();

    param_.input->syncToCPU();

    for (int h = 0; h < in_height; h++) {
      for (int w = 0; w < in_width; w++) {
        int src_index = in_width * channel * h + w * channel;
        float16* src = param_.input->data<float16>() + src_index;
        for (int v = 0; v < factor; v++) {
          for (int i = 0; i < factor; i++) {
            int dst_index = out_width * channel * h * factor +
                            out_width * channel * v + w * channel * factor +
                            channel * i;
            float16* dst = param_.output->data<float16>() + dst_index;
            memcpy(dst, src, channel * sizeof(float16));
          }
        }
      }
    }
    param_.output->flush();
    param_.output->copyScaleFrom(param_.input);
    param_.output->copyMaxFrom(param_.input);
  }

  bool dispatch() {
    cpu_compute();
    return true;
  }

  ResizeParam& param() { return param_; }

 private:
  ResizeParam param_;
  ResizeArgs args_;
};
}  // namespace zynqmp
}  // namespace paddle
