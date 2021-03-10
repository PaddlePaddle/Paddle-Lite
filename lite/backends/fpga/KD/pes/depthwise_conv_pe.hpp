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

#include "lite/backends/fpga/KD/float16.hpp"
#include "lite/backends/fpga/KD/pe.hpp"
#include "lite/backends/fpga/KD/pe_params.hpp"
#include "lite/backends/fpga/KD/pes/conv_process.hpp"

namespace paddle {
namespace zynqmp {

class DepthwiseConvPE : public PE {
 public:
  inline int gcd_(int a, int b) {
    while (b) {
      int temp = a;
      a = b;
      b = temp % b;
    }
    return a;
  }

  inline int lcm_(int a, int b) { return a * b / gcd_(a, b); }

  bool init() {
    Tensor* output = param_.output;
    output->setAligned(true);
    output->setDataLocation(Device);
    return true;
  }

  void apply() {
    DepthwiseConvParam& param = param_;
    Tensor* input = param.input;
    Tensor* output = param.output;
    int channel = output->shape().channel();

    int image_dynamic_range = (1 << 11) - 1;  // int12 max value, pow(2,11)-1
    float16 dynamic_range_fp16 = float_to_half(image_dynamic_range * 1.0);
    float inv_dynamic_range = 1.0 / image_dynamic_range;

    int alignment = 16;

    if (channel % alignment != 0 || channel < alignment) {
      int c_lcm = lcm_(channel, alignment);
      align_repeat_ = c_lcm / (channel);
    }
    Shape shape(N, {2 * channel * align_repeat_});

    float16* b_data = scale_bias_.mutableData<float16>(FP16, shape);
    memset(b_data, 0, scale_bias_.memorySize());

    if (param_.bias()->dataType() == FP32) {
      float* new_bias_data = param_.bias()->data<float>();
      for (int i = 0; i < align_repeat_; i++) {
        for (int j = 0; j < channel; j++) {
          float16 value = float_to_half(new_bias_data[j]);
          b_data[i * channel + j] = value;
        }
      }
    } else {
      float16* new_bias_data = param_.bias()->data<float16>();
      for (int i = 0; i < align_repeat_; i++) {
        for (int j = 0; j < channel; j++) {
          b_data[i * channel + j] = new_bias_data[j];
        }
      }
    }

    if (param_.scale() == nullptr) {
      float16 one = float_to_half(1.0f);
      for (int i = 0; i < align_repeat_; i++) {
        for (int j = 0; j < channel; j++) {
          b_data[channel * align_repeat_ + i * channel + j] = one;
        }
      }
    } else {
      if (param_.scale()->dataType() == FP32) {
        float* new_scale_data = param_.scale()->data<float>();
        for (int i = 0; i < align_repeat_; i++) {
          for (int j = 0; j < channel; j++) {
            float16 value = float_to_half(new_scale_data[j]);
            b_data[channel * align_repeat_ + i * channel + j] = value;
          }
        }
      } else {
        float16* new_scale_data = param_.scale()->data<float16>();
        for (int i = 0; i < align_repeat_; i++) {
          for (int j = 0; j < channel; j++) {
            b_data[channel * align_repeat_ + i * channel + j] =
                new_scale_data[j];
          }
        }
      }
    }
    scale_bias_.flush();

    int filter_dynamic_range = 0;
    Tensor* null_scale = nullptr;

    if (param_.filter->shape().width() == 1 &&
        param_.filter->shape().height() == 1) {
      filter_dynamic_range = (1 << 15) - 1;  // int16 max value, pow(2,15)-1
    } else {
      int fix16_range = (1 << 15) - 1;
      int ext_range = (1 << 19) - 1;
      int max_area =
          static_cast<int>(ext_range / (param_.filter->shape().height() *
                                        param_.filter->shape().width()));
      filter_dynamic_range =
          std::min(max_area, fix16_range);  // int12 max value, pow(2,11)-1
    }

    format_dw_filter(param_.filter,
                     param_.quantizedFilter(),
                     null_scale,
                     filter_dynamic_range);

    DWconvArgs args = {0};
    args.bias_address = b_data;
    args.filter_address = param.quantizedFilter()->data<int16_t>();
    args.filter_scale_address = param_.quantizedFilter()->scale();
    args.kernel.width = param.filter->shape().height();
    args.kernel.height = param.filter->shape().width();
    args.kernel.stride_w = param.strides[0];
    args.kernel.stride_h = param.strides[1];
    args.image.address = input->data<void>();
    args.image.channels = input->shape().channel();
    args.image.height = input->shape().height();
    args.image.width = input->shape().width();
    args.image.pad_width = param.paddings[0];
    args.image.pad_height = param.paddings[1];
    args.image.scale_address = input->max();
    args.output.address = output->data<void>();
    args.output.scale_address = output->max();
    args.out_width = param.output->shape().width();
    args.out_height = param.output->shape().height();
    args.sub_conv_num = 1;
    args.dilation = (param_.dilations[0] <= 1) ? 1 : param_.dilations[0];
    args.inplace.active_param.type = param_.activeParam.type;
    args.inplace.active_param.leaky_relu_factor =
        float_to_half(param_.activeParam.leaky_relu_factor);
    args.quant.dynamic_range =
        *(reinterpret_cast<uint16_t*>(&dynamic_range_fp16));
    args.quant.inv_dynamic_range =
        *(reinterpret_cast<uint32_t*>(&inv_dynamic_range));

    param_.args = args;
  }

  bool dispatch() {
    param_.input->syncToDevice();

    DWconvArgs& args = param_.args;
    if (param_.re_assign) {
      float16* scale_data = scale_bias_.data<float16>();
      int channel = param_.output->shape().channel();
      for (int i = 0; i < align_repeat_; i++) {
        int offset = channel * align_repeat_ + i * channel;
        memcpy(scale_data + offset,
               param_.scale()->data<float16>(),
               param_.scale()->memorySize());
      }
      scale_bias_.flush();
    }
    return compute_fpga_dwconv(param_.args) == 0;
  }

  DepthwiseConvParam& param() { return param_; }

 private:
  DepthwiseConvParam param_;
  Tensor scale_bias_;
  int align_repeat_ = 1;
};

}  // namespace zynqmp
}  // namespace paddle
