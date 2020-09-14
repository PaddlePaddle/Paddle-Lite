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

    int repeat = 1;
    int alignment = 16;
    int length = channel;

    if (channel % alignment != 0 || channel < alignment) {
      int c_lcm = lcm_(channel, alignment);
      repeat = c_lcm / (channel);
    }
    Shape shape(N, {channel * repeat});

    float16* b_data = bias_.mutableData<float16>(FP16, shape);
    if (param_.bias()->dataType() == FP32) {
      float* new_bias_data = param_.bias()->data<float>();
      for (int i = 0; i < repeat; i++) {
        for (int j = 0; j < length; j++) {
          float16 value = float_to_half(new_bias_data[j]);
          b_data[i * length + j] = value;
        }
      }
      bias_.flush();
    } else {
      float16* new_bias_data = param_.bias()->data<float16>();
      for (int i = 0; i < repeat; i++) {
        for (int j = 0; j < length; j++) {
          b_data[i * length + j] = new_bias_data[j];
        }
      }
      bias_.flush();
    }

    if (param_.scale()->dataType() == FP32) {
      float* new_scale_data = param_.scale()->data<float>();
      Tensor* quantized_filter = param.quantizedFilter();
      quantized_filter->mutableData<float16>(FP16, param.filter->shape());
      format_dw_filter(param.filter, param.quantizedFilter(), new_scale_data);

    } else {
      // TODO(chonwhite) filter fall one and channel aligned case
      float16* scale_data = param_.scale()->data<float16>();
      float16* filter_data = param.quantizedFilter()->mutableData<float16>(
          FP16, param.filter->shape());
      memcpy(filter_data,
             scale_data,
             param.filter->shape().numel() * sizeof(float16));
      param.quantizedFilter()->flush();
    }

    DWconvArgs args = {0};
    args.bias_address = b_data;
    args.filter_address = param.quantizedFilter()->data<void>();
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
    args.image.scale_address = input->scale();
    args.output.address = output->data<void>();
    args.output.scale_address = output->scale();
    args.out_width = param.output->shape().width();
    args.out_height = param.output->shape().height();
    args.sub_conv_num = 1;
    param.args = args;

    inplace_.power_enable = false;
    inplace_.normalize_enable = false;
  }

  bool dispatch() {
    param_.input->syncToDevice();
    if (param_.activeParam.type == TYPE_RELU) {
      inplace_.relu_enable = true;
    } else if (param_.activeParam.type == TYPE_RELU6) {
      inplace_.relu6_enable = true;
    } else if (param_.activeParam.type == TYPE_SIGMOID) {
      inplace_.sigmoid_enable = true;
    } else if (param_.activeParam.type == TYPE_LEAKY_RELU) {
      inplace_.leaky_relu_enable = true;
    }

    if (inplace_.relu_enable || inplace_.leaky_relu_enable ||
        inplace_.relu6_enable || inplace_.sigmoid_enable) {
      config_inplace(inplace_);
    }
    bool ret = compute_fpga_dwconv(param_.args) == 0;
    if (inplace_.relu_enable || inplace_.leaky_relu_enable ||
        inplace_.relu6_enable || inplace_.sigmoid_enable) {
      inplace_.relu_enable = false;
      inplace_.leaky_relu_enable = false;
      inplace_.relu6_enable = false;
      inplace_.sigmoid_enable = false;
      config_inplace(inplace_);
    }
    return ret;
  }

  DepthwiseConvParam& param() { return param_; }

 private:
  DepthwiseConvParam param_;
  Tensor bias_;
  InplaceArgs inplace_ = {0};
};

}  // namespace zynqmp
}  // namespace paddle
