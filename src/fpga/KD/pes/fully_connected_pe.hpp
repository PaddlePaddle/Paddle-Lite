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

#include <vector>

#include "../pe.hpp"
#include "../pe_params.hpp"
#include "conv_process.hpp"

namespace paddle_mobile {
namespace zynqmp {

class FullyConnectedPE : public PE {
 public:
  bool init() {
    Tensor* output = param_.output;
    output->setAligned(true);
    output->setDataLocation(Device);
    return true;
  }

  void apply() {
    Tensor* input = param_.input;
    convParam_.input = param_.input;
    convParam_.output = param_.output;
    // convParam_.relu = param_.relu;
    convParam_.groups = 1;
    convParam_.strides = {1, 1};
    convParam_.paddings = {0, 0};
    convParam_.kernelSize = {input->shape().width(), input->shape().height()};
    convParam_.dilations = {1, 1};

    int num = param_.filter->shape().channel();
    int chw = param_.filter->shape().num();

    int height = param_.input->shape().height();
    int width = param_.input->shape().width();
    int filter_channel = chw / height / width;

    int channel = param_.output->shape().channel();
    Shape shape(NCHW, {num, filter_channel, height, width});
    Tensor* conv_filter = new Tensor();
    float* new_filter_data = conv_filter->mutableData<float>(FP32, shape);
    float* filter_data = param_.filter->data<float>();

    for (int i = 0; i < num; i++) {
      for (int j = 0; j < chw; j++) {
        float scale = filter_data[j * num + i];
        new_filter_data[i * chw + j] = scale;
      }
    }

    conv_filter->flush();
    convParam_.filter = conv_filter;

    Shape sb_shape(N, {channel});
    float* scale_data = convParam_.scale()->mutableData<float>(FP32, sb_shape);
    float* bias_data = convParam_.bias()->mutableData<float>(FP32, sb_shape);

    for (int i = 0; i < channel; i++) {
      scale_data[i] = 1.0f;
      bias_data[i] = param_.bias->data<float>()[i];
    }

    fill_split_arg(convParam_);
  }

  bool dispatch() {
    param_.input->syncToDevice();
    int ret = 0;
    std::vector<BasicConvParam*>& params = convParam_.splitParams();
    for (auto conv_param : params) {
      // std::cout << "conv basic \n";
      ret |= compute_fpga_conv_basic(conv_param->args);
    }
    return ret == 0;
  }

  FullyConnectedParam& param() { return param_; }

 private:
  FullyConnectedParam param_;
  ConvParam convParam_;
};
}  // namespace zynqmp
}  // namespace paddle_mobile
