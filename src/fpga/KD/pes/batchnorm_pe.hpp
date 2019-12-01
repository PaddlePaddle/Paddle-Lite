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

#include <fstream>
#include <iostream>

#include "../pe.hpp"
#include "../pe_params.hpp"
#include "scale_pe.hpp"

namespace paddle_mobile {
namespace zynqmp {
class BatchnormPE : public PE {
 public:
  bool init() {
    Tensor* output = param_.output;
    output->setAligned(true);
    output->setDataLocation(Device);

    ScaleParam& scale_param = scalePE_.param();
    scale_param.input = param_.input;
    scale_param.output = param_.output;
    Tensor* scale = new Tensor();
    Tensor* bias = new Tensor();
    Shape shape(N, {output->shape().channel()});

    auto mean_data = param_.mean->data<float>();
    auto variance_data = param_.variance->data<float>();
    auto scale_data = param_.scale->data<float>();
    auto bias_data = param_.bias->data<float>();
    auto new_scale_ptr = scale->mutableData<float>(FP32, shape);
    auto new_bias_ptr = bias->mutableData<float>(FP32, shape);

    float epsilon = param_.epsilon;

    Shape& in_shape = param_.input->shape();
    bool match = in_shape.channel() == 128 && in_shape.height() == 128 &&
                 in_shape.width() == 128;

    for (int c = 0; c < output->shape().channel(); c++) {
      float var = variance_data[c];
      float inv_scale = 1.0 / (std::sqrt(var + epsilon));
      float scale_value = inv_scale * scale_data[c];
      float bias_value = bias_data[c] - scale_value * mean_data[c];
      new_scale_ptr[c] = scale_value;
      new_bias_ptr[c] = bias_value;
    }

    scale->flush();
    bias->flush();

    scale_param.scale = scale;
    scale_param.bias = bias;
    // scale_param.relu = param_.relu;

    scalePE_.init();

    // inplace_.relu_enable = param_.relu.enabled;
    // inplace_.relu_enable = true;
    inplace_.power_enable = false;
    inplace_.normalize_enable = false;

    return true;
  }

  void apply() {
    scalePE_.apply();
    // delete param_.scale;
    // delete param_.bias;
    // param_.scale = nullptr;
    // param_.bias = nullptr;
  }

  bool dispatch() {
    if (param_.activeParam.type = TYPE_RELU) {
      inplace_.relu_enable = true;
    } else if (param_.activeParam.type = TYPE_RELU6) {
      inplace_.relu6_enable = true;
    } else if (param_.activeParam.type = TYPE_SIGMOID) {
      inplace_.sigmoid_enable = true;
    } else if (param_.activeParam.type = TYPE_LEAKY_RELU) {
      inplace_.leaky_relu_enable = true;
    }

    if (inplace_.relu_enable || inplace_.leaky_relu_enable || 
      inplace_.relu6_enable || inplace_.sigmoid_enable) {
      config_inplace(inplace_);
    }
    bool ret = scalePE_.dispatch();

    // bool ret = cpu_compute();
    // inplace_.relu_enable = false;
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

  BatchnormParam& param() { return param_; }

  ~BatchnormPE() {
    scalePE_.param().input = nullptr;
    scalePE_.param().output = nullptr;
  }

 private:
  BatchnormParam param_;
  ScalePE scalePE_;
  InplaceArgs inplace_;
};
}  // namespace zynqmp
}  // namespace paddle_mobile
