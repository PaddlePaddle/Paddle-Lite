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

#include "../llapi/image.h"
#include "../pe.hpp"
#include "../pe_params.hpp"
#include "concat_pe.hpp"
#include "conv_pe.hpp"
#include "conv_process.hpp"

namespace paddle_mobile {
namespace zynqmp {

class ConvPE : public PE {
 public:
  bool init() {
    std::cout << "Conv init" << std::endl;
    return true;
  }

  void apply() {
    // process scale and bias;
    BatchnormParam* bn = param_.batchnorm;
    int channel = param_.output->shape().channel();
    Shape sb_shape(N, {channel});
    float* new_scale_ptr = param_.scale()->mutableData<float>(FP32, sb_shape);
    float* new_bias_ptr = param_.bias()->mutableData<float>(FP32, sb_shape);
    if (bn != nullptr) {
      float* bn_scale_ptr = bn->scale->data<float>();
      float* bn_bias_ptr = bn->bias->data<float>();
      float* bn_var_ptr = bn->variance->data<float>();
      float* bn_mean_ptr = bn->mean->data<float>();
      float epsilon = bn->epsilon;
      for (int i = 0; i < channel; i++) {
        float new_scale =
            bn_scale_ptr[i] /
            static_cast<float>(pow((bn_var_ptr[i] + epsilon), 0.5));
        new_scale_ptr[i] = new_scale;
        new_bias_ptr[i] =
            bn_bias_ptr[i] + (0 - bn_mean_ptr[i]) * new_scale_ptr[i];
      }
    } else {
      for (int i = 0; i < channel; i++) {
        new_scale_ptr[i] = 1.0f;
        new_bias_ptr[i] = 0.0f;
      }
    }
    fill_split_arg(param_);
    if (param_.splitParams().size() > 1) {
      ConcatParam& concat_param = concatPE_.param();
      for (auto conv_param : param_.splitParams()) {
        concat_param.inputs.push_back(&conv_param->output);
      }
      concat_param.output = param_.output;
      concatPE_.init();
      concatPE_.apply();
    }
  }

  bool dispatch() {
    std::vector<BasicConvParam*>& params = param_.splitParams();
    int ret = 0;
    for (auto conv_param : params) {
      ret |= compute_fpga_conv_basic(conv_param->args);
    }
    size_t size = params.size();
    if (ret == 0 && size > 1) {
      concatPE_.dispatch();
    }
    return ret == 0;
  }

  ConvParam& param() { return param_; }

 private:
  ConvParam param_;
  ConcatPE concatPE_;
};

}  // namespace zynqmp
}  // namespace paddle_mobile
