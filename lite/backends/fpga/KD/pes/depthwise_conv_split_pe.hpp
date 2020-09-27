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

#include "lite/backends/fpga/KD/float16.hpp"
#include "lite/backends/fpga/KD/pe.hpp"
#include "lite/backends/fpga/KD/pe_params.hpp"
#include "lite/backends/fpga/KD/pes/conv_process.hpp"

namespace paddle {
namespace zynqmp {

class DepthwiseConvSplitPE : public PE {
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
    DepthwiseConvSplitParam& param = param_;
    Tensor* input = param.input;
    Tensor* output = param.output;
    int channel = output->shape().channel();

    dwconv_split_channel(param);

    if (param.splitParams().size() > 1) {
      SplitParam& split_param = splitPE_.param();
      split_param.input = param_.input;
      for (auto dwconv_param : param_.splitParams()) {
        split_param.outputs.push_back(&dwconv_param->input);
      }
      splitPE_.init();
      splitPE_.apply();

      ConcatParam& concat_param = concatPE_.param();
      for (auto dwconv_param : param_.splitParams()) {
        concat_param.inputs.push_back(&dwconv_param->output);
      }
      concat_param.output = param_.output;
      concatPE_.init();
      concatPE_.apply();
    }
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

    std::vector<BasicDWConvParam*>& params = param_.splitParams();

    if (params.size() > 1) {
      splitPE_.dispatch();
    }

    int ret = 0;
    for (auto dwconv_param : params) {
      ret |= compute_fpga_dwconv(dwconv_param->args);
    }

    if (params.size() > 1) {
      concatPE_.dispatch();
    }

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

  DepthwiseConvSplitParam& param() { return param_; }

 private:
  DepthwiseConvSplitParam param_;
  ConcatPE concatPE_;
  SplitPE splitPE_;
  InplaceArgs inplace_ = {0};
};

}  // namespace zynqmp
}  // namespace paddle
