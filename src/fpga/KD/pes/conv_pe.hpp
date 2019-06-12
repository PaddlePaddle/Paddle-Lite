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

#include <arm_neon.h>
#include <vector>

#include "../pe.hpp"
#include "../pe_params.hpp"
#include "concat_pe.hpp"
#include "conv_pe.hpp"
#include "conv_process.hpp"
#include "elementwise_add_pe.hpp"
#include "scale_pe.hpp"

namespace paddle_mobile {
namespace zynqmp {

class ConvPE : public PE {
 public:
  bool init() {
    Tensor* output = param_.output;
    output->setAligned(true);
    output->setDataLocation(Device);
    return true;
  }

  void apply() {
    split_axis = fill_split_arg(param_);

    if (split_axis == 0 && param_.splitParams().size() > 1) {
      ConcatParam& concat_param = concatPE_.param();
      for (auto conv_param : param_.splitParams()) {
        concat_param.inputs.push_back(&conv_param->output);
      }
      concat_param.output = param_.output;
      concatPE_.init();
      concatPE_.apply();
    }
  }
  void cpu_compute() {
    Tensor* input = param_.input;
    Tensor* output = param_.output;
    input->syncToCPU();

    Tensor float_input;
    Tensor float_output;
    float* image_addr = float_input.mutableData<float>(FP32, input->shape());
    float_input.copyFrom(input);
    // float16* data_out = output->data<float16>();
    float* out = float_output.mutableData<float>(FP32, output->shape());

    int out_channel = output->shape().channel();
    int in_channel = input->shape().channel();

    float* filter_data = param_.filter->data<float>();
    float* mi = new float[in_channel];

    for (int i = 0; i < out_channel; i++) {
      float* image = image_addr;
      float* filter_ptr = filter_data + i * in_channel;
      float* out_ptr = mi;
      #pragma omp parallel for
      for (int j = 0; j < in_channel; j++) {
        // float32x4_t x0 = vld1q_f32(image);
        // float32x4_t x1 = vld1q_f32(filter_ptr);

        // float32x4_t r = vmulq_f32(x0, x1);

        // vst1q_f32(out_ptr, r);
        // image += 4;
        // filter_ptr += 4;
        // out_ptr += 4;

        float value = image_addr[j] * filter_ptr[j];
        mi[j] = value;
      }

      float sum = 0;
      for (int j = 0; j < in_channel; j++) {
        sum += mi[j];
      }
      out[i] = sum;
    }
    delete[] mi;
    float_output.flush();
    output->copyFrom(&float_output);
  }

  bool dispatch() {
    if (param_.input->shape().width() == 1 &&
        param_.input->shape().channel() < 2048) {
      cpu_compute();
      return true;
    }

    inplace_.relu_enable = param_.relu.enabled;
    inplace_.power_enable = false;
    inplace_.normalize_enable = false;

    if (inplace_.relu_enable) {
      inplace_.relu_enable = param_.relu.enabled;
      config_inplace(inplace_);
    }

    std::vector<BasicConvParam*>& params = param_.splitParams();
    int ret = 0;
    for (auto conv_param : params) {
      // std::cout << "image_scale:\n" ;
      conv_param->input.printScale();
      ret |= compute_fpga_conv_basic(conv_param->args);
    }

    if (inplace_.relu_enable) {
      inplace_.relu_enable = false;
      config_inplace(inplace_);
    }


    size_t size = params.size();
    if (split_axis == 0 && ret == 0 && size > 1) {
      concatPE_.dispatch();
    }
    if (split_axis == 1 && ret == 0 && size > 1) {
      // for (int n = 0; n < size - 1; n++) {
      ElementwiseAddParam& add_param = addPE_.param();
      add_param.inputs = {&params[0]->output, &params[1]->output};
      add_param.output = param_.output;
      addPE_.init();
      addPE_.apply();
      addPE_.dispatch();

      // param_.output->printScale();

      // params[0]->input.saveToFile("conv_1.txt");
      // params[1]->input.saveToFile("conv_2.txt");

      // params[0]->output.saveToFile("ew_o1.txt");
      // params[1]->output.saveToFile("ew_o2.txt");
      // std::cout << "\n ================== EW ================== \n";
      // }
    }
    return ret == 0;
  }

  ConvParam& param() { return param_; }

 private:
  ConvParam param_;
  ConcatPE concatPE_;
  ElementwiseAddPE addPE_;
  int split_axis = 0;
  InplaceArgs inplace_ = {0};
};

}  // namespace zynqmp
}  // namespace paddle_mobile
