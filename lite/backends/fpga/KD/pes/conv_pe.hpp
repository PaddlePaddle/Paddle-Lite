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
#include <algorithm>
#include <vector>

#include "lite/backends/fpga/KD/pe.hpp"
#include "lite/backends/fpga/KD/pe_params.hpp"
#include "lite/backends/fpga/KD/pes/concat_pe.hpp"
#include "lite/backends/fpga/KD/pes/conv_pe.hpp"
#include "lite/backends/fpga/KD/pes/conv_process.hpp"
#include "lite/backends/fpga/KD/pes/elementwise_add_pe.hpp"
#include "lite/backends/fpga/KD/pes/scale_pe.hpp"
#include "lite/backends/fpga/KD/pes/split_pe.hpp"

namespace paddle {
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

    split_channel = param_.groups != 1 && param_.splitParams().size() > 1;

    if (split_axis == 0 && param_.splitParams().size() > 1) {
      ConcatParam& concat_param = concatPE_.param();
      for (auto conv_param : param_.splitParams()) {
        concat_param.inputs.push_back(&conv_param->output);
      }
      concat_param.output = param_.output;
      concatPE_.init();
      concatPE_.apply();
    }

    if (split_channel) {
      SplitParam& split_param = splitPE_.param();
      split_param.input = param_.input;
      for (auto conv_param : param_.splitParams()) {
        split_param.outputs.push_back(&conv_param->input);
      }
      splitPE_.init();
      splitPE_.apply();
    }

    if (DLEngine::get_instance().isZU3() &&
        param_.input->shape().dimSize() == 4 &&
        param_.input->shape().width() == 1 &&
        param_.input->shape().channel() >= 2048) {
      use_cpu_ = true;
    }
    if (!use_cpu_) {
      // param_.filter->releaseData();
    }

    // exit(-1);
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
    fpga_reset();
    if (use_cpu_) {
      cpu_compute();
      return true;
    }

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
      if (inplace_.leaky_relu_enable) {
        activeParamterArgs.type = TYPE_LEAKY_RELU;
        activeParamterArgs.leaky_relu_factor =
            fp32_2_fp16(param_.activeParam.leaky_relu_factor);
        config_activation(activeParamterArgs);
      }
    }

    std::vector<BasicConvParam*>& params = param_.splitParams();

    if (split_channel) {
      // splitPE_.param().input->saveToFile("input_image",true);
      splitPE_.dispatch();
    }

    int ret = 0;
    for (auto conv_param : params) {
      // conv_param->input.printScale();
      // if (split_channel) {
      //   conv_param->input.saveToFile("pack_image",true);
      // }
      ret |= compute_fpga_conv_basic(conv_param->args);
    }

    if (inplace_.relu_enable || inplace_.leaky_relu_enable ||
        inplace_.relu6_enable || inplace_.sigmoid_enable) {
      inplace_.relu_enable = false;
      inplace_.leaky_relu_enable = false;
      inplace_.relu6_enable = false;
      inplace_.sigmoid_enable = false;
      config_inplace(inplace_);

      if (inplace_.leaky_relu_enable) {
        activeParamterArgs.type = TYPE_LEAKY_RELU;
        activeParamterArgs.leaky_relu_factor = fp32_2_fp16(0);
        config_activation(activeParamterArgs);
      }
    }

    size_t size = params.size();
    if (split_axis == 0 && ret == 0 && size > 1) {
      // std::cout << "concat size:" << size << std::endl;
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
  bool use_cpu_ = false;
  bool split_channel = false;
  ConvParam param_;
  ConcatPE concatPE_;
  SplitPE splitPE_;
  ElementwiseAddPE addPE_;
  int split_axis = 0;
  InplaceArgs inplace_ = {0};
  ActiveParamterArgs activeParamterArgs;
};

}  // namespace zynqmp
}  // namespace paddle
