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

    if (split_axis == 0 && param_.splitParams().size() > 1) {
      ConcatParam& concat_param = concatPE_.param();
      for (auto conv_param : param_.splitParams()) {
        concat_param.inputs.push_back(&conv_param->output);
      }
      concat_param.output = param_.output;
      concatPE_.init();
      concatPE_.apply();
    }

    if (DLEngine::get_instance().isZU3() &&
        param_.input->shape().dimSize() == 4 &&
        param_.input->shape().width() == 1 &&
        param_.input->shape().width() >= 2048) {
      use_cpu_ = true;
    }

    if (param_.filter->shape().width() == 1 &&
        param_.filter->shape().height() == 1) {  // NOLINT
    }
    if (!use_cpu_) {  // NOLINT
    }
  }

  void cpu_conv_hwc() {
    Tensor* input = param_.input;
    Tensor* output = param_.output;
    input->syncToCPU();

    Tensor float_input;
    Tensor float_output;
    float* image_addr = float_input.mutableData<float>(FP32, input->shape());
    float_input.copyFrom(input);
    float_input.syncToCPU();
    float* out = float_output.mutableData<float>(FP32, output->shape());

    int out_width = output->shape().width();
    int out_channel = output->shape().channel();
    int in_channel = input->shape().channel();

    float* filter_data = param_.filter->data<float>();

    int image_height = input->shape().height();
    int image_width = input->shape().width();
    int image_channels = input->shape().channel();
    int image_pad_h = param_.paddings[0];
    int image_pad_w = param_.paddings[1];
    int kernel_height = param_.filter->shape().height();
    int kernel_width = param_.filter->shape().width();
    int kernel_step_h = param_.strides[0];
    int kernel_step_w = param_.strides[1];
    int pooled_height_ = output->shape().height();
    int pooled_width_ = out_width;
    int filter_chw = image_channels * kernel_height * kernel_width;

    float max = 0;

    for (int ph = 0; ph < pooled_height_; ph++) {
      for (int pw = 0; pw < pooled_width_; pw++) {
        int hstart = ph * kernel_step_h - image_pad_h;
        int wstart = pw * kernel_step_w - image_pad_w;
        int hend =
            std::min(hstart + kernel_height, static_cast<int>(image_height));
        int wend =
            std::min(wstart + kernel_width, static_cast<int>(image_width));
        hstart = std::max(hstart, static_cast<int>(0));
        wstart = std::max(wstart, static_cast<int>(0));
        for (int oc = 0; oc < out_channel; oc++) {
          float sum = 0.0f;
          const int pool_index = (ph * pooled_width_ + pw) * out_channel + oc;
          for (int c = 0; c < image_channels; c++) {
            for (int h = hstart; h < hend; h++) {
              int hi = 0;
              if (ph == 0) {
                hi = h - hstart + image_pad_h;
              } else {
                hi = h - hstart;
              }
              for (int w = wstart; w < wend; w++) {
                int wi = 0;
                if (pw == 0) {
                  wi = w - wstart + image_pad_w;
                } else {
                  wi = w - wstart;
                }
                const int index = (h * image_width + w) * image_channels + c;
                int weight_index = oc * filter_chw +
                                   kernel_width * kernel_height * c +
                                   kernel_width * hi + wi;
                float value = image_addr[index] * filter_data[weight_index];
                sum += value;
              }
            }
          }

          if (param_.relu.enabled && sum < 0) {
            sum = -sum;
          }
          if (sum > max) {
            max = sum;
          }
          out[pool_index] = sum;
        }
      }
    }
    float_output.flush();
    output->copyFrom(&float_output);
    output->scale()[0] = max / 127;
    output->scale()[1] = 127 / max;
  }

  void cpu_compute() {
    Tensor* input = param_.input;
    Tensor* output = param_.output;
    input->syncToCPU();

    Tensor float_input;
    Tensor float_output;
    float* image_addr = float_input.mutableData<float>(FP32, input->shape());
    float_input.copyFrom(input);
    float_input.syncToCPU();

    float* out = float_output.mutableData<float>(FP32, output->shape());

    float* bias_data = param_.bias()->data<float>();

    int out_width = output->shape().width();
    int out_channel = output->shape().channel();
    int in_channel = input->shape().channel();

    float* filter_data = param_.filter->data<float>();
    float* mi = new float[in_channel];
    float max = 0;

    int out_index = 0;
    for (int i = 0; i < out_channel; i++) {
      float* image = image_addr;
      float* filter_ptr = filter_data + i * in_channel;
      float* out_ptr = mi;

      for (int h = 0; h < output->shape().height(); h++) {
        for (int w = 0; w < output->shape().width(); w++) {
          float sum = 0;

          // #pragma omp parallel for
          for (int j = 0; j < in_channel; j++) {
            int image_index = h * out_width * in_channel + w * in_channel + j;
            float value = image_addr[image_index] * filter_ptr[j];
            sum += value;
          }

          sum += bias_data[i];

          if (param_.relu.enabled && sum < 0) {
            sum = 0;
          }
          if (sum > max) {
            max = sum;
          }
          out_index = h * out_width * out_channel + w * out_channel + i;
          out[out_index] = sum;
        }
      }
    }
    delete[] mi;
    float_output.flush();
    output->copyFrom(&float_output);
    output->scale()[0] = max / 127;
    output->scale()[1] = 127 / max;
  }

  bool dispatch() {
    if (use_cpu_) {
      cpu_compute();
      return true;
    }

    inplace_.leaky_relu_enable =
        (param_.relu.leaky_relu_factor != 0) ? true : false;
    inplace_.relu_enable =
        inplace_.leaky_relu_enable ? false : param_.relu.enabled;

    inplace_.power_enable = false;
    inplace_.normalize_enable = false;
    if (inplace_.relu_enable || inplace_.leaky_relu_enable) {
      config_inplace(inplace_);
      if (inplace_.leaky_relu_enable) {
        activeParamterArgs.type = TYPE_LEAK_RELU;
        activeParamterArgs.leaky_relu_factor =
            fp32_2_fp16(param_.relu.leaky_relu_factor);
        config_activation(activeParamterArgs);
      }
    }

    std::vector<BasicConvParam*>& params = param_.splitParams();
    int ret = 0;
    for (auto conv_param : params) {
      ret |= compute_fpga_conv_basic(conv_param->args);
    }

    if (inplace_.relu_enable || inplace_.leaky_relu_enable) {
      inplace_.relu_enable = false;
      inplace_.leaky_relu_enable = false;
      config_inplace(inplace_);

      if (inplace_.leaky_relu_enable) {
        activeParamterArgs.type = TYPE_LEAK_RELU;
        activeParamterArgs.leaky_relu_factor = fp32_2_fp16(0);
        config_activation(activeParamterArgs);
      }
    }

    size_t size = params.size();
    if (split_axis == 0 && ret == 0 && size > 1) {
      concatPE_.dispatch();
    }
    if (split_axis == 1 && ret == 0 && size > 1) {
      ElementwiseAddParam& add_param = addPE_.param();
      add_param.inputs = {&params[0]->output, &params[1]->output};
      add_param.output = param_.output;
      addPE_.init();
      addPE_.apply();
      addPE_.dispatch();
    }
    return ret == 0;
  }

  ConvParam& param() { return param_; }

 private:
  bool use_cpu_ = false;
  ConvParam param_;
  ConcatPE concatPE_;
  ElementwiseAddPE addPE_;
  int split_axis = 0;
  InplaceArgs inplace_ = {0};
  ActiveParamterArgs activeParamterArgs;
};

}  // namespace zynqmp
}  // namespace paddle
