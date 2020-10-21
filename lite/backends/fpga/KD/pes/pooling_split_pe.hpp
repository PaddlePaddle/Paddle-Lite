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
#include <vector>

#include "lite/backends/fpga/KD/pe.hpp"
#include "lite/backends/fpga/KD/pe_params.hpp"
#include "lite/backends/fpga/KD/pes/concat_pe.hpp"
#include "lite/backends/fpga/KD/pes/elementwise_add_pe.hpp"
#include "lite/backends/fpga/KD/pes/pooling_process.hpp"
#include "lite/backends/fpga/KD/pes/scale_pe.hpp"
#include "lite/backends/fpga/KD/pes/split_pe.hpp"

namespace paddle {
namespace zynqmp {

class PoolingSplitPE : public PE {
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
    PoolingParam& param = param_;
    Tensor* input = param.input;
    Tensor* output = param.output;
    int channel = output->shape().channel();

    int k_height = param_.kernelSize[1];
    int k_width = param_.kernelSize[0];

    if (param_.globalPooling) {
      k_width = input->shape().width();
      k_height = input->shape().height();
      param_.kernelSize[0] = k_height;
      param_.kernelSize[1] = k_width;
    } else {
      k_height = param_.kernelSize[0];
      k_width = param_.kernelSize[1];
    }

    use_cpu_ = output->shape().width() == 1 && output->shape().height() == 1 &&
               (k_width > 255 || k_height > 255);

    if (use_cpu_) {
      return;
    }

    pooling_split_channel(param, splitParams_);

    if (splitParams_.size() > 1) {
      SplitParam& split_param = splitPE_.param();
      split_param.input = param_.input;
      for (auto pooling_param : splitParams_) {
        split_param.outputs.push_back(pooling_param->input);
      }
      splitPE_.init();
      splitPE_.apply();

      ConcatParam& concat_param = concatPE_.param();
      for (auto pooling_param : splitParams_) {
        concat_param.inputs.push_back(pooling_param->output);
      }
      concat_param.output = param_.output;
      concatPE_.init();
      concatPE_.apply();
    }
  }

  void compute() {
    Tensor* input = param_.input;
    Tensor* output = param_.output;
    input->syncToCPU();

    Tensor float_input;
    // Tensor float_output;
    float* image_addr = float_input.mutableData<float>(FP32, input->shape());
    float_input.copyFrom(input);
    float16* data_out = output->data<float16>();

    int image_height = input->shape().height();
    int image_width = input->shape().width();
    int image_channels = input->shape().channel();
    int image_pad_h = param_.paddings[0];
    int image_pad_w = param_.paddings[1];
    int kernel_height = param_.kernelSize[1];
    int kernel_width = param_.kernelSize[0];
    int kernel_step_h = param_.strides[0];
    int kernel_step_w = param_.strides[1];

    int pooled_height_ = output->shape().height();
    int pooled_width_ = output->shape().width();

    int kernel = kernel_height * kernel_width;

    float max = 0;

    for (int ph = 0; ph < pooled_height_; ++ph) {
      for (int pw = 0; pw < pooled_width_; ++pw) {
        int hstart = ph * kernel_step_h - image_pad_h;
        int wstart = pw * kernel_step_w - image_pad_w;
        int hend = std::min(hstart + kernel_height, image_height);
        int wend = std::min(wstart + kernel_width, image_width);
        hstart = std::max(hstart, 0);
        wstart = std::max(wstart, 0);

        kernel = (hend - hstart) * (wend - wstart);
        for (int c = 0; c < image_channels; ++c) {
          const int pool_index = (ph * pooled_width_ + pw) * image_channels + c;
          float sum = 0;

          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              const int index = (h * image_width + w) * image_channels + c;
              float value = image_addr[index];
              // ofs_out << value << std::endl;
              sum += value;
            }
          }

          float value = sum / kernel;
          if (value > max) {
            max = value;
          }
          data_out[pool_index] = float_to_half(value);
        }
      }
    }
    output->scale()[0] = max / 127.0f;
    output->scale()[1] = 127.0f / max;
    output->flush();
  }

  bool dispatch() {
    Tensor* output = param_.output;
    param_.input->syncToDevice();

    if (use_cpu_) {
      compute();
      return true;
    }

    if (splitParams_.size() > 1) {
      splitPE_.dispatch();
    }

    int ret = 0;
    int index = 0;

    InplaceArgs inplace_ = {0};
    GlobalPoolArgs globalPoolArgs;
    if (param_.globalPooling) {
      inplace_.relu_enable = false;
      inplace_.leaky_relu_enable = false;
      inplace_.relu6_enable = false;
      inplace_.sigmoid_enable = false;
      inplace_.global_pool_en = true;
      config_inplace(inplace_);

      int kernel_height = param_.kernelSize[1];
      int kernel_width = param_.kernelSize[0];
      globalPoolArgs.global_pool_factor =
          fp32_2_fp16(1.0f / (kernel_height * kernel_width));
      config_global_pool(globalPoolArgs);
    }
    for (auto pooling_param : splitParams_) {
      ret |= compute_fpga_pool(pooling_param->poolingArgs);

      float* scale_address = pooling_param->poolingArgs.output.scale_address;
      output->scale()[0] = scale_address[0];
      output->scale()[1] = scale_address[1];
    }

    if (param_.globalPooling) {
      inplace_.relu_enable = false;
      inplace_.leaky_relu_enable = false;
      inplace_.relu6_enable = false;
      inplace_.sigmoid_enable = false;
      inplace_.global_pool_en = false;
      config_inplace(inplace_);
      globalPoolArgs.global_pool_factor = fp32_2_fp16(1.0f);
      config_global_pool(globalPoolArgs);
    }

    if (splitParams_.size() > 1) {
      concatPE_.dispatch();
    }

    return ret;
  }

  ~PoolingSplitPE() {
    for (auto pooling_param : splitParams_) {
      if (splitParams_.size() > 1) {
        delete pooling_param->input;
        delete pooling_param->output;
        delete pooling_param;
      }
    }
    splitParams_.clear();
  }

  PoolingParam& param() { return param_; }

 private:
  PoolingParam param_;
  ConcatPE concatPE_;
  SplitPE splitPE_;
  std::vector<PoolingParam*> splitParams_;
  bool use_cpu_ = false;
};

}  // namespace zynqmp
}  // namespace paddle
