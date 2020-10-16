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

#include "lite/backends/fpga/KD/pe.hpp"
#include "lite/backends/fpga/KD/pe_params.hpp"

namespace paddle {
namespace zynqmp {

class PoolingPE : public PE {
 public:
  bool init() {
    Tensor* output = param_.output;
    output->setAligned(true);
    output->setDataLocation(Device);
    return true;
  }

  void apply() {
    Tensor* input = param_.input;
    Tensor* output = param_.output;

    uint32_t k_height = 1;
    uint32_t k_width = 1;

    if (param_.globalPooling) {
      k_width = input->shape().width();
      k_height = input->shape().height();
      param_.kernelSize[0] = k_height;
      param_.kernelSize[1] = k_width;
    } else {
      k_height = param_.kernelSize[0];
      k_width = param_.kernelSize[1];
    }

    PoolingArgs args = {0};
    args.mode = param_.type;
    if (param_.globalPooling) {
      args.kernel_reciprocal = fp32_2_fp16(1.0f);
    } else {
      args.kernel_reciprocal = fp32_2_fp16(1.0f / (k_width * k_height));
    }
    args.image.address = input->data<float16>();
    args.image.channels = input->shape().channel();
    args.image.height = input->shape().height();
    args.image.width = input->shape().width();
    args.image.pad_height = param_.paddings[0];
    args.image.pad_width = param_.paddings[1];
    args.image.scale_address = input->scale();
    args.output.address = output->mutableData<float16>();
    args.output.scale_address = output->scale();
    args.kernel.height = k_height;
    args.kernel.width = k_width;
    args.kernel.stride_h = param_.strides[0];
    args.kernel.stride_w = param_.strides[1];
    args.out_height = output->shape().height();
    args.out_width = output->shape().width();
    param_.poolingArgs = args;

    use_cpu_ = output->shape().width() == 1 && output->shape().height() == 1 &&
               (k_width > 255 || k_height > 255);
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

  void cpu_compute1() {
    Tensor* input = param_.input;
    Tensor* output = param_.output;
    input->syncToCPU();

    Tensor float_input;
    float_input.mutableData<float>(FP32, input->shape());
    float_input.copyFrom(input);
    float16* data_out = output->data<float16>();

    int kernel_hw = param_.kernelSize[0] * param_.kernelSize[1];

    float scale_max = 0;
    for (int i = 0; i < output->shape().channel(); i++) {
      float sum = 0;
      for (int j = 0; j < kernel_hw; j++) {
        float value = half_to_float(input->data<float16>()[i * kernel_hw + j]);
        sum += value;
      }
      float value = sum / kernel_hw;
      data_out[i] = float_to_half(value);
      scale_max = std::max(scale_max, std::abs(value));
    }
    output->scale()[0] = scale_max / 127.0f;
    output->scale()[1] = 127.0f / scale_max;
    output->flush();
  }

  void cpu_compute() {
    Tensor* input = param_.input;
    Tensor* output = param_.output;
    input->syncToCPU();

    Tensor float_input;
    float* float_input_data =
        float_input.mutableData<float>(FP32, input->shape());
    float_input.copyFrom(input);

    float16* data_out = output->data<float16>();

    int kernel_hw = param_.kernelSize[0] * param_.kernelSize[1];

    float scale_max = 0;
    for (int i = 0; i < output->shape().channel(); i++) {
      float sum = 0;
      for (int j = 0; j < kernel_hw; j++) {
        sum += float_input_data[i * kernel_hw + j];
      }
      float value = sum / kernel_hw;
      data_out[i] = float_to_half(value);
      scale_max = std::max(scale_max, std::abs(value));
    }
    output->scale()[0] = scale_max / 127.0f;
    output->scale()[1] = 127.0f / scale_max;
    output->flush();
  }

  bool dispatch() {
    if (use_cpu_) {
      // cpu_compute();
      compute();
      return true;
    }
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
          float_to_half(1.0f / (kernel_height * kernel_width));
      config_global_pool(globalPoolArgs);
    }
    int ret = (compute_fpga_pool(param_.poolingArgs) == 0);
    if (param_.globalPooling) {
      inplace_.relu_enable = false;
      inplace_.leaky_relu_enable = false;
      inplace_.relu6_enable = false;
      inplace_.sigmoid_enable = false;
      inplace_.global_pool_en = false;
      config_inplace(inplace_);
      globalPoolArgs.global_pool_factor = float_to_half(0);
      config_global_pool(globalPoolArgs);
    }

    return ret;
  }

  PoolingParam& param() { return param_; }

 private:
  PoolingParam param_;
  bool use_cpu_;
  InplaceArgs inplace_ = {0};
  GlobalPoolArgs globalPoolArgs;
};

}  // namespace zynqmp
}  // namespace paddle
