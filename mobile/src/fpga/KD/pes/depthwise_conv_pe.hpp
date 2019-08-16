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

#include "../float16.hpp"
#include "../pe.hpp"
#include "../pe_params.hpp"
#include "conv_process.hpp"

namespace paddle_mobile {
namespace zynqmp {

class DepthwiseConvPE : public PE {
 public:
  bool init() {
    std::cout << "DWConv init" << std::endl;
    return true;
  }

  void apply() {
    DepthwiseConvParam& param = param_;
    Tensor* input = param.input;
    Tensor* output = param.output;
    int channel = output->shape().channel();

    Tensor* new_scale = param.scale();
    Tensor* new_bias = param.bias();
    Shape shape(NC, {channel, 1});
    float* new_scale_data = new_scale->mutableData<float>(FP32, shape);
    float16* new_bias_data = new_bias->mutableData<float16>(FP16, shape);

    BatchnormParam* batchnorm = param.batchnorm;
    memset(new_scale_data, 0, new_scale->shape().memorySize(sizeof(float16)));
    memset(new_bias_data, 0, new_bias->shape().memorySize(sizeof(float16)));
    if (batchnorm != nullptr) {
      for (size_t i = 0; i < channel; i++) {
        // TODO(chonwhite) combine;
      }
    } else {
      float16 zero = float_to_half(0.0f);
      for (size_t i = 0; i < channel; i++) {
        new_bias_data[i] = zero;
        new_scale_data[i] = 1.0f;
      }
    }

    Tensor* quantized_filter = param.quantizedFilter();
    quantized_filter->mutableData<float16>(FP16, param.filter->shape());
    format_dw_filter(param.filter, param.quantizedFilter(), new_scale_data);

    DWconvArgs args = {0};

    void* filter_address = quantized_filter->data<float>();
    std::cout << "filter:" << filter_address;

    args.bias_address = new_bias_data;
    args.filter_address = param.quantizedFilter()->data<void>();
    args.kernel.width = param.kernelSize[0];
    args.kernel.height = param.kernelSize[1];
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
  }

  bool dispatch() { return compute_fpga_dwconv(param_.args) == 0; }

  DepthwiseConvParam& param() { return param_; }

 private:
  DepthwiseConvParam param_;
};

}  // namespace zynqmp
}  // namespace paddle_mobile
