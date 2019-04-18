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

#include "../pe.hpp"
#include "../pe_params.hpp"
namespace paddle_mobile {
namespace zynqmp {

class PoolingPE : public PE {
 public:
  bool init() {
    Tensor* output = param_.output;
    output->setAligned(true);
    return true;
  }

  void apply() {
    Tensor* input = param_.input;
    Tensor* output = param_.output;

    uint32_t k_width = param_.kernelSize[0];
    uint32_t k_height = param_.kernelSize[1];

    if (param_.globalPooling) {
      k_width = input->shape().width();
      k_height = input->shape().height();
    }

    PoolingArgs args = {0};
    args.mode = param_.type;
    args.kernel_reciprocal = fp32_2_fp16(1.0f / (k_width * k_height));
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
  }

  bool dispatch() { return compute_fpga_pool(param_.poolingArgs) == 0; }

  PoolingParam& param() { return param_; }

 private:
  PoolingParam param_;
};

}  // namespace zynqmp
}  // namespace paddle_mobile
