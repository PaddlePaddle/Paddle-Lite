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

#include "lite/backends/fpga/KD/pe.hpp"
#include "lite/backends/fpga/KD/pe_params.hpp"
namespace paddle {
namespace zynqmp {

class ElementwiseMulPE : public PE {
 public:
  bool init() {
    Tensor* output = param_.output;
    output->setAligned(true);
    output->setDataLocation(Device);
    return true;
  }

  void apply() {
    Tensor* input = param_.input_x;
    Tensor* output = param_.output;

    int wc_aligned = align_to_x(param_.input_x->shape().numel(), 32);

    Shape s(N, {wc_aligned});
    float16* bias_data = bias_tensor.mutableData<float16>(FP16, s);
    memset(bias_data, 0, wc_aligned * sizeof(float16));

    ScaleArgs& args = args_;
    args.scale_address = param_.input_y->data<void>();
    args.bias_address = bias_tensor.data<void>();
    args.wc_alignment = wc_aligned;
    args.channel_alignment = wc_aligned;
    args.image.address = input->data<void>();
    args.image.scale_address = input->scale();
    args.image.channels = wc_aligned;
    args.image.height = 1;
    args.image.width = 1;
    args.image.pad_width = 0;
    args.image.pad_height = 0;
    args.output.address = output->data<void>();
    args.output.scale_address = output->scale();
  }

  void updateInput(Tensor* t, int index) {
    if (index == 0) {
      args_.scale_address = t->data<void>();  // replace inputs?
    }
  }

  bool dispatch() {
    compute_fpga_scale(args_) == 0;
    return true;
  }

  ElementwiseMulParam& param() { return param_; }

 private:
  ElementwiseMulParam param_;
  ScaleArgs args_ = {0};
  Tensor bias_tensor;
};

}  // namespace zynqmp
}  // namespace paddle
