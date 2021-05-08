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

class ReluPE : public PE {
 public:
  bool init() {
    Tensor* output = param_.output;
    output->setAligned(param_.input->aligned());
    output->setDataLocation(Device);
    return true;
  }

  void apply() {
    Shape& input_shape = param_.input->shape();

    bypass_args_.input_data_type = DATA_TYPE_FP16;
    bypass_args_.output_data_type = DATA_TYPE_FP16;
    bypass_args_.input_layout_type = LAYOUT_HWC;
    bypass_args_.output_layout_type = LAYOUT_HWC;
    bypass_args_.image.address = param_.input->data<void>();
    bypass_args_.image.scale_address = param_.input->max();
    bypass_args_.image.channels = input_shape.channel();
    bypass_args_.image.height = input_shape.height();
    bypass_args_.image.width = input_shape.width();
    bypass_args_.output.address = param_.output->data<void>();
    bypass_args_.output.scale_address = param_.output->max();

    bypass_args_.inplace.active_param.type = TYPE_RELU;
    bypass_args_.inplace.active_param.leaky_relu_factor =
        float_to_half(param_.activeParam.leaky_relu_factor);
  }

  bool dispatch() {
    // fpga compute through bypass
    param_.input->syncToDevice();
    perform_bypass(bypass_args_);
    return true;
  }

  InputParam& param() { return param_; }

 private:
  InputParam param_;
  BypassArgs bypass_args_;
  float16 zero = float_to_half(0.0f);
};

}  // namespace zynqmp
}  // namespace paddle
