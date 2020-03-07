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
    output->setAligned(true);
    output->setDataLocation(Device);
    return true;
  }

  void apply() {
    Tensor* src = param_.input;

    args_.input_data_type = DATA_TYPE_FP16;
    args_.output_data_type = DATA_TYPE_FP16;
    args_.input_layout_type = LAYOUT_HWC;
    args_.output_layout_type = LAYOUT_HWC;
    args_.image = {.address = src->data<void>(),
                   .scale_address = src->scale(),
                   .channels = (uint32_t)src->shape().channel(),
                   .width = (uint32_t)src->shape().width(),
                   .height = (uint32_t)src->shape().height(),
                   .pad_width = 0u,
                   .pad_height = 0u};
    args_.output = {
        .address = param_.output->data<void>(),
        .scale_address = param_.output->scale(),
    };

    inplace_.relu_enable = false;
    inplace_.power_enable = false;
    inplace_.normalize_enable = false;
  }

  bool dispatch() {
    inplace_.relu_enable = true;
    config_inplace(inplace_);
    param_.input->syncToDevice();
    param_.output->copyFrom(param_.input);
    param_.output->invalidate();
    inplace_.relu_enable = false;
    config_inplace(inplace_);
    return true;
  }

  InputParam& param() { return param_; }

 private:
  InputParam param_;
  BypassArgs args_;
  InplaceArgs inplace_;
};

}  // namespace zynqmp
}  // namespace paddle
