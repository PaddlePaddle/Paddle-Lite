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

class ElementwiseAddPE : public PE {
 public:
  bool init() {
    Tensor* output = param_.output;
    output->setAligned(true);
    output->setDataLocation(Device);
    return true;
  }

  void apply() {
    int dynamic_range = (1 << 12) - 1;  // int13 max value, pow(2,12)-1
    float16 dynamic_range_fp16 = float_to_half(dynamic_range * 1.0);
    float inv_dynamic_range = 1.0 / dynamic_range;

    Tensor* input0 = param_.inputs[0];
    Tensor* input1 = param_.inputs[1];
    Tensor* output = param_.output;
    EWAddArgs args = {0};
    args.const0 = 0x3c00;
    args.const1 = 0x3c00;  // =1
    args.image0.address = input0->data<float16>();
    args.image0.channels = input0->shape().channel();
    args.image0.scale_address = &input_max_;
    args.image0.height = input0->shape().height();
    args.image0.width = input0->shape().width();
    args.image0.pad_height = 0;
    args.image0.pad_width = 0;
    args.image1.address = input1->data<float16>();
    args.image1.channels = input1->shape().channel();
    args.image1.scale_address = &input_max_;
    args.image1.height = input1->shape().height();
    args.image1.width = input1->shape().width();
    args.image1.pad_height = 0;
    args.image1.pad_width = 0;
    args.output.scale_address = output->max();
    args.output.address = output->data<float16>();
    args.inplace.active_param.type = param_.activeParam.type;
    args.inplace.active_param.leaky_relu_factor =
        float_to_half(param_.activeParam.leaky_relu_factor);
    args.quant.dynamic_range =
        *(reinterpret_cast<uint16_t*>(&dynamic_range_fp16));
    args.quant.inv_dynamic_range =
        *(reinterpret_cast<uint32_t*>(&inv_dynamic_range));
    param_.ewargs = args;
  }

  bool dispatch() {
    param_.inputs[0]->syncToDevice();
    param_.inputs[1]->syncToDevice();
    input_max_ =
        float_to_half(std::max(half_to_float(param_.inputs[0]->max()[0]),
                               half_to_float(param_.inputs[1]->max()[0])));

    return compute_fpga_ewadd(param_.ewargs);
  }

  ElementwiseAddParam& param() { return param_; }

 private:
  ElementwiseAddParam param_;
  float16 input_max_ = 0;
};

}  // namespace zynqmp
}  // namespace paddle
