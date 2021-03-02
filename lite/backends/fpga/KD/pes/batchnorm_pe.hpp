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

#include <fstream>
#include <iostream>

#include "lite/backends/fpga/KD/pe.hpp"
#include "lite/backends/fpga/KD/pe_params.hpp"
#include "lite/backends/fpga/KD/pes/scale_pe.hpp"

namespace paddle {
namespace zynqmp {
class BatchnormPE : public PE {
 public:
  bool init() {
    Tensor* output = param_.output;
    output->setAligned(true);
    output->setDataLocation(Device);

    ScaleParam& scale_param = scalePE_.param();
    scale_param.input = param_.input;
    scale_param.output = param_.output;
    Shape shape(N, {output->shape().channel()});
    auto mean_data = param_.mean->data<float>();
    auto variance_data = param_.variance->data<float>();
    auto scale_data = param_.scale->data<float>();
    auto bias_data = param_.bias->data<float>();
    auto new_scale_ptr = scale_.mutableData<zynqmp::float16>(FP16, shape);
    auto new_bias_ptr = bias_.mutableData<zynqmp::float16>(FP16, shape);
    float epsilon = param_.epsilon;

    Shape& in_shape = param_.input->shape();
    bool match = in_shape.channel() == 128 && in_shape.height() == 128 &&
                 in_shape.width() == 128;

    for (int c = 0; c < output->shape().channel(); c++) {
      float var = variance_data[c];
      float inv_scale = 1.0 / (std::sqrt(var + epsilon));
      float scale_value = inv_scale * scale_data[c];
      float bias_value = bias_data[c] - scale_value * mean_data[c];
      new_scale_ptr[c] = zynqmp::float_to_half(scale_value);
      new_bias_ptr[c] = zynqmp::float_to_half(bias_value);
    }

    scale_.flush();
    bias_.flush();

    scale_param.scale = &scale_;
    scale_param.bias = &bias_;
    scale_param.activeParam.type = param_.activeParam.type;
    scalePE_.init();
    return true;
  }

  void apply() { scalePE_.apply(); }

  bool dispatch() { return scalePE_.dispatch(); }

  BatchnormParam& param() { return param_; }

 private:
  BatchnormParam param_;
  ScalePE scalePE_;
  Tensor scale_;
  Tensor bias_;
};
}  // namespace zynqmp
}  // namespace paddle
