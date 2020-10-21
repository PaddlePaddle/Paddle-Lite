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
    output->setDataLocation(CPU);
    return true;
  }

  void apply() {}

  bool dispatch() {
    param_.input->invalidate();
    int16_t* input_data = param_.input->data<int16_t>();
    float16* out_data = param_.output->data<float16>();
    for (int i = 0; i < param_.input->shape().alignedElementCount(); i++) {
      int16_t v = param_.input->data<float16>()[i];
      if (v > 0) {
        out_data[i] = input_data[i];
      } else {
        out_data[i] = zero;
      }
    }
    param_.output->copyScaleFrom(param_.input);
    param_.output->flush();
    return true;
  }

  InputParam& param() { return param_; }

 private:
  InputParam param_;
  float16 zero = float_to_half(0.0f);
};

}  // namespace zynqmp
}  // namespace paddle
