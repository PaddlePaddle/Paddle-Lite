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
#include "../float16.hpp"
#include "../llapi/zynqmp_api.h"

namespace paddle_mobile {
namespace zynqmp {

class InputPE : public PE {
 public:
  bool init() {
    Tensor* output = param_.output;
    output->setAligned(true);
    output->setDataLocation(Device);
    return true;
  }

  bool dispatch() {
    fpga_reset();
    // std::cout << "InputPE dispatch \n";
    Tensor* input = param_.input;
    Tensor* output = param_.output;

    Tensor* src = input;
    input->flush();
    Tensor half_tensor;
    if (input->dataType() == DataType::FP32) {
      half_tensor.mutableData<void*>(DataType::FP16, input->shape());
      half_tensor.copyFrom(input);
      src = &half_tensor;
    }
    output->mutableData<void>();
    src->alignImage(output, true);
    return true;
  }

  InputParam& param() { return param_; }

 private:
  InputParam param_;
};
}  // namespace zynqmp
}  // namespace paddle_mobile
