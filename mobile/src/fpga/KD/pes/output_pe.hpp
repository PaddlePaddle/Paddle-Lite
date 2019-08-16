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

class OutputPE : public PE {
 public:
  bool init() {
    Tensor* output = param_.output;
    output->setAligned(false);
    return true;
  }

  bool dispatch() {
    Tensor* input = param_.input;
    Tensor* output = param_.output;
    Tensor* src_tensor = input;
    Tensor float_tensor;
    input->invalidate();
    float_tensor.mutableData<float>(DataType::FP32, input->shape());
    if (input->dataType() == DataType::FP16) {
      float_tensor.copyFrom(input);
      src_tensor = &float_tensor;
    }
    src_tensor->unalignImage(output, true);
    return true;
  }

  OutputParam& param() { return param_; }

 private:
  OutputParam param_;
};
}  // namespace zynqmp
}  // namespace paddle_mobile
