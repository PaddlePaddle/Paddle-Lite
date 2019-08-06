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

#include "lite/fpga/KD/pe.hpp"
#include "lite/fpga/KD/pe_params.hpp"

namespace paddle {
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
    std::cout << input->data<float16>() << " " << output->data<float16>()
              << std::endl;
    if (input->aligned()) {
      std::cout << " 1" << std::endl;
      Tensor tmp;
      tmp.setAligned(true);
      std::cout << " 1" << std::endl;
      tmp.mutableData<float16>(FP16, input->shape());
      std::cout << " 1" << std::endl;
      tmp.copyFrom(input);
      std::cout << " 1" << std::endl;
      tmp.unalignImage();
      std::cout << " 1" << std::endl;
      output->copyFrom(&tmp);
      std::cout << " 1" << std::endl;
    } else {
      output->copyFrom(input);
      std::cout << " copy" << std::endl;
    }
    return true;
  }

  OutputParam& param() { return param_; }

 private:
  OutputParam param_;
};
}  // namespace zynqmp
}  // namespace paddle
