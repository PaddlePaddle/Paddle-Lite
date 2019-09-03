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

class PriorBoxPE : public PE {
 public:
  bool init() {
    param_.outputBoxes->setAligned(false);
    param_.outputVariances->setAligned(false);
    param_.outputBoxes->setDataLocation(CPU);
    param_.outputVariances->setDataLocation(CPU);
    return true;
  }

  bool dispatch();

  void apply();

  PriorBoxParam& param() { return param_; }

 private:
  PriorBoxParam param_;
  Tensor* cachedBoxes_ = nullptr;
  Tensor* cachedVariances_ = nullptr;

  void compute_prior_box();
};
}  // namespace zynqmp
}  // namespace paddle
