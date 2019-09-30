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

#include <stdio.h>
#include "lite/backends/fpga/KD/pe.hpp"
#include "lite/backends/fpga/KD/pes/conv_pe.hpp"
#include "lite/backends/fpga/KD/pes/depthwise_conv_pe.hpp"
#include "lite/backends/fpga/KD/pes/fully_connected_pe.hpp"
#include "lite/backends/fpga/KD/pes/input_pe.hpp"
#include "lite/backends/fpga/KD/pes/output_pe.hpp"
#include "lite/backends/fpga/KD/pes/pooling_pe.hpp"
#include "lite/backends/fpga/KD/pes/softmax_pe.hpp"

namespace paddle {
namespace zynqmp {

class Context {
 public:
  template <typename Ptype>
  Ptype& pe() {
    if (pe_ == nullptr) {
      pe_ = new Ptype();
    }
    return static_cast<Ptype&>(*pe_);
  }

  ~Context() {
    if (pe_ != nullptr) {
      delete pe_;
    }
  }

 private:
  PE* pe_ = nullptr;
};
}  // namespace zynqmp
}  // namespace paddle
