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

#ifndef Context_hpp
#define Context_hpp

#include <stdio.h>
#include "pe.hpp"
#include "pes/conv_pe.hpp"
#include "pes/depthwise_conv_pe.hpp"
#include "pes/fully_connected_pe.hpp"
#include "pes/input_pe.hpp"
#include "pes/output_pe.hpp"
#include "pes/pooling_pe.hpp"
#include "pes/softmax_pe.hpp"

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

#endif /* Context_hpp */
