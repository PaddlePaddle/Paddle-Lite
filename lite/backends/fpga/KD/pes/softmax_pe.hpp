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

#include <math.h>
#include <algorithm>
#include <limits>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#include "lite/backends/arm/math/funcs.h"
#endif

#include "lite/backends/fpga/KD/pe.hpp"
#include "lite/backends/fpga/KD/pe_params.hpp"
#include "lite/backends/fpga/KD/pes/pooling_pe.hpp"

namespace paddle {
namespace zynqmp {

class SoftmaxPE : public PE {
 public:
  bool init();
  void apply();
  bool dispatch();

  SoftmaxParam& param();

 private:
  bool use_cpu_ = false;
  SoftmaxParam param_;
  PoolingPE poolingPE_;
};

}  // namespace zynqmp
}  // namespace paddle
