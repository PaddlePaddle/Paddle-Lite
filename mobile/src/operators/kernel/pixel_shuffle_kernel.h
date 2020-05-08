/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

#ifdef LRN_OP

#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef __ARM_NEON
#include <arm_neon.h>
#include "operators/math/math.h"
#endif
#include "framework/operator.h"
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

template <typename DeviceType, typename T>
class PixelShuffleKernel
    : public framework::OpKernelBase<DeviceType,
                                     PixelShuffleParam<DeviceType>> {
 public:
  void Compute(const PixelShuffleParam<DeviceType> &param);
  bool Init(PixelShuffleParam<DeviceType> *param);
};
}  // namespace operators
}  // namespace paddle_mobile

#endif
