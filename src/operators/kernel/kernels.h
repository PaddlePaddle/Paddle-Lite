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

#include "framework/operator.h"
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

#define DECLARE_KERNEL(KernelClass, KernelParam)                              \
  template <typename DeviceType, typename T>                                  \
  class KernelClass                                                           \
      : public framework::OpKernelBase<DeviceType, KernelParam<DeviceType>> { \
   public:                                                                    \
    bool Init(KernelParam<DeviceType> *param);                                \
    void Compute(const KernelParam<DeviceType> &param);                       \
  };

#ifdef TOP_K_OP
DECLARE_KERNEL(TopKKernel, TopKParam)
#endif  // TOP_K_OP

#ifdef CAST_OP
DECLARE_KERNEL(CastKernel, CastParam)
#endif  // CAST_OP

}  // namespace operators
}  // namespace paddle_mobile
