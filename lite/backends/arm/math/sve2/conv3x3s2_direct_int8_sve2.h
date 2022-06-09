// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include "lite/backends/arm/math/conv_block_utils.h"
#include "lite/backends/arm/math/conv_impl.h"
#include "lite/core/context.h"
#include "lite/core/parallel_defines.h"
#include "lite/operators/op_params.h"
#ifdef ARM_WITH_OMP
#include <omp.h>
#endif

namespace paddle {
namespace lite {
namespace arm {
namespace math {
namespace sve2 {

#if defined(__aarch64__) && defined(LITE_WITH_ARM8_SVE2)
template <typename Dtype>
void conv_3x3s2_direct_int8_sve2(const int8_t* din,
                                 Dtype* dout,
                                 int num,
                                 int chout,
                                 int hout,
                                 int wout,
                                 int chin,
                                 int hin,
                                 int win,
                                 const int8_t* weights,
                                 const float* bias,
                                 const operators::ConvParam& param,
                                 Context<TARGET(kARM)>* ctx,
                                 const float* scale);
#endif

}  // namespace sve2
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
