// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#ifndef LITE_BACKENDS_ARM_MATH_FP16_CONV_DEPTHWISE_COMMON_FP16_H_
#define LITE_BACKENDS_ARM_MATH_FP16_CONV_DEPTHWISE_COMMON_FP16_H_

#include <vector>
#include "lite/backends/arm/math/fp16/common_preprocess.h"
#include "lite/backends/arm/math/fp16/conv_block_utils_fp16.h"
#include "lite/core/context.h"
#include "lite/core/parallel_defines.h"
#ifdef ARM_WITH_OMP
#include <omp.h>
#endif

namespace paddle {
namespace lite {
namespace arm {
namespace math {
namespace fp16 {
void conv_depthwise_common_line(const float16_t* i_data,
                                float16_t* o_data,
                                int ic,
                                int ih,
                                int iw,
                                int bs,
                                int oc,
                                int oh,
                                int ow,
                                int kh,
                                int kw,
                                std::vector<int> strides,
                                std::vector<int> dilations,
                                std::vector<int> paddings,
                                const float16_t* weights,
                                const float16_t* bias,
                                const operators::ConvParam& param,
                                ARMContext* ctx);

}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle

#endif  // LITE_BACKENDS_ARM_MATH_FP16_CONV_DEPTHWISE_COMMON_FP16_H_
