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

#pragma once
#include <arm_neon.h>
#include <cmath>
#include "lite/backends/arm/math/fp16/gemm_prepacked_fp16.h"
#include "lite/core/target_wrapper.h"
#include "lite/operators/op_params.h"
#include "lite/utils/cp_logging.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {
namespace fp16 {
inline void trans_gemm_weights_fp16(const Tensor& tin,
                                    Tensor& tout,  // NOLINT
                                    int group,
                                    ARMContext* ctx) {
  CHECK_EQ(tin.dims().size(), 4) << "conv weights dims size must = 4";
  int m = tin.dims()[0] / group;
  int k = tin.dims().count(1, 4);
  int hblock = lite::arm::math::fp16::get_hblock_fp16(ctx);
  int m_roundup = hblock * ((m + hblock - 1) / hblock);
  int group_size_round_up = ((m_roundup * k + 15) / 16) * 16;
  float16_t* w_trans_ptr = nullptr;
  tout.Resize({group_size_round_up * group});
  w_trans_ptr = tout.mutable_data<float16_t>();
  const auto* w_data = tin.data<float16_t>();
  for (int g = 0; g < group; ++g) {
    const float16_t* weights_group = w_data + g * m * k;
    float16_t* weights_trans_ptr = w_trans_ptr + g * group_size_round_up;
    lite::arm::math::fp16::prepackA_fp16(
        weights_trans_ptr, weights_group, 1.f, k, 0, m, 0, k, false, ctx);
  }
}

}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
