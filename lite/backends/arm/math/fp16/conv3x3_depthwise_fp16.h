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
#include "lite/core/context.h"
#include "lite/core/tensor.h"
#include "lite/operators/op_params.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {
namespace fp16 {
#define CONV_DEPTHWISE_PARAM                                                  \
  float16_t *dout, const float16_t *din, const float16_t *weights,            \
      const float16_t *bias, const float16_t *scale, bool flag_bias, int num, \
      int ch_in, int h_in, int w_in, int h_out, int w_out, ARMContext *ctx

#define DECLARE_3x3_DW_FP16_FUNC(stride, pad, size, act_type)                  \
  void conv_depthwise_3x3##stride##pad##_bias_##act_type##_##size##_fp16_fp16( \
      CONV_DEPTHWISE_PARAM)

DECLARE_3x3_DW_FP16_FUNC(s1, p1, common, noact);
DECLARE_3x3_DW_FP16_FUNC(s1, p1, common, relu);
DECLARE_3x3_DW_FP16_FUNC(s1, p1, common, relu6);
DECLARE_3x3_DW_FP16_FUNC(s1, p1, common, leaky_relu);

DECLARE_3x3_DW_FP16_FUNC(s1, p0, common, noact);
DECLARE_3x3_DW_FP16_FUNC(s1, p0, common, relu);
DECLARE_3x3_DW_FP16_FUNC(s1, p0, common, relu6);
DECLARE_3x3_DW_FP16_FUNC(s1, p0, common, leaky_relu);

DECLARE_3x3_DW_FP16_FUNC(s2, p1, common, noact);
DECLARE_3x3_DW_FP16_FUNC(s2, p1, common, relu);
DECLARE_3x3_DW_FP16_FUNC(s2, p1, common, relu6);
DECLARE_3x3_DW_FP16_FUNC(s2, p1, common, leaky_relu);

DECLARE_3x3_DW_FP16_FUNC(s2, p0, common, noact);
DECLARE_3x3_DW_FP16_FUNC(s2, p0, common, relu);
DECLARE_3x3_DW_FP16_FUNC(s2, p0, common, relu6);
DECLARE_3x3_DW_FP16_FUNC(s2, p0, common, leaky_relu);

DECLARE_3x3_DW_FP16_FUNC(s1, p1, small, noact);
DECLARE_3x3_DW_FP16_FUNC(s1, p1, small, relu);
DECLARE_3x3_DW_FP16_FUNC(s1, p1, small, relu6);
DECLARE_3x3_DW_FP16_FUNC(s1, p1, small, leaky_relu);

DECLARE_3x3_DW_FP16_FUNC(s1, p0, small, noact);
DECLARE_3x3_DW_FP16_FUNC(s1, p0, small, relu);
DECLARE_3x3_DW_FP16_FUNC(s1, p0, small, relu6);
DECLARE_3x3_DW_FP16_FUNC(s1, p0, small, leaky_relu);

DECLARE_3x3_DW_FP16_FUNC(s2, p1, small, noact);
DECLARE_3x3_DW_FP16_FUNC(s2, p1, small, relu);
DECLARE_3x3_DW_FP16_FUNC(s2, p1, small, relu6);
DECLARE_3x3_DW_FP16_FUNC(s2, p1, small, leaky_relu);

DECLARE_3x3_DW_FP16_FUNC(s2, p0, small, noact);
DECLARE_3x3_DW_FP16_FUNC(s2, p0, small, relu);
DECLARE_3x3_DW_FP16_FUNC(s2, p0, small, relu6);
DECLARE_3x3_DW_FP16_FUNC(s2, p0, small, leaky_relu);

}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
