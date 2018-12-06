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

// Inspired by https://arxiv.org/abs/1509.09308 and refered from nnpack and ncnn
// project.

#ifdef CONV_OP

#ifdef __aarch64__

#include "operators/math/pad.h"
#include "operators/math/winograd/winograd_transform.h"

namespace paddle_mobile {
namespace operators {
namespace math {

template <>
void winograd_transform_weight<8, 3>(const framework::Tensor &weight,
                                     framework::Tensor *output) {
  /*
   * w0 = g0
   * w1 = ((g0 + g2) + g1) * (-2.0 / 9)
   * w2 = ((g0 + g2) - g1) * (-2.0 / 9)
   * w3 = ((g0 + 4 * g2) + 2 * g1) * (1.0 / 90)
   * w4 = ((g0 + 4 * g2) - 2 * g1) * (1.0 / 90)
   * w5 = ((g2 + 4 * g0) + 2 * g1) * (1.0 / 180)
   * w6 = ((g2 + 4 * g0) - 2 * g1) * (1.0 / 180)
   * w7 = g2
   */
  // TODO(hjchen2)
  PADDLE_MOBILE_THROW_EXCEPTION(
      "Winograd for arm v8 has not been implemented.");
}

template <>
void winograd_transform_input<8, 3>(const framework::Tensor &input,
                                    framework::Tensor *output) {
  /*
   * x0 = (d0 - d6) + (d4 - d2) * 5.25
   * x1 = (d2 + d6) - 4.25 * (d4 + d3) + (d1 + d5)
   * x2 = (d2 + d6) - 4.25 * (d4 - d3) - (d1 + d5)
   * x3 = (0.25 * d2 - 1.25 * d4 + d6) + (0.5 * d1 - 2.5 * d3 + 2 * d5)
   * x4 = (0.25 * d2 - 1.25 * d4 + d6) - (0.5 * d1 - 2.5 * d3 + 2 * d5)
   * x5 = (4 * d2 - 5 * d4 + d6) + (2 * d1 - 2.5 * d3 + 0.5 * d5)
   * x6 = (4 * d2 - 5 * d4 + d6) - (2 * d1 - 2.5 * d3 + 0.5 * d5)
   * x7 = (d7 - d1) + (d3 - d5) * 5.25
   */
  // TODO(hjchen2)
  PADDLE_MOBILE_THROW_EXCEPTION(
      "Winograd for arm v8 has not been implemented.");
}

template <>
void winograd_transform_output<8, 3>(const framework::Tensor &input,
                                     const framework::Tensor &weight,
                                     framework::Tensor *output) {
  // TODO(hjchen2)
  PADDLE_MOBILE_THROW_EXCEPTION(
      "Winograd for arm v8 has not been implemented.");
}

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile

#endif  // __aarch64__
#endif  // CONV_OP
