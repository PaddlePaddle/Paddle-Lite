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

#if defined(__ARM_NEON__) && defined(__aarch64__)

#include "operators/math/depthwise_conv3x3.h"
#ifdef __ARM_NEON__
#include <arm_neon.h>
#endif

namespace paddle_mobile {
namespace operators {
namespace math {

// template<>
// void DepthwiseConv3x3<int8_t, int32_t>(
//     const framework::Tensor *input, const framework::Tensor *filter,
//     const std::vector<int> &strides, framework::Tensor *output) {
//   PADDLE_MOBILE_THROW_EXCEPTION(
//       "Depthwise conv with generic strides has not been implemented.");
// }

template <>
void DepthwiseConv3x3S1<int8_t, int32_t>(const framework::Tensor &input,
                                         const framework::Tensor &filter,
                                         const std::vector<int> &paddings,
                                         framework::Tensor *output) {
  PADDLE_MOBILE_THROW_EXCEPTION(
      "Depthwise conv3x3 with stride 1 for arm v8 has not been implemented.");
}

template <>
void DepthwiseConv3x3S2<int8_t, int32_t>(const framework::Tensor &input,
                                         const framework::Tensor &filter,
                                         const std::vector<int> &paddings,
                                         framework::Tensor *output) {
  PADDLE_MOBILE_THROW_EXCEPTION(
      "Depthwise conv3x3 with stride 2 for arm v8 has not been implemented.");
}

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile

#endif
