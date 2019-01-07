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

#ifdef POOL_OP

#pragma once

#include <string>
#include <vector>
#include "common/types.h"
#include "operators/math/pooling.h"

namespace paddle_mobile {
namespace operators {

template <typename P>
void PoolCompute(const PoolParam<CPU> &param) {
  const framework::Tensor *input = param.Input();
  framework::Tensor *output = param.Output();
  const std::string &pooling_type = param.PoolingType();
  std::vector<int> ksize = param.Ksize();
  std::vector<int> strides = param.Strides();
  std::vector<int> paddings = param.Paddings();
  if (param.isGlobalPooling()) {
    for (size_t i = 0; i < ksize.size(); ++i) {
      paddings[i] = 0;
      ksize[i] = static_cast<int>(input->dims()[i + 2]);
    }
  }
  if (ksize[0] == 3 && ksize[0] == ksize[1]) {
    if (pooling_type == "max" && strides[0] == strides[1]) {
      if (strides[0] == 1) {
        math::Pooling3x3<MAX, 1>()(*input, paddings, output);
      } else if (strides[0] == 2) {
        math::Pooling3x3<MAX, 2>()(*input, paddings, output);
      } else {
        math::Pooling<MAX>()(*input, ksize, strides, paddings, output);
      }
    } else if (pooling_type == "avg" && strides[0] == strides[1]) {
      if (strides[0] == 1) {
        math::Pooling3x3<AVG, 1>()(*input, paddings, output);
      } else if (strides[0] == 2) {
        math::Pooling3x3<AVG, 2>()(*input, paddings, output);
      } else {
        math::Pooling<AVG>()(*input, ksize, strides, paddings, output);
      }
    }
  } else if (ksize[0] == 2 && ksize[0] == ksize[1]) {
    if (pooling_type == "max" && strides[0] == strides[1]) {
      if (strides[0] == 1) {
        math::Pooling2x2<MAX, 1>()(*input, paddings, output);
      } else if (strides[0] == 2) {
        math::Pooling2x2<MAX, 2>()(*input, paddings, output);
      } else {
        math::Pooling<MAX>()(*input, ksize, strides, paddings, output);
      }
    } else if (pooling_type == "avg" && strides[0] == strides[1]) {
      if (strides[0] == 1) {
        math::Pooling2x2<AVG, 1>()(*input, paddings, output);
      } else if (strides[0] == 2) {
        math::Pooling2x2<AVG, 2>()(*input, paddings, output);
      } else {
        math::Pooling<AVG>()(*input, ksize, strides, paddings, output);
      }
    }
  } else {
    if (pooling_type == "max") {
      math::Pooling<MAX>()(*input, ksize, strides, paddings, output);
    } else if (pooling_type == "avg") {
      math::Pooling<AVG>()(*input, ksize, strides, paddings, output);
    } else {
      // Others
    }
  }
}

}  // namespace operators
}  // namespace paddle_mobile
#endif
