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
#include "operators/math/pooling.h"

namespace paddle_mobile {
namespace operators {
using framework::Tensor;

inline void PoolBasic(std::string pooling_type, std::vector<int> ksize,
                      std::vector<int> strides, std::vector<int> paddings,
                      const Tensor *in_x, Tensor *out) {
  if (pooling_type == "max") {
    math::PoolFunctor<CPU, math::MaxPool<float>, float> pool2d_forward;
    math::MaxPool<float> pool_process;
    pool2d_forward(*in_x, ksize, strides, paddings, pool_process, out);

  } else if (pooling_type == "avg") {
    math::PoolFunctor<CPU, math::AvgPool<float>, float> pool2d_forward;
    math::AvgPool<float> pool_process;
    pool2d_forward(*in_x, ksize, strides, paddings, pool_process, out);
  }
}
template <typename P>
void PoolCompute(const PoolParam &param) {
  const Tensor *in_x = param.Input();
  Tensor *out = param.Output();
  std::string pooling_type = param.PoolingType();

  std::vector<int> ksize = param.Ksize();

  std::vector<int> strides = param.Strides();

  std::vector<int> paddings = param.Paddings();
  if (ksize.size() != 2) {
    LOG(paddle_mobile::LogLevel::kLOG_ERROR)
        << "Pool op only supports 2D and 3D input.";
  }

  if (param.isGlobalPooling()) {
    for (size_t i = 0; i < ksize.size(); ++i) {
      paddings[i] = 0;
      ksize[i] = static_cast<int>(in_x->dims()[i + 2]);
    }
  } else if (ksize[0] == 3 && ksize[0] == ksize[1]) {
    if (pooling_type == "max") {
      if (strides[0] == strides[1] && strides[0] == 1 &&
          paddings[0] == paddings[1] && paddings[1] == 1) {
        math::Pool3x3Maxs1p1(in_x, out);
      } else {
        math::Pool3x3Max(strides, paddings, in_x, out);
      }
    } else if (pooling_type == "avg") {
      if (strides[0] == strides[1] && strides[0] == 1 &&
          paddings[0] == paddings[1] && paddings[1] == 1) {
        math::Pool3x3Avgs1p1(in_x, out);
      } else {
        math::Pool3x3Avg(strides, paddings, in_x, out);
      }
    }

  } else if (ksize[0] == 2 && ksize[0] == ksize[1]) {
#if __ARM_NEON
#if __aarch64__
    PoolBasic(pooling_type, ksize, strides, paddings, in_x, out);
#else
    if (pooling_type == "max") {
      math::Pool2x2Max(strides, paddings, in_x, out);
    } else if (pooling_type == "avg") {
      math::Pool2x2Avg(strides, paddings, in_x, out);
    }
#endif
#else
    PoolBasic(pooling_type, ksize, strides, paddings, in_x, out);
#endif  // __ARM_NEON

  } else {
    PoolBasic(pooling_type, ksize, strides, paddings, in_x, out);
  }
}

}  // namespace operators
}  // namespace paddle_mobile
#endif
