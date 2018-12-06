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

#include <climits>
#include <cmath>
#include "common/log.h"
#include "framework/tensor.h"
#include "pool_2x2.h"
#include "pool_3x3.h"

namespace paddle_mobile {
namespace operators {
namespace math {

#define FLT_MAX __FLT_MAX__

/*
 * \brief Extracting simple operations from pooling.
 *        Both MaxPool and AvgPool need "initial", "compute" and "finalize"
 * operation.
 *        MaxPool initializes temp variable to the negative maximum to find the
 * maximum value in the pooling field.
 *        AvgPool initializes temp variable to the zero to accumulate all values
 * in pool pooling, and finally takes the average.
 *        MaxPoolGrad and AvgPoolGrad are gradient operations respectively.
 */
template <typename T>
class MaxPool {
 public:
  inline T initial() {
    if (typeid(T) == typeid(int8_t)) {
      return static_cast<T>(-SCHAR_MAX);
    }
    return static_cast<T>(-FLT_MAX);
  }

  inline void compute(const T &x, T *y) { *y = *y > x ? *y : x; }

  inline void finalize(const T &pool_field, T *y) {}
};

template <typename Itype, typename Otype>
class AvgPool {
 public:
  inline Otype initial() { return static_cast<Otype>(0); }

  inline void compute(const Itype &x, Otype *y) { *y += x; }

  inline void finalize(const float &pool_field, Otype *y) {
    if (typeid(Itype) == typeid(int8_t)) {
      float tmp = *y / pool_field;
      if (tmp > SCHAR_MAX) {
        *y = SCHAR_MAX;
      } else if (tmp < -SCHAR_MAX) {
        *y = -SCHAR_MAX;
      } else {
        *y = static_cast<Otype>(std::round(tmp));
      }
    } else {
      *y /= pool_field;
    }
  }
};

template <typename DeviceType, typename PoolProcess, typename T>
class PoolFunctor {
 public:
  void operator()(const framework::Tensor &input, const std::vector<int> &ksize,
                  const std::vector<int> &strides,
                  const std::vector<int> &paddings, PoolProcess pool_compute,
                  framework::Tensor *output);
};
}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile

#endif
