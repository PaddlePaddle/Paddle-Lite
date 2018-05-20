/* Copyright (c) 2016 Baidu, Inc. All Rights Reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
==============================================================================*/

#pragma once

#include "common/log.h"
#include "framework/tensor.h"

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
template <class T> class MaxPool {
public:
  inline T initial() { return static_cast<T>(-FLT_MAX); }

  inline void compute(const T &x, T *y) { *y = *y > x ? *y : x; }

  inline void finalize(const T &pool_field, T *y) {}
};

template <class T> class AvgPool {
public:
  inline T initial() { return static_cast<T>(0); }

  inline void compute(const T &x, T *y) { *y += x; }

  inline void finalize(const T &pool_field, T *y) { *y /= pool_field; }
};

template <typename DeviceType, typename PoolProcess, typename T>
class PoolFunctor {
public:
  void operator()(const framework::Tensor &input, const std::vector<int> &ksize,
                  const std::vector<int> &strides,
                  const std::vector<int> &paddings, PoolProcess pool_compute,
                  framework::Tensor *output);
};
}
} // namespace operators
} // namespace paddle_mobile
