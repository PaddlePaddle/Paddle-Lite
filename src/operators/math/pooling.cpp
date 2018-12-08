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

#include "operators/math/pooling.h"
#include <algorithm>
#include <vector>
#include "common/types.h"
#ifdef _OPENMP
#include <omp.h>
#endif

namespace paddle_mobile {
namespace operators {
namespace math {

/*
 * All tensors are in NCHW format.
 * Ksize, strides, paddings are two elements. These two elements represent
 * height and width, respectively.
 */
template <typename PoolProcess, typename T>
class PoolFunctor<CPU, PoolProcess, T> {
 public:
  void operator()(const framework::Tensor &input, const std::vector<int> &ksize,
                  const std::vector<int> &strides,
                  const std::vector<int> &paddings, PoolProcess pool_process,
                  framework::Tensor *output) {
    const int batch_size = input.dims()[0];

    const int input_height = input.dims()[2];

    const int input_width = input.dims()[3];

    const int output_channels = output->dims()[1];

    const int output_height = output->dims()[2];
    const int output_width = output->dims()[3];
    const int ksize_height = ksize[0];
    const int ksize_width = ksize[1];
    const int stride_height = strides[0];
    const int stride_width = strides[1];
    const int padding_height = paddings[0];
    const int padding_width = paddings[1];

    const int input_stride = input_height * input_width;
    const int output_stride = output_height * output_width;

    const T *input_data = input.data<T>();
    T *output_data = output->mutable_data<T>();
    for (int i = 0; i < batch_size; i++) {
      for (int c = 0; c < output_channels; ++c) {
        #pragma omp parallel for
        for (int ph = 0; ph < output_height; ++ph) {
          int hstart = ph * stride_height - padding_height;
          int hend = std::min(hstart + ksize_height, input_height);
          hstart = std::max(hstart, 0);
          for (int pw = 0; pw < output_width; ++pw) {
            int wstart = pw * stride_width - padding_width;
            int wend = std::min(wstart + ksize_width, input_width);
            wstart = std::max(wstart, 0);

            auto ele = pool_process.initial();
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                pool_process.compute(input_data[h * input_width + w], &ele);
              }
            }
            int pool_size = (hend - hstart) * (wend - wstart);
            pool_process.finalize(static_cast<float>(pool_size), &ele);
            output_data[ph * output_width + pw] = static_cast<T>(ele);
          }
        }
        input_data += input_stride;
        output_data += output_stride;
      }
    }
  }
};

template class PoolFunctor<CPU, math::AvgPool<float, float>, float>;
template class PoolFunctor<CPU, math::MaxPool<float>, float>;
template class PoolFunctor<CPU, math::AvgPool<int8_t, int32_t>, int8_t>;
template class PoolFunctor<CPU, math::MaxPool<int8_t>, int8_t>;
}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile

#endif
