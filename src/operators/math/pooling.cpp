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
namespace paddle_mobile {
namespace operators {
namespace math {

template <PoolingType P>
void Pooling<P>::operator()(const framework::Tensor &input,
                            const std::vector<int> &kernel_size,
                            const std::vector<int> &strides,
                            const std::vector<int> &paddings,
                            framework::Tensor *output) {
  const int batch_size = input.dims()[0];
  const int input_height = input.dims()[2];
  const int input_width = input.dims()[3];
  const int output_channels = output->dims()[1];
  const int output_height = output->dims()[2];
  const int output_width = output->dims()[3];
  const int ksize_height = kernel_size[0];
  const int ksize_width = kernel_size[1];
  const int stride_height = strides[0];
  const int stride_width = strides[1];
  const int padding_height = paddings[0];
  const int padding_width = paddings[1];

  const float *input_data = input.data<float>();
  float *output_data = output->mutable_data<float>();
  const size_t input_spatial_size = input_height * input_width;
  const size_t output_spatial_size = output_height * output_width;

  #pragma omp parallel for collapse(2)
  // num_threads(framework::threads())
  for (int i = 0; i < batch_size; i++) {
    for (int c = 0; c < output_channels; ++c) {
      int channel = i * output_channels + c;
      const float *input_ptr = input_data + channel * input_spatial_size;
      float *output_ptr = output_data + channel * output_spatial_size;

      for (int ph = 0; ph < output_height; ++ph) {
        int hstart = ph * stride_height - padding_height;
        int hend = std::min(hstart + ksize_height, input_height);
        hstart = std::max(hstart, 0);
        for (int pw = 0; pw < output_width; ++pw) {
          int wstart = pw * stride_width - padding_width;
          int wend = std::min(wstart + ksize_width, input_width);
          wstart = std::max(wstart, 0);

          PoolingVal<P> val;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              val += input_ptr[h * input_width + w];
            }
          }
          output_ptr[ph * output_width + pw] = val.Value();
        }
      }
    }
  }
}

template struct Pooling<MAX>;
template struct Pooling<AVG>;

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile

#endif  // POOL_OP
