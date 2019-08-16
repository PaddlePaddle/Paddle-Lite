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

#include "operators/math/pad.h"

namespace paddle_mobile {
namespace operators {
namespace math {

template <typename T>
class PadFunctor<CPU, T> {
 public:
  void operator()(const framework::Tensor &input, const int pad_top,
                  const int pad_bottom, const int pad_left, const int pad_right,
                  framework::Tensor *output) {
    const T *in_data = input.data<T>();
    T *out_data = output->mutable_data<T>();
    // should check output shape is valid for such pad parameters
    const framework::DDim &input_shape = input.dims();
    const framework::DDim &output_shape = output->dims();
    // fill output with 0
    memset(out_data, 0, sizeof(T) * output->numel());
    // should make sure the shape of output is match with input
    for (int i = 0; i < input_shape[0]; ++i) {
      for (int c = 0; c < input_shape[1]; ++c) {
        out_data += pad_top * output_shape[3];
        for (int h = 0; h < input_shape[2]; ++h) {
          memcpy(out_data + pad_left, in_data, sizeof(T) * input_shape[3]);
          out_data += output_shape[3];
          in_data += input_shape[3];
        }
        out_data += pad_bottom * output_shape[3];
      }
    }
  }
};

template class PadFunctor<CPU, float>;
template class PadFunctor<CPU, int8_t>;

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile
