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

#ifdef SCALE_OP

#include "operators/kernel/scale_kernel.h"

namespace paddle_mobile {
namespace operators {

/*
 * @b 特化到具体平台的实现, param 从 op 层传入
 * */
template <>
void ScaleKernel<CPU, float>::Compute(const ScaleParam<CPU> &param) {
  const auto *input_x = param.InputX();
  auto *input_x_ptr = input_x->data<float>();
  auto *out = param.Out();
  auto *out_ptr = out->mutable_data<float>();

  const vector<float> scales = param.Scales();
  bool has_bias = param.HasBias();

  const int dim_size = input_x->dims().size();
  switch (dim_size) {
    case 1: {
      const int input_width = input_x->dims()[0];
      if (has_bias) {
        const vector<float> biases = param.Biases();
        #pragma omp parallel for
        for (int w = 0; w < input_width; w++) {
          out_ptr[w] = input_x_ptr[w] * scales[w] + biases[w];
        }
      } else {
        #pragma omp parallel for
        for (int w = 0; w < input_width; w++) {
          out_ptr[w] = input_x_ptr[w] * scales[w];
        }
      }
    } break;
    case 2: {
      const int input_height = input_x->dims()[0];
      const int input_width = input_x->dims()[1];

      if (has_bias) {
        const vector<float> biases = param.Biases();
        #pragma omp parallel for
        for (int h = 0; h < input_height; ++h) {
          const float *iptr = input_x_ptr + h * input_width;
          float *optr = out_ptr + h * input_width;
          for (int w = 0; w < input_width; ++w) {
            optr[w] = iptr[w] * scales[w] + biases[w];
          }
        }
      } else {
        #pragma omp parallel for
        for (int h = 0; h < input_height; ++h) {
          const float *iptr = input_x_ptr + h * input_width;
          float *optr = out_ptr + h * input_width;
          for (int w = 0; w < input_width; ++w) {
            optr[w] = iptr[w] * scales[w];
          }
        }
      }
    } break;
    case 3: {
      const int chan_size = input_x->dims()[0];
      const int input_height = input_x->dims()[1];
      const int input_width = input_x->dims()[2];
      int size = input_width * input_height;

      if (has_bias) {
        const vector<float> biases = param.Biases();

        #pragma omp parallel for
        for (int c = 0; c < chan_size; ++c) {
          const float *iptr = input_x_ptr + c * size;
          float *optr = out_ptr + c * size;
          for (int i = 0; i < size; ++i) {
            optr[i] = iptr[i] * scales[c] + biases[c];
          }
        }
      } else {
        #pragma omp parallel for
        for (int c = 0; c < chan_size; ++c) {
          const float *iptr = input_x_ptr + c * size;
          float *optr = out_ptr + c * size;
          for (int i = 0; i < size; ++i) {
            optr[i] = iptr[i] * scales[c];
          }
        }
      }
    } break;

    case 4: {
      const int batch_size = input_x->dims()[0];
      const int chan_size = input_x->dims()[0];
      const int input_height = input_x->dims()[1];
      const int input_width = input_x->dims()[2];
      int size = input_width * input_height;

      if (has_bias) {
        const vector<float> biases = param.Biases();

        #pragma omp parallel for
        for (int b = 0; b < batch_size; ++b) {
          for (int c = 0; c < chan_size; ++c) {
            const float *iptr = input_x_ptr + b * c * size;
            float *optr = out_ptr + b * c * size;
            for (int i = 0; i < size; ++i) {
              optr[i] = iptr[i] * scales[c] + biases[c];
            }
          }
        }
      } else {
        #pragma omp parallel for
        for (int b = 0; b < batch_size; ++b) {
          for (int c = 0; c < chan_size; ++c) {
            const float *iptr = input_x_ptr + b * c * size;
            float *optr = out_ptr + b * c * size;
            for (int i = 0; i < size; ++i) {
              optr[i] = iptr[i] * scales[c];
            }
          }
        }
      }
    } break;
    default:
      break;
  }
}
}  // namespace operators
}  // namespace paddle_mobile

#endif
