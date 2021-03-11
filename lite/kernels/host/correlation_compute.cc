// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lite/kernels/host/correlation_compute.h"
#include "lite/backends/host/math/pad2d.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

inline bool BoundsCheck(int i, int len) { return i >= 0 && i < len; }

template <class T>
void CorrelationCompute<T>::Run() {
  auto& param = this->template Param<param_t>();
  auto* input1 = param.input1;
  auto* input2 = param.input2;
  auto* output = param.output;
  int pad_size = param.pad_size;
  int kernel_size = param.kernel_size;
  int stride1 = param.stride1;
  int stride2 = param.stride2;
  int max_displacement = param.max_displacement;

  auto in_dims = input1->dims();
  int in = in_dims[0];
  int ic = in_dims[1];
  int ih = in_dims[2];
  int iw = in_dims[3];

  auto out_dims = output->dims();
  int on = out_dims[0];
  int oc = out_dims[1];
  int oh = out_dims[2];
  int ow = out_dims[3];

  int displacement_rad = max_displacement / stride2;
  int displacement_size = 2 * displacement_rad + 1;
  int kernel_rad = (kernel_size - 1) / 2;
  int kernel_range = kernel_size * kernel_size * ic;

  const T* input1_data = input1->template data<T>();
  const T* input2_data = input2->template data<T>();
  T* out_data = param.output->template mutable_data<T>();

  for (int b = 0; b < on; b++) {
    int in_idx = b * ic * ih * iw;
    for (int h = 0; h < oh; h++) {
      int h1 = (h - pad_size) * stride1 + max_displacement + kernel_rad;
      for (int w = 0; w < ow; w++) {
        int w1 = (w - pad_size) * stride1 + max_displacement + kernel_rad;
        int out_idx = b * oc * oh * ow + h * ow + w;
        for (int ti = -displacement_rad; k <= displacement_rad; ti++) {
          for (int tj = -displacement_rad; l <= displacement_rad; tj++) {
            int tc = (ti + displacement_rad) * displacement_size +
                     (tj + displacement_rad);
            int h2 = h1 + ti * stride2;
            int w2 = w1 + tj * stride2;
            int o_idx = out_idx + tc * oh * ow;
            for (int i = -kernel_rad; i <= kernel_rad; i++) {
              int x1 = h1 + i;
              int x2 = h2 + i;
              if (!BoundsCheck(x1, ih) || !BoundsCheck(x2, ih)) continue;
              for (int j = -kernel_rad; j <= kernel_rad; j++) {
                int y1 = w1 + j;
                int y2 = w1 + j;
                if (!BoundsCheck(y1, iw) || !BoundsCheck(y2, iw)) continue;
                for (int c = 0; c < ic; c++) {
                  out_data[o_idx] +=
                      input1_data[in_idx + c * ih * iw + x1 * iw + y1] *
                      input2_data[in_idx + c * ih * iw + x2 * iw + y2];
                }
              }
            }
            out_data[o_idx] /= kernel_range;
          }
        }
      }
    }
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(correlation,
                     kHost,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::host::CorrelationCompute<float>,
                     def)
    .BindInput("Input1", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Input2", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
