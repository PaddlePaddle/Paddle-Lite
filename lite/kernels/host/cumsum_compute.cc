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

#include "lite/kernels/host/cumsum_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <class T, PrecisionType PType>
void CumsumCompute<T, PType>::Run() {
  auto& param = this->template Param<param_t>();
  const lite::Tensor* x = param.X;
  lite::Tensor* out = param.Out;

  auto x_dims = x->dims();
  const T* x_data = x->template data<T>();
  T* out_data = out->template mutable_data<T>();

  if (param.flatten || x_dims.size() == 1) {
    int64_t x_size = x->numel();
    if (param.reverse) {
      if (param.exclusive) {
        out_data[x_size - 1] = 0;
        for (int64_t i = x_size - 1; i > 0; i--) {
          out_data[i - 1] = x_data[i] + out_data[i];
        }
      } else {
        out_data[x_size - 1] = x_data[x_size - 1];
        for (int64_t i = x_size - 2; i >= 0; i--) {
          out_data[i] = x_data[i] + out_data[i + 1];
        }
      }
    } else {
      if (param.exclusive) {
        out_data[0] = 0;
        for (int64_t i = 0; i < x_size - 1; i++) {
          out_data[i + 1] = x_data[i] + out_data[i];
        }
      } else {
        out_data[0] = x_data[0];
        for (int64_t i = 1; i < x_size; i++) {
          out_data[i] = x_data[i] + out_data[i - 1];
        }
      }
    }
  } else {
    int axis = param.axis < 0 ? param.axis + x_dims.size() : param.axis;
    int64_t pre = x_dims.count(0, axis);
    int64_t count = x_dims[axis];
    int64_t post = x_dims.count(axis + 1, x_dims.size());
    if (param.reverse) {
      if (param.exclusive) {
        for (int64_t i = 0; i < pre; i++) {
          for (int64_t j = 0; j < post; j++) {
            int64_t step = i * count * post + j;
            const T* src = x_data + step;
            T* dst = out_data + step;
            dst[(count - 1) * post] = 0;
            int p = 1;
            for (int64_t k = count - 1; k > 0; k--, p++) {
              dst[(k - 1) * post] = src[k * post] + dst[k * post];
            }
          }
        }
      } else {
        for (int64_t i = 0; i < pre; i++) {
          for (int64_t j = 0; j < post; j++) {
            int64_t step = i * count * post + j;
            const T* src = x_data + step;
            T* dst = out_data + step;
            dst[(count - 1) * post] = src[(count - 1) * post];
            int p = 1;
            for (int64_t k = count - 2; k >= 0; k--, p++) {
              dst[k * post] = src[k * post] + dst[(k + 1) * post];
            }
          }
        }
      }
    } else {
      if (param.exclusive) {
        for (int64_t i = 0; i < pre; i++) {
          for (int64_t j = 0; j < post; j++) {
            int64_t step = i * count * post + j;
            const T* src = x_data + step;
            T* dst = out_data + step;
            dst[0] = 0;
            for (int64_t k = 0; k < count - 1; k++) {
              dst[(k + 1) * post] = src[k * post] + dst[k * post];
            }
          }
        }
      } else {
        for (int64_t i = 0; i < pre; i++) {
          for (int64_t j = 0; j < post; j++) {
            int64_t step = i * count * post + j;
            const T* src = x_data + step;
            T* dst = out_data + step;
            dst[0] = src[0];
            for (int64_t k = 1; k < count; k++) {
              dst[k * post] = src[k * post] + dst[(k - 1) * post];
            }
          }
        }
      }
    }
  }
  return;
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using cumsum_float =
    paddle::lite::kernels::host::CumsumCompute<float, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(cumsum, kHost, kFloat, kAny, cumsum_float, float32)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kAny))})
    .Finalize();

using cumsum_int32 =
    paddle::lite::kernels::host::CumsumCompute<int32_t, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(cumsum, kHost, kFloat, kAny, cumsum_int32, int32)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kInt32),
                                       DATALAYOUT(kAny))})
    .Finalize();

using cumsum_int64 =
    paddle::lite::kernels::host::CumsumCompute<int64_t, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(cumsum, kHost, kFloat, kAny, cumsum_int64, int64)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kInt64),
                                       DATALAYOUT(kAny))})
    .Finalize();
