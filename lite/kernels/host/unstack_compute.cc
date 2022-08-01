// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/host/unstack_compute.h"
#include <cstring>
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <typename T, PrecisionType PType>
void UnstackCompute<T, PType>::Run() {
  auto& param = this->template Param<operators::UnstackParam>();
  auto x = param.X;
  auto outs = param.Out;
  auto x_dims = x->dims();
  int axis = param.axis;
  if (axis < 0) {
    axis += x_dims.size();
  }

  size_t stride_copy = 1;
  for (size_t i = axis + 1; i < x_dims.size(); i++) {
    stride_copy *= static_cast<size_t>(x_dims[i]);
  }
  size_t stride_move = stride_copy * static_cast<size_t>(x_dims[axis]);
  size_t copy_times = static_cast<size_t>(x_dims.production()) / stride_move;

  const T* x_data = x->template data<T>();
  for (size_t i = 0; i < outs.size(); i++) {
    const T* x_ptr = x_data + i * stride_copy;
    T* out_ptr = outs[i]->template mutable_data<T>();
    for (size_t j = 0; j < copy_times; j++) {
      std::memcpy(out_ptr, x_ptr, sizeof(T) * stride_copy);
      x_ptr += stride_move;
      out_ptr += stride_copy;
    }
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using unstack_float =
    paddle::lite::kernels::host::UnstackCompute<float, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(unstack, kHost, kFloat, kAny, unstack_float, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindOutput("Y",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kAny))})
    .Finalize();

using unstack_int32 =
    paddle::lite::kernels::host::UnstackCompute<int32_t, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(unstack, kHost, kFloat, kAny, unstack_int32, unstack_int32)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindOutput("Y",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kInt32),
                                       DATALAYOUT(kAny))})
    .Finalize();
