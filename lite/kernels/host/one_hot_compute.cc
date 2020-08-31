// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/host/one_hot_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <typename T>
void OneHotKernelFunctor(const Tensor* in,
                         Tensor* out,
                         int depth,
                         bool allow_out_of_range = false) {
  auto* p_in_data = in->data<int64_t>();
  auto numel = in->numel();
  auto* p_out_data = out->mutable_data<T>();
  memset(p_out_data, 0, out->numel() * sizeof(T));
  if (allow_out_of_range) {
    for (int i = 0; i < numel; ++i) {
      if (p_in_data[i] >= 0 && p_in_data[i] < depth) {
        p_out_data[i * depth + static_cast<int>(p_in_data[i])] = 1.0;
      }
    }
  } else {
    for (int i = 0; i < numel; ++i) {
      CHECK_GE(p_in_data[i], 0) << "Illegal index value, Input(input) value "
                                   "should be at least 0, but received input ("
                                << p_in_data[i] << ") less than 0";
      CHECK_LE(p_in_data[i], depth)
          << "Illegal index value, Input(input) value should be less than "
             "Input(depth), but received input ("
          << p_in_data[i] << ") not less than depth (" << depth << ")";
      p_out_data[i * depth + static_cast<int>(p_in_data[i])] = 1.0;
    }
  }
}

void OneHotCompute::Run() {
  auto& param = this->template Param<param_t>();
  switch (param.dtype) {
    case static_cast<int>(lite::core::FluidType::INT64):
      OneHotKernelFunctor<int64_t>(
          param.X, param.Out, param.depth, param.allow_out_of_range);
      break;
    case static_cast<int>(lite::core::FluidType::INT32):
      OneHotKernelFunctor<int32_t>(
          param.X, param.Out, param.depth, param.allow_out_of_range);
      break;
    case static_cast<int>(lite::core::FluidType::FP32):
      OneHotKernelFunctor<float>(
          param.X, param.Out, param.depth, param.allow_out_of_range);
      break;
    default:
      LOG(ERROR) << "Unsupported data type for one_hot op:" << param.dtype;
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    one_hot, kHost, kAny, kAny, paddle::lite::kernels::host::OneHotCompute, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindInput("depth_tensor",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kAny),
                                       DATALAYOUT(kAny))})
    .Finalize();
