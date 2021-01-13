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

#include "lite/kernels/host/argmax_compute.h"
#include <string>
#include <vector>
#include "lite/backends/host/math/argmax.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <typename T>
void ArgmaxCompute<T>::Run() {
  auto& param = Param<operators::ArgmaxParam>();
  lite::Tensor* input = param.X;
  lite::Tensor* output = param.Out;
  int axis = param.Axis;
  if (axis < 0) {
    axis += input->dims().size();
  }

  switch (param.dtype) {
    // default indices type: int64_t
    case -1: {
      lite::host::math::argmax_func<T, int64_t>(input, axis, output);
      break;
    }
    // static_cast<int>(lite::core::FluidType::INT32) == 2
    case 2: {
      lite::host::math::argmax_func<T, int32_t>(input, axis, output);
      break;
    }
    // static_cast<int>(lite::core::FluidType::INT64) == 3
    case 3: {
      lite::host::math::argmax_func<T, int64_t>(input, axis, output);
      break;
    }
    default: {
      LOG(FATAL) << "Attribute `dtype` in arg_max op must be 2 or 3, which "
                    "indicates that indices dtype must be int32 or int64, "
                    "default dtype is int64.";
      break;
    }
  }
#ifdef LITE_WITH_PROFILE
  kernel_func_name_ = "argmax_func";
#endif
  return;
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(arg_max,
                     kHost,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::host::ArgmaxCompute<float>,
                     fp32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindPaddleOpVersion("arg_max", 1)
    .Finalize();

#ifdef LITE_BUILD_EXTRA
// arg_max only supports float input except that LITE_WITH_EXTRA=ON
REGISTER_LITE_KERNEL(arg_max,
                     kHost,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::host::ArgmaxCompute<int64_t>,
                     int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindPaddleOpVersion("arg_max", 1)
    .Finalize();

REGISTER_LITE_KERNEL(arg_max,
                     kHost,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::host::ArgmaxCompute<int32_t>,
                     int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindPaddleOpVersion("arg_max", 1)
    .Finalize();

REGISTER_LITE_KERNEL(arg_max,
                     kHost,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::host::ArgmaxCompute<int16_t>,
                     int16)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindPaddleOpVersion("arg_max", 1)
    .Finalize();

REGISTER_LITE_KERNEL(arg_max,
                     kHost,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::host::ArgmaxCompute<uint8_t>,
                     uint8)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kUInt8))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindPaddleOpVersion("arg_max", 1)
    .Finalize();
#endif  // LITE_BUILD_EXTRA
