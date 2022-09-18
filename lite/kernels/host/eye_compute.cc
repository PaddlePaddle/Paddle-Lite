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

#include "lite/kernels/host/eye_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <typename T>
void EyeCompute::MakeIdentityMatrix() {
  auto& param = *param_.get_mutable<param_t>();
  int64_t num_rows = param.num_rows;
  int64_t num_columns = param.num_columns;
  T value = static_cast<T>(0);
  auto data = param.out->template mutable_data<T>();
  for (int i = 0; i < param.out->numel(); i++) {
    data[i] = value;
  }
  for (int i = 0; i < num_rows; i++) {
    for (int j = 0; j < num_columns; j++) {
      if (i == j) data[i * num_columns + j] = static_cast<T>(1);
    }
  }
}

void EyeCompute::Run() {
  auto& param = *param_.get_mutable<param_t>();
  int dtype = param.dtype;

  switch (dtype) {
    case static_cast<int32_t>(lite::core::FluidType::BOOL):
      MakeIdentityMatrix<bool>();
      break;
    case static_cast<int32_t>(lite::core::FluidType::INT16):
      MakeIdentityMatrix<int16_t>();
      break;
    case static_cast<int32_t>(lite::core::FluidType::INT32):
      MakeIdentityMatrix<int32_t>();
      break;
    case static_cast<int32_t>(lite::core::FluidType::INT64):
      MakeIdentityMatrix<int64_t>();
      break;
#ifdef ENABLE_ARM_FP16
    case static_cast<int32_t>(lite::core::FluidType::FP16):
      MakeIdentityMatrix<float16_t>();
      break;
#endif
    case static_cast<int32_t>(lite::core::FluidType::FP32):
      MakeIdentityMatrix<float>();
      break;
    case static_cast<int32_t>(lite::core::FluidType::INT8):
      MakeIdentityMatrix<int8_t>();
      break;
    case static_cast<int32_t>(lite::core::FluidType::UINT8):
      MakeIdentityMatrix<uint8_t>();
      break;
    default:
      LOG(WARNING) << "not supported dtype " << param.dtype << ", using float.";
      MakeIdentityMatrix<float>();
      break;
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    eye, kHost, kAny, kNCHW, paddle::lite::kernels::host::EyeCompute, def)
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindPaddleOpVersion("eye", 1)
    .Finalize();
