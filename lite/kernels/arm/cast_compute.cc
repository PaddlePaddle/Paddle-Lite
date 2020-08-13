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

#include "lite/kernels/arm/cast_compute.h"
#include <algorithm>
#include "lite/backends/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

template <class in_type, class out_type>
out_type TransOp(in_type in) {
  return static_cast<out_type>(in);
}

void CastCompute::PrepareForRun() {}

void CastCompute::Run() {
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto& param = this->Param<operators::CastParam>();

  auto input_dims = param.X->dims();

  // BOOL = 0;INT16 = 1;INT32 = 2;INT64 = 3;FP16 = 4;FP32 = 5;FP64 = 6;
  // SIZE_T = 19;UINT8 = 20;INT8 = 21;
  if (param.in_dtype == param.out_dtype && param.in_dtype == 2) {
    const auto* x_data = param.X->data<float>();
    auto* o_data = param.Out->mutable_data<float>();
    memcpy(o_data, x_data, sizeof(float) * param.X->numel());
  } else if (param.in_dtype == 21 && param.out_dtype == 5) {  // int8->float32
    const char* x_data_begin = param.X->data<char>();
    const char* x_data_end = x_data_begin + param.X->numel();
    float* out_data = param.Out->mutable_data<float>();
    std::transform(x_data_begin, x_data_end, out_data, TransOp<char, float>);
  } else if (param.in_dtype == 2 && param.out_dtype == 5) {  // int32 -> float32
    const int32_t* x_data_begin = param.X->data<int32_t>();
    const int32_t* x_data_end = x_data_begin + param.X->numel();
    float* out_data = param.Out->mutable_data<float>();
    std::transform(x_data_begin, x_data_end, out_data, TransOp<int32_t, float>);
  } else if (param.in_dtype == 20 && param.out_dtype == 5) {  // uint8->float32
    const unsigned char* x_data_begin = param.X->data<unsigned char>();
    const unsigned char* x_data_end = x_data_begin + param.X->numel();
    float* out_data = param.Out->mutable_data<float>();
    std::transform(
        x_data_begin, x_data_end, out_data, TransOp<unsigned char, float>);
  } else if (param.in_dtype == 3 && param.out_dtype == 2) {
    const int64_t* x_data_begin = param.X->data<int64_t>();
    const int64_t* x_data_end = x_data_begin + param.X->numel();
    int32_t* out_data = param.Out->mutable_data<int32_t>();
    std::transform(
        x_data_begin, x_data_end, out_data, TransOp<int64_t, int32_t>);
  } else if (param.in_dtype == 0 && param.out_dtype == 5) {  // bool->fp32
    const bool* x_data_begin = param.X->data<bool>();
    const bool* x_data_end = x_data_begin + param.X->numel();
    float* out_data = param.Out->mutable_data<float>();
    std::transform(x_data_begin, x_data_end, out_data, TransOp<bool, float>);
  } else if (param.in_dtype == 3 && param.out_dtype == 5) {  // int64->fp32
    const int64_t* x_data_begin = param.X->data<int64_t>();
    const int64_t* x_data_end = x_data_begin + param.X->numel();
    float* out_data = param.Out->mutable_data<float>();
    std::transform(x_data_begin, x_data_end, out_data, TransOp<int64_t, float>);
  } else if (param.in_dtype == 2 && param.out_dtype == 3) {  // INT32 -> INT64
    const int32_t* x_data_begin = param.X->data<int32_t>();
    const int32_t* x_data_end = x_data_begin + param.X->numel();
    int64_t* out_data = param.Out->mutable_data<int64_t>();
    std::transform(
        x_data_begin, x_data_end, out_data, TransOp<int32_t, int64_t>);
  } else {
    LOG(FATAL) << "other has not been implemented transform with dtype"
               << param.in_dtype << " X, dtype" << param.out_dtype << " Out";
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    cast, kARM, kAny, kNCHW, paddle::lite::kernels::arm::CastCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kAny))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kAny))})
    .Finalize();
