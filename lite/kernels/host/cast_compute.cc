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

#include "lite/kernels/host/cast_compute.h"
#include <algorithm>
#ifdef ENABLE_ARM_FP16
#include "lite/backends/arm/math/fp16/funcs_fp16.h"
#endif
namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <class in_type, class out_type>
out_type TransOp(in_type in) {
  return static_cast<out_type>(in);
}

void CastCompute::PrepareForRun() {}

void CastCompute::Run() {
  auto& param = this->Param<operators::CastParam>();
  auto input_dims = param.X->dims();
  if (param.X->precision() == PrecisionType::kFloat) {
    param.in_dtype = 5;
  }
  // BOOL = 0;INT16 = 1;INT32 = 2;INT64 = 3;FP16 = 4;FP32 = 5;FP64 = 6;
  // SIZE_T = 19;UINT8 = 20;INT8 = 21;
  if (param.in_dtype == param.out_dtype && param.in_dtype == 5) {
    const auto* x_data = param.X->data<float>();
    auto* o_data = param.Out->mutable_data<float>();
    memcpy(o_data, x_data, sizeof(float) * param.X->numel());
  } else if (param.in_dtype == param.out_dtype &&
             param.in_dtype == 3) {  // int64->int64
    const auto* x_data = param.X->data<int64_t>();
    auto* o_data = param.Out->mutable_data<int64_t>();
    memcpy(o_data, x_data, sizeof(int64_t) * param.X->numel());
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
  } else if (param.in_dtype == 3 && param.out_dtype == 2) {  // int64->int32
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
  } else if (param.in_dtype == 0 && param.out_dtype == 3) {  // bool->int64
    const bool* x_data_begin = param.X->data<bool>();
    const bool* x_data_end = x_data_begin + param.X->numel();
    int64_t* out_data = param.Out->mutable_data<int64_t>();
    std::transform(x_data_begin, x_data_end, out_data, TransOp<bool, int64_t>);
  } else if (param.in_dtype == 0 && param.out_dtype == 2) {  // bool->int32
    const bool* x_data_begin = param.X->data<bool>();
    const bool* x_data_end = x_data_begin + param.X->numel();
    int32_t* out_data = param.Out->mutable_data<int32_t>();
    std::transform(x_data_begin, x_data_end, out_data, TransOp<bool, int32_t>);
  } else if (param.in_dtype == 0 && param.out_dtype == 0) {  // bool -> bool
    const bool* x_data_begin = param.X->data<bool>();
    const bool* x_data_end = x_data_begin + param.X->numel();
    bool* out_data = param.Out->mutable_data<bool>();
    std::transform(x_data_begin, x_data_end, out_data, TransOp<bool, bool>);
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
  } else if (param.in_dtype == 5 && param.out_dtype == 2) {  // float32 -> INT32
    const float* x_data_begin = param.X->data<float>();
    const float* x_data_end = x_data_begin + param.X->numel();
    int32_t* out_data = param.Out->mutable_data<int32_t>();
    std::transform(x_data_begin, x_data_end, out_data, TransOp<float, int32_t>);
  } else if (param.in_dtype == 5 &&
             param.out_dtype == 20) {  // float32 -> uint8
    const float* x_data_begin = param.X->data<float>();
    const float* x_data_end = x_data_begin + param.X->numel();
    unsigned char* out_data = param.Out->mutable_data<unsigned char>();
    std::transform(
        x_data_begin, x_data_end, out_data, TransOp<float, unsigned char>);
  } else if (param.in_dtype == 5 && param.out_dtype == 3) {  // float32 -> INT64
    const float* x_data_begin = param.X->data<float>();
    const float* x_data_end = x_data_begin + param.X->numel();
    int64_t* out_data = param.Out->mutable_data<int64_t>();
    std::transform(x_data_begin, x_data_end, out_data, TransOp<float, int64_t>);
  } else if (param.in_dtype == 5 && param.out_dtype == 0) {  // float32 -> bool
    const float* x_data_begin = param.X->data<float>();
    const float* x_data_end = x_data_begin + param.X->numel();
    bool* out_data = param.Out->mutable_data<bool>();
    std::transform(x_data_begin, x_data_end, out_data, TransOp<float, bool>);
  } else if (param.in_dtype == 0 && param.out_dtype == 2) {  // bool -> INT32
    const bool* x_data_begin = param.X->data<bool>();
    const bool* x_data_end = x_data_begin + param.X->numel();
    int32_t* out_data = param.Out->mutable_data<int32_t>();
    std::transform(x_data_begin, x_data_end, out_data, TransOp<bool, int32_t>);
  } else if (param.in_dtype == 2 && param.out_dtype == 0) {  // INT32 -> bool
    const int32_t* x_data_begin = param.X->data<int32_t>();
    const int32_t* x_data_end = x_data_begin + param.X->numel();
    bool* out_data = param.Out->mutable_data<bool>();
    std::transform(x_data_begin, x_data_end, out_data, TransOp<int32_t, bool>);
  } else if (param.in_dtype == 2 && param.out_dtype == 2) {  // INT32 -> INT32
    const int32_t* x_data_begin = param.X->data<int32_t>();
    const int32_t* x_data_end = x_data_begin + param.X->numel();
    int32_t* out_data = param.Out->mutable_data<int32_t>();
    std::transform(
        x_data_begin, x_data_end, out_data, TransOp<int32_t, int32_t>);
#if defined(ENABLE_ARM_FP16) && defined(LITE_WITH_ARM)
  } else if (param.in_dtype == 4 &&
             param.out_dtype == 5) {  // float16 -> float32
    const float16_t* in_data = param.X->data<float16_t>();
    float* out_data = param.Out->mutable_data<float>();
    lite::arm::math::fp16::fp16_to_fp32(in_data, out_data, param.X->numel());
  } else if (param.in_dtype == 5 &&
             param.out_dtype == 4) {  // float32 -> float16
    const float* in_data = param.X->data<float>();
    float16_t* out_data = param.Out->mutable_data<float16_t>();
    lite::arm::math::fp16::fp32_to_fp16(in_data, out_data, param.X->numel());
#endif
  } else {
    LOG(FATAL) << "other has not been implemented transform with dtype"
               << param.in_dtype << " X, dtype" << param.out_dtype << " Out";
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    cast, kHost, kAny, kNCHW, paddle::lite::kernels::host::CastCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .Finalize();
