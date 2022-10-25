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

#include "lite/kernels/arm/lookup_table_compute.h"
#include <string>
#include <vector>
#include "lite/api/paddle_place.h"
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/core/type_system.h"
#ifdef ENABLE_ARM_FP16
#include "lite/backends/arm/math/fp16/funcs_fp16.h"
#endif

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

template <typename T_W, typename T_IDS>
void LookupTableCompute<T_W, T_IDS>::Run() {
  auto& param = this->Param<param_t>();
  auto w = param.W;
  auto ids = param.Ids;
  auto out = param.Out;

  auto table_dim = w->dims();
  int64_t ids_numel = ids->numel();
  auto ids_data = ids->template data<T_IDS>();

  int64_t row_number = table_dim[0];
  int64_t row_width = table_dim[1];
  auto table_data = w->template data<T_W>();
  auto dout = out->template mutable_data<T_W>();

  for (int64_t i = 0; i < ids_numel; ++i) {
    int ids_int = ids_data[i];
    if (param.padding_idx != -1 && ids_data[i] == param.padding_idx) {
      memset(dout + i * row_width, 0, row_width * sizeof(T_W));
    } else {
      CHECK_LT(ids_data[i], row_number)
          << "look uptable ids[i] < row_number check failed";
      CHECK_GE(ids_data[i], 0) << "lookuptable ids[i] >= 0 check failed";
#ifdef ENABLE_ARM_FP16
      if (w->precision() != PRECISION(kFP16)) {
        auto table_data = w->template data<float>();
        memcpy(dout + i * row_width,
               table_data + ids_int * row_width,
               row_width * sizeof(float));
      } else {
        const float16_t* in_data = w->template data<float16_t>();
        Tensor tmp_tensor;
        tmp_tensor.Resize(std::vector<int64_t>({1, row_width}));
        tmp_tensor.set_precision(PrecisionType::kFloat);
        float* table_data = tmp_tensor.mutable_data<float>();
        lite::arm::math::fp16::fp16_to_fp32(
            in_data + ids_int * row_width, table_data, tmp_tensor.numel());
        memcpy(dout + i * row_width, table_data, row_width * sizeof(float));
      }
#else
      auto table_data = w->template data<float>();
      memcpy(dout + i * row_width,
             table_data + ids_int * row_width,
             row_width * sizeof(float));
#endif
    }
  }
  *(out->mutable_lod()) = ids->lod();
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using LookupTableFloatInt64 =
    paddle::lite::kernels::arm::LookupTableCompute<float, int64_t>;
using LookupTableFloatInt32 =
    paddle::lite::kernels::arm::LookupTableCompute<float, int32_t>;

REGISTER_LITE_KERNEL(
    lookup_table, kARM, kAny, kNCHW, LookupTableFloatInt64, def)
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Ids", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

REGISTER_LITE_KERNEL(
    lookup_table_v2, kARM, kAny, kNCHW, LookupTableFloatInt64, def)
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Ids", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindPaddleOpVersion("lookup_table_v2", 1)
    .Finalize();

REGISTER_LITE_KERNEL(
    lookup_table, kARM, kAny, kNCHW, LookupTableFloatInt32, float_int32)
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Ids", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

REGISTER_LITE_KERNEL(
    lookup_table_v2, kARM, kAny, kNCHW, LookupTableFloatInt32, float_int32)
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Ids", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindPaddleOpVersion("lookup_table_v2", 1)
    .Finalize();
