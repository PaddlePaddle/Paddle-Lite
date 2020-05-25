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

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void LookupTableCompute::Run() {
  auto& param = this->Param<param_t>();
  auto w = param.W;
  auto ids = param.Ids;
  auto out = param.Out;

  auto table_dim = w->dims();
  int64_t ids_numel = ids->numel();
  auto ids_data = ids->data<int64_t>();

  int64_t row_number = table_dim[0];
  int64_t row_width = table_dim[1];
  auto table_data = w->data<float>();
  auto dout = out->mutable_data<float>();

  for (int64_t i = 0; i < ids_numel; ++i) {
    int ids_int = ids_data[i];
    if (param.padding_idx != -1 && ids_data[i] == param.padding_idx) {
      memset(dout + i * row_width, 0, row_width * sizeof(float));
    } else {
      CHECK_LT(ids_data[i], row_number)
          << "look uptable ids[i] < row_number check failed";
      CHECK_GE(ids_data[i], 0) << "lookuptable ids[i] >= 0 check failed";

      memcpy(dout + i * row_width,
             table_data + ids_int * row_width,
             row_width * sizeof(float));
    }
  }
  *(out->mutable_lod()) = ids->lod();
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(lookup_table,
                     kARM,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::arm::LookupTableCompute,
                     def)
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Ids", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

REGISTER_LITE_KERNEL(lookup_table_v2,
                     kARM,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::arm::LookupTableCompute,
                     def)
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Ids", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
