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

#include "lite/kernels/arm/scatter_compute.h"
#include "lite/backends/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

template <typename Dtype>
void ScatterCompute<Dtype>::Run() {
  auto& param = this->template Param<operators::ScatterParam>();
  const float* updates_data = param.updates->template data<float>();
  const Dtype* indexs_data = param.indexs->template data<Dtype>();
  float* output_data = param.output->template mutable_data<float>();
  bool overwrite = param.overwrite;
  int index_size = param.indexs->dims()[0];
  auto in_dims = param.x->dims();
  int num = 1;
  for (int i = 1; i < in_dims.size(); i++) {
    num *= in_dims[i];
  }
  lite::arm::math::scatter<Dtype>(indexs_data,
                                  updates_data,
                                  output_data,
                                  index_size,
                                  in_dims[0],
                                  num,
                                  overwrite);
  if (!param.x->lod().empty()) {
    param.output->set_lod(param.x->lod());
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
using ScatterCompute_int64 =
    paddle::lite::kernels::arm::ScatterCompute<int64_t>;
REGISTER_LITE_KERNEL(
    scatter, kARM, kFloat, kNCHW, ScatterCompute_int64, ids_int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindInput("Ids", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindInput("Updates",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .Finalize();

using ScatterCompute_int32 =
    paddle::lite::kernels::arm::ScatterCompute<int32_t>;
REGISTER_LITE_KERNEL(
    scatter, kARM, kFloat, kNCHW, ScatterCompute_int32, ids_int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindInput("Ids", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("Updates",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .Finalize();
