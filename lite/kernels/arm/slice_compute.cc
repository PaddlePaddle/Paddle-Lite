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
#include "lite/kernels/arm/slice_compute.h"
#include <vector>
#include "lite/backends/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void SliceCompute::PrepareForRun() {}

void SliceCompute::Run() {
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto& param = this->Param<operators::SliceParam>();

  auto input_dims = param.X->dims();
  int dim_size = param.X->dims().size();

  std::vector<int> starts = param.starts;
  std::vector<int> ends = param.ends;
  std::vector<int> axes = param.axes;
  const auto* x_data = param.X->data<int>();
  auto* o_data = param.Out->mutable_data<int>();
  lite::arm::math::slice(
      x_data, input_dims.data(), axes, starts, ends, o_data, &ctx);
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    slice, kARM, kFloat, kNCHW, paddle::lite::kernels::arm::SliceCompute, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

// REGISTER_LITE_KERNEL(
//    slice, kARM, kFloat, kNCHW, paddle::lite::kernels::arm::SliceCompute, def)
//    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), Precision(kINT32))})
//    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM),
//    Precision(kINT32))})
//    .Finalize();
