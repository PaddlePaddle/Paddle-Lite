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

#include "lite/kernels/arm/sequence_expand_compute.h"
#include "lite/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void SequenceExpandCompute::PrepareForRun() {}

void SequenceExpandCompute::Run() {
  auto& param = Param<operators::SequenceExpandParam>();
  const float* x_data = param.X->data<float>();
  int width = param.X->numel() / param.X->dims()[0];
  auto& output = param.Out;
  const auto x_lod = param.X->lod();
  const auto y_lod = param.Y->lod();
  int ref_level = param.ref_level;
  if (ref_level == -1) {
    ref_level = y_lod.size() - 1;
  }
  lite::arm::math::SequenceExpandImpl(
      x_data, x_lod, width, y_lod[ref_level], output);
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(sequence_expand,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::SequenceExpandCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
