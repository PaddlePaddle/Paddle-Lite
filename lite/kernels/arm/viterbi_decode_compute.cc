// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/arm/viterbi_decode_compute.h"
#include "lite/backends/arm/math/viterbi_decode.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void ViterbiDecodeCompute::Run() {
  auto& param = Param<operators::ViterbiDecodeParam>();
  auto input = param.input;
  auto transition = param.transition;
  auto length = param.length;
  auto include_bos_eos_tag = param.include_bos_eos_tag;
  auto scores = param.scores;
  auto path = param.path;
  lite::arm::math::viterbi_decode(
      *param.input, *transition, *length, include_bos_eos_tag, scores, path);
  return;
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(viterbi_decode,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::ViterbiDecodeCompute,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Length",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindInput("Transition", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Path",
                {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindOutput("Scores", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindPaddleOpVersion("viterbi_decode", 1)
    .Finalize();
