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

#include "lite/kernels/arm/sequence_softmax_compute.h"
#include <vector>
#include "lite/api/paddle_place.h"
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void SequenceSoftmaxCompute::PrepareForRun() {}

void SequenceSoftmaxCompute::Run() {
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto& param = this->Param<operators::SequenceSoftmaxParam>();

  const auto* x_data = param.X->data<float>();
  auto* o_data = param.Out->mutable_data<float>();
  auto input_dims = param.X->dims();
  int in_h = input_dims[0];
  int in_w = param.X->numel() / in_h;
  CHECK_EQ(in_w, 1) << "input dims is not valid";
  auto seq_offset = param.X->lod()[0];
  CHECK_EQ(in_h, seq_offset.back()) << "input dims is not valid";

  lite::arm::math::sequence_softmax(x_data, seq_offset, o_data, &ctx);
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(sequence_softmax,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::SequenceSoftmaxCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
