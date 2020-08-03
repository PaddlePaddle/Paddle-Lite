/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "lite/kernels/arm/sequence_pool_compute.h"
#include <string>
#include <vector>
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void SequencePoolCompute::PrepareForRun() {}

void SequencePoolCompute::Run() {
  auto& param = Param<operators::SequencePoolParam>();
  auto& output = param.Out;
  const auto* din = param.X->data<float>();
  float* dout = output->mutable_data<float>();
  int64_t* max_index = param.MaxIndex->mutable_data<int64_t>();
  const auto pool_type = param.pool_type;
  const auto lod = param.X->lod()[0];

  int64_t width = param.X->numel() / param.X->dims()[0];

  if (pool_type == "SUM") {
    lite::arm::math::seq_pool_sum(din, dout, lod, width);
  } else if (pool_type == "AVERAGE") {
    lite::arm::math::seq_pool_average(din, dout, lod, width);
  } else if (pool_type == "SQRT") {
    lite::arm::math::seq_pool_sqrt(din, dout, lod, width);
  } else if (pool_type == "MAX") {
    lite::arm::math::seq_pool_max(din, dout, max_index, lod, width);
  } else if (pool_type == "MIN") {
    lite::arm::math::seq_pool_min(din, dout, max_index, lod, width);
  } else if (pool_type == "FIRST") {
    lite::arm::math::seq_pool_first(din, dout, lod, width);
  } else if (pool_type == "LAST") {
    lite::arm::math::seq_pool_last(din, dout, lod, width);
  } else {
    LOG(ERROR) << " UNKNOWN sequence pool type";
  }
  int batch_size = lod.size() - 1;
  std::vector<uint64_t> offset_new(static_cast<uint64_t>(batch_size + 1));
  for (int i = 0; i <= batch_size; i++) {
    offset_new[i] = i;
  }
  output->mutable_lod()->clear();
  output->mutable_lod()->push_back(offset_new);
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(sequence_pool,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::SequencePoolCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("MaxIndex", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
