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
#ifdef ENABLE_ARM_FP16
#include "lite/backends/arm/math/fp16/funcs_fp16.h"
#endif

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {
template <PrecisionType Ptype, typename Dtype>
void SequencePoolCompute<Ptype, Dtype>::PrepareForRun() {}

template <PrecisionType Ptype, typename Dtype>
void SequencePoolCompute<Ptype, Dtype>::Run() {
  auto& param = this->template Param<operators::SequencePoolParam>();
  auto& output = param.Out;
  const auto* din = param.X->template data<Dtype>();
  Dtype* dout = output->template mutable_data<Dtype>();
  int64_t* max_index = param.MaxIndex->template mutable_data<int64_t>();
  const auto pool_type = param.pool_type;
  const auto lod = param.X->lod()[param.X->lod().size() - 1];
  const auto pad_value = param.pad_value;

  int64_t width = param.X->numel() / param.X->dims()[0];

  if (pool_type == "SUM") {
    lite::arm::math::seq_pool_sum<Dtype>(din, dout, lod, width, pad_value);
  } else if (pool_type == "AVERAGE") {
    lite::arm::math::seq_pool_average<Dtype>(din, dout, lod, width, pad_value);
  } else if (pool_type == "SQRT") {
    lite::arm::math::seq_pool_sqrt<Dtype>(din, dout, lod, width, pad_value);
  } else if (pool_type == "MAX") {
    lite::arm::math::seq_pool_max<Dtype>(
        din, dout, max_index, lod, width, pad_value);
  } else if (pool_type == "MIN") {
    lite::arm::math::seq_pool_min<Dtype>(
        din, dout, max_index, lod, width, pad_value);
  } else if (pool_type == "FIRST") {
    lite::arm::math::seq_pool_first<Dtype>(din, dout, lod, width, pad_value);
  } else if (pool_type == "LAST") {
    lite::arm::math::seq_pool_last<Dtype>(din, dout, lod, width, pad_value);
  } else {
    LOG(ERROR) << " UNKNOWN sequence pool type" << pool_type;
  }
  int batch_size = lod.size() - 1;
  std::vector<uint64_t> offset_new;
  if (param.X->lod().size() == 2) {
    offset_new.resize(param.X->lod()[0].size());
    offset_new = param.X->lod()[0];
  } else {
    offset_new.resize(batch_size + 1);
    for (int i = 0; i <= batch_size; i++) {
      offset_new[i] = i;
    }
  }

  output->mutable_lod()->clear();
  output->mutable_lod()->push_back(offset_new);
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
typedef paddle::lite::kernels::arm::SequencePoolCompute<PRECISION(kFloat),
                                                        float>
    SeqPoolFp32;

#ifdef ENABLE_ARM_FP16
typedef paddle::lite::kernels::arm::SequencePoolCompute<PRECISION(kFP16),
                                                        float16_t>
    SeqPoolFp16;

REGISTER_LITE_KERNEL(sequence_pool, kARM, kFP16, kNCHW, SeqPoolFp16, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindOutput("MaxIndex",
                {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .Finalize();
#endif  // ENABLE_ARM_FP16

REGISTER_LITE_KERNEL(sequence_pool, kARM, kFloat, kNCHW, SeqPoolFp32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindOutput("MaxIndex",
                {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .Finalize();
