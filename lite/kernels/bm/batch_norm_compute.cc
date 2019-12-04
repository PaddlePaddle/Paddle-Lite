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

#include "lite/kernels/bm/batch_norm_compute.h"
#include <string>
#include <vector>
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace bm {

void BatchNormCompute::PrepareForRun() {
  return;
}

void BatchNormCompute::Run() {
  return;
}

template <PrecisionType Ptype_out>
void BatchNormComputeInt8<Ptype_out>::PrepareForRun() {
  return;
}

template <PrecisionType Ptype_out> 
void BatchNormComputeInt8<Ptype_out>::Run() {
  return;
}

}  // namespace bm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
  batch_norm, kBM, kFloat, kNCHW, paddle::lite::kernels::bm::BatchNormCompute, def)
  .BindInput("X", {LiteType::GetTensorTy(TARGET(kBM))})
  .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kBM))})
  .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kBM))})
  .BindInput("Mean", {LiteType::GetTensorTy(TARGET(kBM))})
  .BindInput("Variance", {LiteType::GetTensorTy(TARGET(kBM))})
  .BindOutput("Y", {LiteType::GetTensorTy(TARGET(kBM))})
  .BindOutput("MeanOut", {LiteType::GetTensorTy(TARGET(kBM))})
  .BindOutput("VarianceOut", {LiteType::GetTensorTy(TARGET(kBM))})
  .BindOutput("SavedMean", {LiteType::GetTensorTy(TARGET(kBM))})
  .BindOutput("SavedVariance", {LiteType::GetTensorTy(TARGET(kBM))})
  .Finalize();

REGISTER_LITE_KERNEL(
  batch_norm, kBM, kInt8, kNCHW, paddle::lite::kernels::bm::BatchNormComputeInt8<PRECISION(kInt8)>, def)
  .BindInput("X", {LiteType::GetTensorTy(TARGET(kBM))})
  .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kBM))})
  .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kBM))})
  .BindInput("Mean", {LiteType::GetTensorTy(TARGET(kBM))})
  .BindInput("Variance", {LiteType::GetTensorTy(TARGET(kBM))})
  .BindOutput("Y", {LiteType::GetTensorTy(TARGET(kBM))})
  .BindOutput("MeanOut", {LiteType::GetTensorTy(TARGET(kBM))})
  .BindOutput("VarianceOut", {LiteType::GetTensorTy(TARGET(kBM))})
  .BindOutput("SavedMean", {LiteType::GetTensorTy(TARGET(kBM))})
  .BindOutput("SavedVariance", {LiteType::GetTensorTy(TARGET(kBM))})
  .Finalize();
