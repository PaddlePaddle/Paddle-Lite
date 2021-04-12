// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/host/sequence_pad_compute.h"
#include "lite/backends/host/math/sequence_padding.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <class T>
void SequencePadCompute<T>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<HostContext>();

  auto* x = param.X;
  auto* pad_value = param.PadValue;
  auto* len_t = param.Length;
  auto* out = param.Out;
  CHECK(!x->lod().empty()) << "Input X should have lod data.";
  int padded_length = param.padded_length;

  lite::host::math::PaddingLoDTensorFunctor<lite::TargetType::kHost, T>()(
      ctx,
      *x,
      out,
      *pad_value,
      padded_length,
      0,
      false,
      lite::host::math::kBatchLengthWidth);

  auto* len_data = len_t->template mutable_data<int64_t>();
  auto x_lod = x->lod();
  for (size_t i = 1; i < x_lod[0].size(); i++) {
    len_data[i - 1] = x_lod[0][i] - x_lod[0][i - 1];
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(sequence_pad,
                     kHost,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::host::SequencePadCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindInput("PadValue",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindOutput("Length",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .Finalize();

REGISTER_LITE_KERNEL(sequence_pad,
                     kHost,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::host::SequencePadCompute<int>,
                     int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("PadValue",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Length",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .Finalize();

REGISTER_LITE_KERNEL(sequence_pad,
                     kHost,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::host::SequencePadCompute<int64_t>,
                     int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindInput("PadValue",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindOutput("Length",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .Finalize();
