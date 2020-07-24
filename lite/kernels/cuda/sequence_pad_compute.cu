// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/backends/cuda/math/sequence_padding.h"
#include "lite/core/op_registry.h"
#include "lite/core/target_wrapper.h"
#include "lite/kernels/cuda/sequence_pad_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

template <typename T, PrecisionType Ptype>
void SequencePadCompute<T, Ptype>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<CUDAContext>();
  auto stream = ctx.exec_stream();

  const auto* x = param.X;
  const auto* pad_value = param.PadValue;
  auto* out = param.Out;
  auto* len_t = param.Length;
  int seq_num = x->lod()[0].size() - 1;
  int padded_length;
  if (param.padded_length == -1) {
    int max_seq_len = 0;
    for (int i = 0; i < seq_num; ++i) {
      max_seq_len = std::max(
          max_seq_len, static_cast<int>(x->lod()[0][i + 1] - x->lod()[0][i]));
    }
    padded_length = max_seq_len;
  } else {
    padded_length = param.padded_length;
  }

  int max_seq_len = 0;
  int step_width = x->numel() / x->dims()[0];

  // calc for param.Lenght
  seq_len_.resize(seq_num);
  seq_offsets_vec_.resize(x->lod()[0].size());
  for (size_t i = 0; i < seq_num; ++i) {
    max_seq_len = std::max(
        max_seq_len, static_cast<int>(x->lod()[0][i + 1] - x->lod()[0][i]));
    seq_len_[i] = x->lod()[0][i + 1] - x->lod()[0][i];
    seq_offsets_vec_[i] = x->lod()[0][i];
  }
  seq_offsets_vec_[seq_num] = x->lod()[0][seq_num];
  TargetWrapperCuda::MemcpyAsync(
      len_t->template mutable_data<int64_t>(TARGET(kCUDA)),
      seq_len_.data(),
      sizeof(int64_t) * seq_len_.size(),
      IoDirection::HtoD,
      stream);
  seq_offsets_.Resize({static_cast<int64_t>(x->lod()[0].size())});
  TargetWrapperCuda::MemcpyAsync(
      seq_offsets_.mutable_data<size_t>(TARGET(kCUDA)),
      seq_offsets_vec_.data(),
      sizeof(size_t) * seq_offsets_vec_.size(),
      IoDirection::HtoD,
      stream);

  const T* seq_data = x->template data<T>();
  T* pad_data = out->template mutable_data<T>(TARGET(kCUDA));
  const T* pad_value_data = pad_value->template data<T>();

  lite::cuda::math::SequencePadding(pad_data,
                                    seq_data,
                                    pad_value_data,
                                    pad_value->numel() == 1,
                                    seq_offsets_.data<size_t>(),
                                    seq_num,
                                    padded_length,
                                    step_width,
                                    &stream);
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using SeqPadFp32 =
    paddle::lite::kernels::cuda::SequencePadCompute<float, PRECISION(kFloat)>;

using SeqPadFp16 =
    paddle::lite::kernels::cuda::SequencePadCompute<half, PRECISION(kFP16)>;

REGISTER_LITE_KERNEL(sequence_pad, kCUDA, kFloat, kNCHW, SeqPadFp32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindInput("PadValue", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("Length",
                {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kInt64))})
    .Finalize();

REGISTER_LITE_KERNEL(sequence_pad, kCUDA, kFP16, kNCHW, SeqPadFp16, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .BindInput("PadValue",
               {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .BindOutput("Length",
                {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kInt64))})
    .Finalize();
