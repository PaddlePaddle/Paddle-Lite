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

#include <algorithm>

#include "lite/backends/cuda/math/sequence_padding.h"
#include "lite/core/op_registry.h"
#include "lite/core/target_wrapper.h"
#include "lite/kernels/cuda/sequence_unpad_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

template <typename T, PrecisionType Ptype>
void SequenceUnpadCompute<T, Ptype>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<CUDAContext>();
  auto stream = ctx.exec_stream();

  auto x_dims = param.X->dims();
  auto len_dims = param.Length->dims();

  auto* seq_len_ptr = param.Length->template data<int64_t>();
  seq_len_cpu_.Resize(param.Length->dims());
  TargetWrapperCuda::MemcpyAsync(seq_len_cpu_.mutable_data<int64_t>(),
                                 seq_len_ptr,
                                 sizeof(int64_t) * param.Length->numel(),
                                 IoDirection::DtoH,
                                 stream);
  TargetWrapperCuda::StreamSync(stream);

  int64_t batch_size = len_dims[0];
  std::vector<uint64_t> out_lod0(batch_size + 1, 0);
  for (int64_t i = 0; i < batch_size; ++i) {
    out_lod0[i + 1] = out_lod0[i] + seq_len_cpu_.data<int64_t>()[i];
  }
  paddle::lite::LoD out_lod;
  out_lod.push_back(out_lod0);

  int64_t out_dim0 = out_lod0.back();
  std::vector<int64_t> out_dims{out_dim0};
  if (x_dims.size() == 2) {
    out_dims.push_back(1);
  } else {
    for (size_t i = 2; i < x_dims.size(); ++i) {
      out_dims.push_back(x_dims[i]);
    }
  }
  param.Out->Resize(out_dims);
  param.Out->set_lod(out_lod);

  const auto* pad_tensor = param.X;
  auto* seq_tensor = param.Out;

  int padded_length = pad_tensor->dims()[1];
  int seq_num = seq_tensor->lod()[0].size() - 1;
  int max_seq_len = 0;
  int step_width = seq_tensor->numel() / seq_tensor->dims()[0];

  seq_offsets_vec_.resize(seq_tensor->lod()[0].size());
  for (size_t i = 0; i < seq_num; ++i) {
    max_seq_len = std::max(max_seq_len,
                           static_cast<int>(seq_tensor->lod()[0][i + 1] -
                                            seq_tensor->lod()[0][i]));
    seq_offsets_vec_[i] = seq_tensor->lod()[0][i];
  }
  seq_offsets_vec_[seq_num] = seq_tensor->lod()[0][seq_num];
  seq_offsets_.Resize({static_cast<int64_t>(seq_tensor->lod()[0].size())});
  TargetWrapperCuda::MemcpyAsync(
      seq_offsets_.mutable_data<size_t>(TARGET(kCUDA)),
      seq_offsets_vec_.data(),
      sizeof(size_t) * seq_offsets_vec_.size(),
      IoDirection::HtoD,
      stream);

  const T* pad_data = pad_tensor->template data<T>();
  T* seq_data = seq_tensor->template mutable_data<T>(TARGET(kCUDA));

  lite::cuda::math::SequenceUnpadding(seq_data,
                                      pad_data,
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

using SeqUnadFp32 =
    paddle::lite::kernels::cuda::SequenceUnpadCompute<float, PRECISION(kFloat)>;

using SeqUnadFp16 =
    paddle::lite::kernels::cuda::SequenceUnpadCompute<half, PRECISION(kFP16)>;

REGISTER_LITE_KERNEL(sequence_unpad, kCUDA, kFloat, kNCHW, SeqUnadFp32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindInput("Length",
               {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .Finalize();

REGISTER_LITE_KERNEL(sequence_unpad, kCUDA, kFP16, kNCHW, SeqUnadFp16, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .BindInput("Length",
               {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .Finalize();
