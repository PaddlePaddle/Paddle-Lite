/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <vector>
#include "lite/core/op_registry.h"
#include "lite/kernels/cuda/search_seq_depadding_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {
using Tensor = lite::Tensor;

#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

template <typename Dtype>
__global__ void ker_sequence_depadding_fwd(Dtype* out_data,
                                           const Dtype* in_data,
                                           const int* seq_id_map,
                                           const int seq_num,
                                           const int max_len,
                                           const int emb_size,
                                           const int count) {
  CUDA_KERNEL_LOOP(tid, count) {
    int emb_id = tid % emb_size;
    int word_id = tid / emb_size;
    int seq_id = seq_id_map[word_id];
    out_data[tid] = in_data[seq_id * emb_size + emb_id];
  }
}

void SearchSeqDepaddingCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<CUDAContext>();
  auto cuda_stream = ctx.exec_stream();

  auto* pad = param.pad;
  auto* src = param.src;
  auto* out = param.out;

  auto* in_data = pad->data<float>();
  out->Resize({src->dims()[0], pad->dims()[1]});
  auto* out_data = out->mutable_data<float>(TARGET(kCUDA));
  const int count = out->numel();

  const auto& pad_seq_offset = pad->lod()[0];
  const auto& src_seq_offset = src->lod()[0];
  int max_len = pad_seq_offset[1];
  int seq_num = pad_seq_offset.size() - 1;
  int emb_size = pad->dims()[1];

  LoD out_lod;
  out_lod.push_back(src_seq_offset);
  out->set_lod(out_lod);
  std::vector<int> seq_id_map;
  for (int i = 0; i < seq_num; i++) {
    int cur_len = src_seq_offset[i + 1] - src_seq_offset[i];
    for (int j = 0; j < cur_len; j++) {
      seq_id_map.push_back(i * max_len + j);
    }
  }

  int map_size = seq_id_map.size();
  seq_id_map_tensor.Resize({map_size, 1, 1, 1});
  int* seq_id_map_data = seq_id_map_tensor.mutable_data<int>(TARGET(kCUDA));
  TargetW::MemcpyAsync(seq_id_map_data,
                       &seq_id_map[0],
                       seq_id_map.size() * sizeof(int),
                       IoDirection::HtoD,
                       cuda_stream);

  int threads = 512;
  int blocks = (count + threads - 1) / threads;
  ker_sequence_depadding_fwd<<<blocks, threads, 0, cuda_stream>>>(
      out_data, in_data, seq_id_map_data, seq_num, max_len, emb_size, count);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) LOG(ERROR) << cudaGetErrorString(error);
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(search_seq_depadding,
                     kCUDA,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::cuda::SearchSeqDepaddingCompute,
                     def)
    .BindInput("Src",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindInput("Pad",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .Finalize();
