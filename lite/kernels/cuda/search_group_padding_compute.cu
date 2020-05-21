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
#include "lite/backends/cuda/cuda_utils.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/cuda/search_group_padding_compute.h"

#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {
using Tensor = lite::Tensor;

template <typename Dtype>
__global__ void ker_search_group_padding(Dtype* out_emb_padding_data,
                                         Dtype* out_padding_data,
                                         const Dtype* in_data,
                                         const uint64_t* offset,
                                         const int seq_num,
                                         const int max_len,
                                         const int emb_size,
                                         const Dtype pad_id,
                                         const int count) {
  CUDA_KERNEL_LOOP(tid, count) {
    int emb_id = tid % emb_size;
    int word_id = tid / emb_size;
    int seq_id = word_id / max_len;
    int word_id_in_seq = word_id % max_len;
    int cur_len = offset[seq_id + 1] - offset[seq_id];
    if (word_id_in_seq < cur_len) {
      out_emb_padding_data[tid] =
          in_data[(offset[seq_id] + word_id_in_seq) * emb_size + emb_id];
    } else {
      out_emb_padding_data[tid] = 0.f;
      if (emb_id == 0) {
        out_padding_data[word_id] = pad_id;
      }
    }
  }
}

void SearchGroupPaddingCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<CUDAContext>();
  auto cuda_stream = ctx.exec_stream();

  const Tensor* x = param.x;
  Tensor* out_emb_padding = param.out_emb_padding;
  Tensor* out_new = param.out_new;
  Tensor* out_padding = param.out_padding;
  const float pad_id = static_cast<float>(param.pad_id);
  const float* in_data = x->data<float>();
  const auto& in_seq_offset = x->lod()[0];
  int batch = in_seq_offset.size() - 1;
  int max_seq = 0;
  for (int i = 0; i < batch; ++i) {
    if (in_seq_offset[i + 1] - in_seq_offset[i] > max_seq) {
      max_seq = in_seq_offset[i + 1] - in_seq_offset[i];
    }
  }
  std::vector<size_t> new_offset;
  new_offset.resize(batch + 1);
  for (int i = 0; i < batch + 1; ++i) {
    new_offset[i] = i * max_seq;
  }
  std::vector<int64_t> x_dims = x->dims().Vectorize();
  LoD out_emb_padding_lod;
  out_emb_padding_lod.push_back(new_offset);
  out_emb_padding->set_lod(out_emb_padding_lod);
  out_emb_padding->Resize({batch * max_seq, x_dims[1]});
  float* out_emb_padding_data =
      out_emb_padding->mutable_data<float>(TARGET(kCUDA));

  LoD out_new_lod;
  out_new_lod.push_back(in_seq_offset);
  out_new->set_lod(out_new_lod);
  out_new->Resize({x_dims[0], 1});

  LoD out_padding_lod;
  out_padding_lod.push_back(new_offset);
  out_padding->set_lod(out_padding_lod);
  out_padding->Resize({batch * max_seq, 1});
  float* out_padding_data = out_padding->mutable_data<float>(TARGET(kCUDA));

  const int count = out_emb_padding->numel();
  const auto& out_emb_padding_seq_offset = out_emb_padding->lod()[0];
  int max_len = out_emb_padding_seq_offset[1];
  int seq_num = out_emb_padding_seq_offset.size() - 1;
  int emb_size = x->dims()[1];
  _in_seq_offset.Resize({seq_num + 1, 1, 1, 1});
  uint64_t* offset_data = _in_seq_offset.mutable_data<uint64_t>(TARGET(kCUDA));

  TargetWrapperCuda::MemcpyAsync(offset_data,
                                 in_seq_offset.data(),
                                 sizeof(uint64_t) * in_seq_offset.size(),
                                 IoDirection::HtoD,
                                 cuda_stream);

  TargetWrapperCuda::MemsetAsync(
      out_padding_data,
      0,
      out_padding->dims()[0] * out_padding->dims()[1] * sizeof(float),
      cuda_stream);

  ker_search_group_padding<
      float><<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(
      out_emb_padding_data,
      out_padding_data,
      in_data,
      offset_data,
      seq_num,
      max_len,
      emb_size,
      pad_id,
      count);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) LOG(INFO) << cudaGetErrorString(error);
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(search_group_padding,
                     kCUDA,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::cuda::SearchGroupPaddingCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out_emb_padding",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .BindOutput("Out_new",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .BindOutput("Out_padding",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .Finalize();
