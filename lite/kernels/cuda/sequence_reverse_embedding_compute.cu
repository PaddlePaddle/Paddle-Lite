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

#include "lite/backends/cuda/cuda_utils.h"
#include "lite/core/op_registry.h"
#include "lite/core/target_wrapper.h"
#include "lite/kernels/cuda/sequence_reverse_embedding_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

template <typename T>
__host__ __device__ inline size_t UpperBound(const T* x,
                                             const int num,
                                             const T& val) {
  // The following code is from
  // https://en.cppreference.com/w/cpp/algorithm/upper_bound
  auto* first = x;
  int64_t count = static_cast<int64_t>(num);
  while (count > 0) {
    auto step = (count >> 1);
    auto* it = first + step;
    if (val < *it) {
      count = step;
    } else {
      first = ++it;
      count -= (step + 1);
    }
  }
  return static_cast<size_t>(first - x);
}

template <typename T>
__global__ void SequenceReverseEmbeddingKernel(const int64_t* ids,
                                               const T* table,
                                               T* out,
                                               const int64_t* lod,
                                               const int lod_count,
                                               const int width,
                                               const int count,
                                               const bool padding_flag,
                                               const int64_t padding_idx) {
  CUDA_KERNEL_LOOP(tid, count) {
    int64_t row = tid / width;
    int col = tid % width;
    auto lod_idx = UpperBound(lod, lod_count, row);
    auto reverse_row = lod[lod_idx - 1] + lod[lod_idx] - 1 - row;

    if (padding_flag) {
      if (ids[reverse_row] == padding_idx)
        out[tid] = 0;
      else
        out[tid] = table[ids[reverse_row] * width + col];
    } else {
      out[tid] = table[ids[reverse_row] * width + col];
    }
  }
}

template <typename T, PrecisionType Ptype>
void SequenceReverseEmbeddingCompute<T, Ptype>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<CUDAContext>();
  auto stream = ctx.exec_stream();
  auto io_stream = ctx.io_stream();

  auto* table_data = param.W->template data<T>();
  auto* out_data = param.Out->template mutable_data<T>(TARGET(kCUDA));
  auto* ids_data = param.Ids->template data<int64_t>();
  const auto lod = param.Ids->lod()[param.Ids->lod().size() - 1];
  const int lod_count = lod.size();
  const int width = param.W->dims()[1];
  const int count = param.Out->numel();

  lod_info_.Resize({static_cast<int64_t>(lod.size())});
  int64_t* lod_data = lod_info_.mutable_data<int64_t>(TARGET(kCUDA));
  TargetWrapperCuda::MemcpyAsync(lod_data,
                                 lod.data(),
                                 sizeof(int64_t) * lod.size(),
                                 IoDirection::HtoD,
                                 stream);

  int64_t padding_idx = param.padding_idx;
  bool padding_flag = padding_idx != -1;
  SequenceReverseEmbeddingKernel<
      T><<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(ids_data,
                                                                  table_data,
                                                                  out_data,
                                                                  lod_data,
                                                                  lod_count,
                                                                  width,
                                                                  count,
                                                                  padding_flag,
                                                                  padding_idx);
  CUDA_POST_KERNEL_CHECK;
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using SeqReverseEmbFp32 = paddle::lite::kernels::cuda::
    SequenceReverseEmbeddingCompute<float, PRECISION(kFloat)>;

REGISTER_LITE_KERNEL(
    sequence_reverse_embedding, kCUDA, kFloat, kNCHW, SeqReverseEmbFp32, def)
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFloat))})
    .BindInput("Ids", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kInt64))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFloat))})
    .Finalize();
