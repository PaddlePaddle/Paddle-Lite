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

#include <algorithm>
#include "lite/backends/cuda/cuda_utils.h"
#include "lite/backends/cuda/math/sequence_padding.h"
#include "lite/backends/cuda/math/utils.h"

namespace paddle {
namespace lite {
namespace cuda {
namespace math {

enum CopyType { kSeqToPad, kPadToSeq };

template <typename T, CopyType Type>
__global__ void SequencePadKernel(T* dst,
                                  const T* src,
                                  const T* pad_value,
                                  bool is_constant_pad,
                                  const size_t* seq_offsets,
                                  const int seq_num,
                                  const int pad_seq_len,
                                  const int step_width) {
  size_t seq_idx = blockIdx.y;
  size_t seq_len = seq_offsets[seq_idx + 1] - seq_offsets[seq_idx];

  size_t step_idx = blockIdx.x * blockDim.y + threadIdx.y;
  size_t seq_data_offset = (seq_offsets[seq_idx] + step_idx) * step_width;
  size_t pad_data_offset = (seq_idx * pad_seq_len + step_idx) * step_width;
  T* dst_data = dst + (Type == kSeqToPad ? pad_data_offset : seq_data_offset);
  const T* src_data =
      src + (Type == kSeqToPad ? seq_data_offset : pad_data_offset);

  if (step_idx < seq_len) {
    for (size_t i = threadIdx.x; i < step_width; i += blockDim.x) {
      dst_data[i] = src_data[i];
    }
  } else if (step_idx < pad_seq_len && Type == kSeqToPad) {
    for (size_t i = threadIdx.x; i < step_width; i += blockDim.x) {
      dst_data[i] = is_constant_pad ? pad_value[0] : pad_value[i];
    }
  }
}

template <typename T>
void SequencePadding(T* pad_data,
                     const T* seq_data,
                     const T* pad_value_data,
                     bool is_constant_pad,
                     const size_t* seq_offsets_data,
                     int seq_num,
                     int pad_seq_len,
                     int step_width,
                     cudaStream_t* stream) {
  const int kBlockSize = 512;
  /* At least use 32 threads to copy sequence_width elements,
   * and at least 8 elements for each thread.
   */
  size_t block_dim_x =
      std::min(((((step_width + 7) >> 3) + 31) >> 5) << 5, kBlockSize);
  size_t block_dim_y = kBlockSize / block_dim_x;
  dim3 threads(block_dim_x, block_dim_y);

  size_t grid_dim_x = (pad_seq_len + block_dim_y - 1) / block_dim_y;
  size_t grid_dim_y = seq_num;
  dim3 grid(grid_dim_x, grid_dim_y);

  SequencePadKernel<T, kSeqToPad><<<grid, threads, 0, *stream>>>(
      pad_data,
      seq_data,
      pad_value_data,
      is_constant_pad,
      seq_offsets_data,
      seq_num,
      pad_seq_len,
      step_width);
  CUDA_POST_KERNEL_CHECK;
}

template <typename T>
void SequenceUnpadding(T* seq_data,
                       const T* pad_data,
                       const size_t* seq_offsets_data,
                       int seq_num,
                       int pad_seq_len,
                       int step_width,
                       cudaStream_t* stream) {
  const int kBlockSize = 512;
  /* At least use 32 threads to copy sequence_width elements,
   * and at least 8 elements for each thread.
   */
  size_t block_dim_x =
      std::min(((((step_width + 7) >> 3) + 31) >> 5) << 5, kBlockSize);
  size_t block_dim_y = kBlockSize / block_dim_x;
  dim3 threads(block_dim_x, block_dim_y);

  size_t grid_dim_x = (pad_seq_len + block_dim_y - 1) / block_dim_y;
  size_t grid_dim_y = seq_num;
  dim3 grid(grid_dim_x, grid_dim_y);

  SequencePadKernel<T, kPadToSeq><<<grid, threads, 0, *stream>>>(
      seq_data,
      pad_data,
      nullptr,
      false,
      seq_offsets_data,
      seq_num,
      pad_seq_len,
      step_width);
  CUDA_POST_KERNEL_CHECK;
}

template void SequencePadding(float* pad_data,
                              const float* seq_data,
                              const float* pad_value_data,
                              bool is_constant_pad,
                              const size_t* seq_offsets_data,
                              int seq_num,
                              int pad_seq_len,
                              int step_width,
                              cudaStream_t* stream);

template void SequencePadding(half* pad_data,
                              const half* seq_data,
                              const half* pad_value_data,
                              bool is_constant_pad,
                              const size_t* seq_offsets_data,
                              int seq_num,
                              int pad_seq_len,
                              int step_width,
                              cudaStream_t* stream);

template void SequenceUnpadding(float* seq_data,
                                const float* pad_data,
                                const size_t* seq_offsets_data,
                                int seq_num,
                                int pad_seq_len,
                                int step_width,
                                cudaStream_t* stream);

template void SequenceUnpadding(half* seq_data,
                                const half* pad_data,
                                const size_t* seq_offsets_data,
                                int seq_num,
                                int pad_seq_len,
                                int step_width,
                                cudaStream_t* stream);

}  // namespace math
}  // namespace cuda
}  // namespace lite
}  // namespace paddle
