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
#include "lite/backends/cuda/math/sequence_helper.h"
#include "lite/backends/cuda/math/utils.h"

namespace paddle {
namespace lite {
namespace cuda {
namespace math {

template <typename Dtype>
__global__ void Map2Out(
    Dtype* output, const Dtype* input, const int* map, int count, int lastdim) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < count) {
    int seq = tid / lastdim;
    output[map[seq] * lastdim + tid % lastdim] = input[tid];
  }
}

template <typename Dtype>
__global__ void Map2In(
    Dtype* output, const Dtype* input, const int* map, int count, int lastdim) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < count) {
    int seq = tid / lastdim;
    output[tid] = input[map[seq] * lastdim + tid % lastdim];
  }
}

template <typename Dtype>
void Map2OutFunc(const Dtype* input,
                 Dtype* output,
                 int word_size,
                 int seq_sum,
                 cudaStream_t stream,
                 int* dev_map_vec) {
  int count = seq_sum * word_size;
  int block_dim = count;
  int grid_dim = 1;

  if (count > 1024) {
    block_dim = 256;
    grid_dim = (count + block_dim - 1) / block_dim;
  }

  Map2Out<<<grid_dim, block_dim, 0, stream>>>(
      output, input, dev_map_vec, count, word_size);
}

template <typename Dtype>
void Map2InFunc(const Dtype* input,
                Dtype* output,
                int hidden_size,
                int seq_sum,
                cudaStream_t stream,
                int* dev_map_vec) {
  int count = seq_sum * hidden_size;
  int block_dim = count;
  int grid_dim = 1;
  if (count > 1024) {
    block_dim = 256;
    grid_dim = (count + block_dim - 1) / block_dim;
  }

  Map2In<<<grid_dim, block_dim, 0, stream>>>(
      output, input, dev_map_vec, count, hidden_size);
}

template <typename Dtype>
void SeqSortedseqTranseUtil::Seq2SortedSeq(const Dtype* input,
                                           Dtype* output,
                                           int word_size,
                                           cudaStream_t stream) {
  int seq_sum = map_vec_.size();
  Map2OutFunc(input, output, word_size, seq_sum, stream, dev_map_vec_);
}

template <typename Dtype>
void SeqSortedseqTranseUtil::SortedSeq2Seq(const Dtype* input,
                                           Dtype* output,
                                           int hidden_size,
                                           cudaStream_t stream) {
  int seq_sum = map_vec_.size();
  Map2InFunc(input, output, hidden_size, seq_sum, stream, dev_map_vec_);
}

bool SeqSortedseqTranseUtil::GetSortedMap(const std::vector<int>& offset_vec,
                                          cudaStream_t stream_id) {
  int batch_size = offset_vec.size() - 1;
  int word_sum = offset_vec[offset_vec.size() - 1];
  std::vector<int> length_vec(batch_size);
  length_index_.resize(batch_size);
  int emit_length = 0;

  if (batch_size == 1) {
    emit_length = offset_vec[1] - offset_vec[0];
    emit_offset_vec_.resize(emit_length + 1);

    for (int i = 0; i <= emit_length; ++i) {
      emit_offset_vec_[i] = i;
    }

    return false;
  }

  int max_len = 0;

  for (int i = 0; i < offset_vec.size() - 1; ++i) {
    int len = offset_vec[i + 1] - offset_vec[i];
    max_len = max_len > len ? max_len : len;
    length_vec[i] = len;
    length_index_[i] = i;
  }

  emit_length = max_len;

  if (max_len == 1) {
    emit_offset_vec_.resize(2);
    emit_offset_vec_[0] = 0;
    emit_offset_vec_[1] = emit_length * batch_size;
    return false;
  }

  std::stable_sort(length_index_.begin(),
                   length_index_.end(),
                   [&length_vec](int i1, int i2) {
                     return length_vec[i1] > length_vec[i2];
                   });

  emit_offset_vec_.resize(max_len + 1);
  map_vec_.resize(word_sum);

  if (word_sum > dev_map_vec_length_) {
    if (dev_map_vec_ != nullptr) {
      TargetWrapperCuda::Free(static_cast<void*>(dev_map_vec_));
    }

    dev_map_vec_ =
        static_cast<int*>(TargetWrapperCuda::Malloc(sizeof(int) * word_sum));
    dev_map_vec_length_ = word_sum;
  }

  int target_word_id = 0;
  std::vector<int> length_vec_cnt = length_vec;
  int last_batch_size = batch_size;
  for (int word_id_in_seq = 0; word_id_in_seq < max_len; word_id_in_seq++) {
    emit_offset_vec_[word_id_in_seq] = target_word_id;

    for (int batch_id = 0; batch_id < last_batch_size; batch_id++) {
      int old_batch_id = length_index_[batch_id];

      if (length_vec_cnt[old_batch_id] > 0) {
        int inner_word_id_in_seq = word_id_in_seq;

        if (is_reverse_) {
          inner_word_id_in_seq = length_vec[old_batch_id] - 1 - word_id_in_seq;
        }

        int old_word_id = offset_vec[old_batch_id] + inner_word_id_in_seq;
        map_vec_[old_word_id] = target_word_id;
        length_vec_cnt[old_batch_id]--;
        target_word_id++;
      } else {
        last_batch_size--;
        break;
      }
    }
  }

  TargetWrapperCuda::MemcpyAsync(dev_map_vec_,
                                 map_vec_.data(),
                                 sizeof(int) * word_sum,
                                 IoDirection::HtoD,
                                 stream_id);
  emit_offset_vec_[max_len] = word_sum;
  emit_length_ = emit_length;
  return true;
}

template void SeqSortedseqTranseUtil::Seq2SortedSeq(const float* input,
                                                    float* output,
                                                    int word_size,
                                                    cudaStream_t stream);
template void SeqSortedseqTranseUtil::SortedSeq2Seq(const float* input,
                                                    float* output,
                                                    int hidden_size,
                                                    cudaStream_t stream);
template void SeqSortedseqTranseUtil::Seq2SortedSeq(const half* input,
                                                    half* output,
                                                    int word_size,
                                                    cudaStream_t stream);
template void SeqSortedseqTranseUtil::SortedSeq2Seq(const half* input,
                                                    half* output,
                                                    int hidden_size,
                                                    cudaStream_t stream);

}  // namespace math
}  // namespace cuda
}  // namespace lite
}  // namespace paddle
