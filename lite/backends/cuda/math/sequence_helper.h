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

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include "lite/backends/cuda/target_wrapper.h"

namespace paddle {
namespace lite {
namespace cuda {
namespace math {

class SeqSortedseqTranseUtil {
 public:
  explicit SeqSortedseqTranseUtil(bool is_reverse = false, bool is_bi = false)
      : is_reverse_(is_reverse),
        is_bi_(is_bi),
        dev_map_vec_(nullptr),
        dev_map_vec_length_(0) {}

  ~SeqSortedseqTranseUtil() {
    if (dev_map_vec_ != nullptr) {
      TargetWrapperCuda::Free(static_cast<void*>(dev_map_vec_));
    }
  }

  std::vector<int>& GetLengthIndex() { return length_index_; }
  std::vector<int>& GetEmitOffsetVec() { return emit_offset_vec_; }
  std::vector<int>& GetMapVec() { return map_vec_; }
  int* GetDevMapVec() { return dev_map_vec_; }
  int GetEmitLength() { return emit_length_; }

  template <typename Dtype>
  void Seq2SortedSeq(const Dtype* input,
                     Dtype* output,
                     int word_size,
                     cudaStream_t stream);

  template <typename Dtype>
  void SortedSeq2Seq(const Dtype* input,
                     Dtype* output,
                     int hidden_size,
                     cudaStream_t stream);

  bool GetSortedMap(const std::vector<int>& offset_vec, cudaStream_t stream_id);

 private:
  std::vector<int> length_index_;
  std::vector<int> emit_offset_vec_;
  std::vector<int> map_vec_;
  int emit_length_;

  bool is_reverse_;
  bool is_bi_;
  int* dev_map_vec_;
  int dev_map_vec_length_;
};

}  // namespace math
}  // namespace cuda
}  // namespace lite
}  // namespace paddle
