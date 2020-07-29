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

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <string>
#include <vector>

#include "lite/backends/cuda/cuda_utils.h"
#include "lite/core/context.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace cuda {
namespace math {

template <typename T>
class CopyMatrixRowsFunctor {
 public:
  // If is_src_index is true, copy the indexed rows of input src to the output
  // dst. If is_src_index is false, copy the input src to the indexed of output
  // dst. The indexes rows are based on the input index.
  void operator()(const lite::Tensor& src,
                  lite::Tensor* dst,
                  const std::vector<uint64_t>& index_lod,
                  bool is_src_index,
                  const cudaStream_t& stream);

 private:
  lite::Tensor index_tensor_;
};

template <typename T>
class LoDTensor2BatchFunctor {
  // Calculate the length of each sequence and
  // sort sequence index by the length.
  // example:  sequences = {s0, s1, s2}
  //            s0: 0 0 0 0, s1: 1 1 1 1 1, s2: 2 2 2
  //            seq_info[3] = {(4, 5, 1), (0, 4, 0), (9, 3, 2)}
  struct SeqInfo {
    SeqInfo(size_t start_val, size_t len_val, size_t seq_val)
        : start(start_val), length(len_val), seq_idx(seq_val) {}
    size_t start;
    size_t length;
    size_t seq_idx;
  };

 public:
  void operator()(const lite::Tensor& lod_tensor,
                  lite::Tensor* batch_tensor,
                  bool is_reverse,
                  const cudaStream_t& stream) const {
    auto lods = lod_tensor.lod();
    CHECK_EQ(lods.size(), 1UL) << "Only support one level sequence now.";
    const auto& lod = lods[0];

    std::vector<SeqInfo> seq_info;
    for (int seq_id = 0; seq_id < static_cast<int>(lod.size()) - 1; ++seq_id) {
      size_t length = lod[seq_id + 1] - lod[seq_id];
      seq_info.emplace_back(lod[seq_id], length, seq_id);
    }

    std::sort(seq_info.begin(), seq_info.end(), [](SeqInfo a, SeqInfo b) {
      return a.length > b.length;
    });

    // Calculate the start position of each batch.
    // example:  sequences = {s0, s1, s2}
    //           s0: 0 0 0 0, s1: 1 1 1 1 1, s2: 2 2 2
    //           max_seqlen = 5,
    //           batchIndex = {b0, b1, b2, b3, b4}
    //           b0: 1 0 2, b1: 1 0 2, b2: 1 0 2, b3: 1 0, b4: 1
    //           batch_start_positions[6] = {0, 3, 6, 9, 11, 12}
    //              batch_start_positions[0] = 0
    //              batch_start_positions[1] = len(b0)
    //              batch_start_positions[2] = len(b0) + len(b1)
    //              ...
    //           seq2batch_idx[12] = {4, 0, 9,
    //                                5, 1, 10,
    //                                6, 2, 11,
    //                                7, 3,
    //                                8}
    //           seq_order = {1, 0, 2}, the sort order.
    //               where 1 is the second sequence,
    //                     0 is the first sequence,
    //                     2 is the third sequence.

    LoD batch_lods;
    batch_lods.emplace_back(std::vector<uint64_t>{0});
    batch_lods.emplace_back(std::vector<uint64_t>{0});
    batch_lods.emplace_back(std::vector<uint64_t>{0});

    // batch_lods[0] is the start positions for batch LoDTensor
    size_t max_seqlen = seq_info[0].length;
    batch_lods[0].resize(max_seqlen + 1);
    // batch_lods[1] is the raw index in the input LoDTensor
    batch_lods[1].resize(static_cast<size_t>(lod_tensor.dims()[0]));
    // batch_lods[2] is the sort order for the input LoDTensor.
    batch_lods[2].resize(seq_info.size());

    auto* batch_starts = batch_lods[0].data();
    auto* seq2batch_idx = batch_lods[1].data();
    batch_starts[0] = 0;
    for (size_t n = 0; n < max_seqlen; ++n) {
      size_t batch_id = batch_starts[n];
      for (size_t i = 0; i < seq_info.size(); ++i) {
        size_t seq_len = seq_info[i].length;
        size_t start = seq_info[i].start;
        if (n < seq_len) {
          seq2batch_idx[batch_id] =
              is_reverse ? start + seq_len - 1 - n : start + n;
          ++batch_id;
        } else {
          break;
        }
      }
      batch_starts[n + 1] = batch_id;
    }
    auto* seq_order = batch_lods[2].data();
    for (size_t i = 0; i < seq_info.size(); ++i) {
      seq_order[i] = seq_info[i].seq_idx;
    }

    batch_tensor->set_lod(batch_lods);

    lite::cuda::math::CopyMatrixRowsFunctor<T> to_batch;
    to_batch(lod_tensor, batch_tensor, batch_lods[1], true, stream);
    CUDA_POST_KERNEL_CHECK;
  }
};

template <typename T>
class Batch2LoDTensorFunctor {
 public:
  void operator()(const lite::Tensor& batch_tensor,
                  lite::Tensor* lod_tensor,
                  const cudaStream_t& stream) {
    auto in_lod = batch_tensor.lod();
    CHECK_GT(in_lod.size(), 2UL) << "The LoD of LoDTensor should include at "
                                    "least 2-level sequence infomation.";
    CHECK_EQ(in_lod[1].size(), static_cast<size_t>(lod_tensor->dims()[0]))
        << "The LoD information should be consistent with the dims.";
    lite::cuda::math::CopyMatrixRowsFunctor<T> to_seq;
    to_seq(batch_tensor, lod_tensor, in_lod[1], false, stream);
    CUDA_POST_KERNEL_CHECK;
  }
};

}  // namespace math
}  // namespace cuda
}  // namespace lite
}  // namespace paddle
