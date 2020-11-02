/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "lite/backends/host/math/sequence_padding.h"

namespace paddle {
namespace lite {
namespace host {
namespace math {

template <typename T>
void CopyValidData(lite::Tensor* dst_tensor,
                   const lite::Tensor* src_tensor,
                   const std::vector<uint64_t>& seq_offsets,
                   int pad_seq_len,
                   int step_width,
                   bool norm_by_len,
                   CopyType type,
                   PadLayout layout) {
  int seq_num = seq_offsets.size() - 1;
  const T* src_data = src_tensor->template data<T>();
  T* dst_data = dst_tensor->template mutable_data<T>();

  int seq_cpy_gap = step_width;
  int pad_cpy_gap =
      layout == kBatchLengthWidth ? step_width : seq_num * step_width;
  for (int seq_idx = 0; seq_idx < seq_num; ++seq_idx) {
    int valid_seq_len = seq_offsets[seq_idx + 1] - seq_offsets[seq_idx];
    CHECK_GE(pad_seq_len, valid_seq_len) << "The padded sequence length can "
                                            "not be less than its original "
                                            "length.";
    int seq_data_offset = seq_offsets[seq_idx] * step_width;
    int pad_data_offset = layout == kBatchLengthWidth
                              ? seq_idx * pad_seq_len * step_width
                              : seq_idx * step_width;
    float scale = 1.0f / static_cast<float>(valid_seq_len);

    for (int step_idx = 0; step_idx < valid_seq_len; ++step_idx) {
      const T* src =
          src_data + (type == kSeqToPad ? seq_data_offset : pad_data_offset);
      T* dst =
          dst_data + (type == kSeqToPad ? pad_data_offset : seq_data_offset);
      memcpy(dst, src, step_width * sizeof(T));
      if (norm_by_len) {
        for (int i = 0; i < step_width; ++i) {
          *(dst + i) *= scale;
        }
      }
      seq_data_offset += seq_cpy_gap;
      pad_data_offset += pad_cpy_gap;
    }
  }
}

template <typename T>
static void fast_mem_init(void* dest,
                          size_t dest_size,
                          const T* src,
                          size_t num_bytes) {
  if (dest == nullptr || dest_size == 0 || src == nullptr) return;

  memcpy(dest, src, num_bytes);

  dest_size *= num_bytes;
  while (dest_size > num_bytes) {
    size_t remaining = dest_size - num_bytes;
    size_t count = (remaining > num_bytes) ? num_bytes : remaining;
    memcpy((unsigned char*)dest + num_bytes, dest, count);
    num_bytes += count;
  }
}

template <typename T>
class PaddingLoDTensorFunctor<lite::TargetType::kHost, T> {
 public:
  void operator()(const lite::Context<lite::TargetType::kHost>& context,
                  const lite::Tensor& seq_tensor,
                  lite::Tensor* pad_tensor,
                  const lite::Tensor& pad_value,
                  int pad_seq_len = -1,
                  int lod_level = 0,
                  bool norm_by_times = false,
                  const PadLayout layout = kBatchLengthWidth) {
    auto seq_lod = seq_tensor.lod();
    const auto seq_offsets = lite::fluid::ToAbsOffset(seq_lod)[lod_level];
    const auto& seq_tensor_dims = seq_tensor.dims();
    const auto& pad_tensor_dims = pad_tensor->dims();
    if (pad_seq_len == -1) {
      pad_seq_len = MaximumSequenceLength(seq_offsets);
    }
    int step_width = seq_tensor.numel() / seq_tensor_dims[0];

    CheckDims(seq_tensor_dims,
              pad_tensor_dims,
              seq_offsets,
              pad_seq_len,
              step_width,
              layout);
    CHECK(pad_value.numel() == 1 || pad_value.numel() == step_width)
        << "The numel of 'pad_value' can only be 1 or be equal to the "
           "'step_width'.";

    // fill padding value
    T* pad_data = pad_tensor->template mutable_data<T>();
    const T* pad_value_data = pad_value.data<T>();
    if (pad_value.numel() == 1) {
      fast_mem_init<T>(
          pad_data, pad_tensor->numel(), pad_value_data, sizeof(T));
    } else {
      for (int i = 0; i < pad_tensor->numel(); i += step_width) {
        memcpy(pad_data + i, pad_value_data, step_width * sizeof(T));
      }
    }

    CopyValidData<T>(pad_tensor,
                     &seq_tensor,
                     seq_offsets,
                     pad_seq_len,
                     step_width,
                     norm_by_times,
                     kSeqToPad,
                     layout);
  }
};

template <typename T>
class UnpaddingLoDTensorFunctor<lite::TargetType::kHost, T> {
 public:
  void operator()(const lite::Context<lite::TargetType::kHost>& context,
                  const lite::Tensor& pad_tensor,
                  lite::Tensor* seq_tensor,
                  int pad_seq_len = -1,
                  int lod_level = 0,
                  bool norm_by_times = false,
                  const PadLayout layout = kBatchLengthWidth) {
    auto seq_offsets = lite::fluid::ToAbsOffset(seq_tensor->lod())[lod_level];
    const auto& seq_tensor_dims = seq_tensor->dims();
    const auto& pad_tensor_dims = pad_tensor.dims();
    if (pad_seq_len == -1) {
      pad_seq_len = MaximumSequenceLength(seq_offsets);
    }
    int step_width = seq_tensor->numel() / seq_tensor_dims[0];

    CheckDims(seq_tensor_dims,
              pad_tensor_dims,
              seq_offsets,
              pad_seq_len,
              step_width,
              layout);

    CopyValidData<T>(seq_tensor,
                     &pad_tensor,
                     seq_offsets,
                     pad_seq_len,
                     step_width,
                     norm_by_times,
                     kPadToSeq,
                     layout);
  }
};

template class PaddingLoDTensorFunctor<lite::TargetType::kHost, int>;
template class PaddingLoDTensorFunctor<lite::TargetType::kHost, int64_t>;
template class PaddingLoDTensorFunctor<lite::TargetType::kHost, float>;
template class PaddingLoDTensorFunctor<lite::TargetType::kHost, double>;

template class UnpaddingLoDTensorFunctor<lite::TargetType::kHost, int>;
template class UnpaddingLoDTensorFunctor<lite::TargetType::kHost, int64_t>;
template class UnpaddingLoDTensorFunctor<lite::TargetType::kHost, float>;
template class UnpaddingLoDTensorFunctor<lite::TargetType::kHost, double>;

}  // namespace math
}  // namespace host
}  // namespace lite
}  // namespace paddle
