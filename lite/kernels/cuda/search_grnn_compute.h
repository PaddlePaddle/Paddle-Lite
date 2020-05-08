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
#include <memory>
#include <vector>
#include "lite/backends/cuda/blas.h"
#include "lite/backends/cuda/math/gemm.h"
#include "lite/core/kernel.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

class SeqSortedseqTranseUtil {
 public:
  explicit SeqSortedseqTranseUtil(bool is_reverse = false, bool is_bi = false)
      : _is_reverse(is_reverse),
        _is_bi(is_bi),
        _dev_map_vec(nullptr),
        _dev_map_vec_length(0) {}

  ~SeqSortedseqTranseUtil() {
    if (_dev_map_vec != nullptr) {
      TargetWrapperCuda::Free(static_cast<void*>(_dev_map_vec));
    }
  }

  std::vector<int>& get_length_index() { return _length_index; }
  std::vector<int>& get_emit_offset_vec() { return _emit_offset_vec; }
  std::vector<int>& get_map_vec() { return _map_vec; }
  int* get_dev_map_vec() { return _dev_map_vec; }
  int get_emit_length() { return _emit_length; }

  template <typename Dtype>
  void seq_2_sorted_seq(const Dtype* input,
                        Dtype* output,
                        int word_size,
                        cudaStream_t stream);

  template <typename Dtype>
  void sorted_seq_2_seq(const Dtype* input,
                        Dtype* output,
                        int hidden_size,
                        cudaStream_t stream);

  bool get_sorted_map(const std::vector<int>& offset_vec,
                      cudaStream_t stream_id);

 private:
  std::vector<int> _length_index;
  std::vector<int> _emit_offset_vec;
  std::vector<int> _map_vec;
  int _emit_length;

  bool _is_reverse;
  bool _is_bi;
  int* _dev_map_vec;
  int _dev_map_vec_length;
};

class SearchGrnnCompute
    : public KernelLite<TARGET(kCUDA), PRECISION(kFloat), DATALAYOUT(kNCHW)> {
 public:
  using param_t = operators::SearchGrnnParam;
  using TargetW = TargetWrapper<TARGET(kCUDA)>;

  void PrepareForRun() override;
  void Run() override;
  virtual ~SearchGrnnCompute() = default;

 private:
  // Weights preprocess:
  // wi need to be transpose, the axes should be (2, 0, 1)
  // wh0 should transpose, {wh1 wh2} need be transpose, the axes should be {2,
  // 0, 1}
  void WeightsPreprocess();

 private:
  std::unique_ptr<lite::cuda::math::Gemm<float, float>> gemm_impl_;

  lite::Tensor _temp_tensor_in;
  lite::Tensor _temp_tensor_out;
  lite::Tensor _temp_wx;
  lite::Tensor _temp_wh;
  lite::Tensor _temp_zero;
  lite::Tensor _temp_weights_h2h;

  lite::Tensor _wi;
  lite::Tensor _wh;

  SeqSortedseqTranseUtil _seq_util;
};

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
