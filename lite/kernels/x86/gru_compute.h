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

#include <string>
#include <vector>
#include "lite/backends/x86/fluid/eigen.h"
#include "lite/backends/x86/math/blas.h"
#include "lite/backends/x86/math/gru_compute.h"
#include "lite/backends/x86/math/gru_cpu_kernel.h"
#include "lite/backends/x86/math/gru_kernel.h"
#include "lite/backends/x86/math/math_function.h"
#include "lite/backends/x86/math/sequence2batch.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/core/types.h"

// DECLARE_int32(paddle_num_threads);
extern int32_t paddle_num_threads;

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

using Tensor = lite::Tensor;

template <typename T>
inline void ReorderInitState(const lite::Context<TARGET(kX86)>& context,
                             const Tensor& src,
                             const std::vector<uint64_t>& index_lod,
                             Tensor* dst,
                             bool indexed_src) {
  lite::x86::math::CopyMatrixRowsFunctor<TARGET(kX86), T> row_shuffle;
  dst->Resize(src.dims());
  dst->template mutable_data<T>();
  row_shuffle(context, src, index_lod, dst, indexed_src);
}

static inline int64_t CalculateSeqWidth(const DDim& dims) {
  return dims.count(1, dims.size());
}

template <typename T>
class GRUCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  void Run() override {
    auto& context = ctx_->As<X86Context>();
    auto& param = *param_.get_mutable<operators::GRUParam>();

    bool origin_mode = param.origin_mode;
    bool is_reverse = param.is_reverse;

    auto* input = param.input;
    auto* h0 = param.h0;
    auto* weight = param.weight;
    const T* weight_data = weight->template data<T>();
    auto* bias = param.bias;

    auto* batch_gate = param.batch_gate;
    auto* batch_reset_hidden_prev = param.batch_reset_hidden_prev;
    auto* batch_hidden = param.batch_hidden;
    T* batch_gate_ptr = batch_gate->template mutable_data<T>();
    T* batch_reset_hidden_prev_ptr =
        batch_reset_hidden_prev->template mutable_data<T>();
    T* batch_hidden_ptr = batch_hidden->template mutable_data<T>();

    auto* hidden = param.hidden;
    hidden->template mutable_data<T>();

    const auto& hidden_dims = hidden->dims();

    lite::x86::math::LoDTensor2BatchFunctor<TARGET(kX86), T> to_batch;
    to_batch(context, *input, batch_gate, true, is_reverse);

    if (bias) {
      lite::x86::math::RowwiseAdd<TARGET(kX86), T> add_bias;
      add_bias(context, *batch_gate, *bias, batch_gate);
    }

    int frame_size = hidden_dims[1];
    lite::x86::math::GRUMetaValue<T> gru_value;
    gru_value.gate_weight = const_cast<T*>(weight_data);
    gru_value.state_weight =
        const_cast<T*>(weight_data + 2 * frame_size * frame_size);
    Tensor ordered_h0;

    if (h0) {
      // Since the batch computing for GRU reorders the input sequences
      // according to their length. The initialized cell state also needs
      // to reorder.
      const std::vector<uint64_t>& order(batch_gate->lod()[2]);
      ReorderInitState<T>(context, *h0, order, &ordered_h0, true);
      gru_value.prev_out_value = ordered_h0.mutable_data<T>();
    } else {
      gru_value.prev_out_value = nullptr;
    }

    const auto& batch_starts = batch_gate->lod()[0];
    size_t seq_len = batch_starts.size() - 1;
    int64_t batch_gate_width = CalculateSeqWidth(batch_gate->dims());
    int64_t batch_reset_hidden_prev_width =
        CalculateSeqWidth(batch_reset_hidden_prev->dims());
    int64_t batch_hidden_width = CalculateSeqWidth(batch_hidden->dims());
    auto active_node =
        lite::x86::math::detail::GetActivationType(param.activation);
    auto active_gate =
        lite::x86::math::detail::GetActivationType(param.gate_activation);

#ifdef PADDLE_WITH_MKLML
    // use MKL packed to speedup GEMM
    if (paddle_num_threads >= 4) {
      auto blas = lite::x86::math::GetBlas<TARGET(kX86), T>(context);
      T* packed_gate = blas.GEMM_ALLOC(CblasBMatrix,
                                       1 /*height of C*/,
                                       frame_size * 2 /*width of weight*/,
                                       frame_size /*height of height*/);
      CHECK(packed_gate);
      blas.GEMM_PACK(CblasBMatrix,
                     CblasNoTrans,
                     1 /*cur bs?*/,
                     frame_size * 2,
                     frame_size,
                     T(1.0),
                     gru_value.gate_weight,
                     frame_size * 2,
                     packed_gate);
      T* packed_state = blas.GEMM_ALLOC(CblasBMatrix,
                                        1 /*height of C*/,
                                        frame_size /*width of weight*/,
                                        frame_size /*height of height*/);
      CHECK(packed_state);
      blas.GEMM_PACK(CblasBMatrix,
                     CblasNoTrans,
                     1 /*cur bs?*/,
                     frame_size,
                     frame_size,
                     T(1.0),
                     gru_value.state_weight,
                     frame_size,
                     packed_state);
      for (size_t n = 0; n < seq_len; n++) {
        int64_t bstart = static_cast<int64_t>(batch_starts[n]);
        int64_t bend = static_cast<int64_t>(batch_starts[n + 1]);
        int64_t cur_batch_size = bend - bstart;

        gru_value.output_value = batch_hidden_ptr + bstart * batch_hidden_width;
        gru_value.gate_value = batch_gate_ptr + bstart * batch_gate_width;
        gru_value.reset_output_value = batch_reset_hidden_prev_ptr +
                                       bstart * batch_reset_hidden_prev_width;

        if (gru_value.prev_out_value) {
          blas.GEMM_COMPUTE(CblasNoTrans,
                            CblasPacked,
                            cur_batch_size,
                            frame_size * 2,
                            frame_size,
                            gru_value.prev_out_value,
                            frame_size,
                            packed_gate,
                            frame_size * 2,
                            T(1),
                            gru_value.gate_value,
                            frame_size * 3);
        }

        lite::x86::math::detail::forward_final_output(
            lite::x86::math::detail::forward::gru_finalOutput<T>(),
            gru_value,
            frame_size,
            cur_batch_size,
            active_node,
            origin_mode);

        gru_value.prev_out_value = gru_value.output_value;
      }

      blas.GEMM_FREE(packed_gate);
      blas.GEMM_FREE(packed_state);
    } else {
#endif
      for (size_t n = 0; n < seq_len; n++) {
        int64_t bstart = static_cast<int64_t>(batch_starts[n]);
        int64_t bend = static_cast<int64_t>(batch_starts[n + 1]);
        int64_t cur_batch_size = bend - bstart;

        gru_value.output_value = batch_hidden_ptr + bstart * batch_hidden_width;
        gru_value.gate_value = batch_gate_ptr + bstart * batch_gate_width;
        gru_value.reset_output_value = batch_reset_hidden_prev_ptr +
                                       bstart * batch_reset_hidden_prev_width;

        lite::x86::math::GRUUnitFunctor<TARGET(kX86), T>::compute(
            context,
            gru_value,
            frame_size,
            cur_batch_size,
            active_node,
            active_gate,
            origin_mode);

        gru_value.prev_out_value = gru_value.output_value;
      }
#ifdef PADDLE_WITH_MKLML
    }
#endif
    lite::x86::math::Batch2LoDTensorFunctor<TARGET(kX86), T> to_seq;
    batch_hidden->set_lod(batch_gate->lod());
    to_seq(context, *batch_hidden, hidden);
  }
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
