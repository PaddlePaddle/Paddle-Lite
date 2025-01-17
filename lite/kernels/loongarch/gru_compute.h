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
#include "lite/backends/loongarch/fluid/eigen.h"
#include "lite/backends/loongarch/math/blas.h"
#include "lite/backends/loongarch/math/gru_compute.h"
#include "lite/backends/loongarch/math/gru_cpu_kernel.h"
#include "lite/backends/loongarch/math/gru_kernel.h"
#include "lite/backends/loongarch/math/math_function.h"
#include "lite/backends/loongarch/math/sequence2batch.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/core/types.h"

// DECLARE_int32(paddle_num_threads);
extern int32_t paddle_num_threads;

namespace paddle {
namespace lite {
namespace kernels {
namespace loongarch {

using Tensor = lite::Tensor;

template <typename T>
inline void ReorderInitState(const lite::Context<TARGET(kLoongArch)>& context,
                             const Tensor& src,
                             const std::vector<uint64_t>& index_lod,
                             Tensor* dst,
                             bool indexed_src) {
  lite::loongarch::math::CopyMatrixRowsFunctor<TARGET(kLoongArch), T> row_shuffle;
  dst->Resize(src.dims());
  dst->template mutable_data<T>();
  row_shuffle(context, src, index_lod, dst, indexed_src);
}

static inline int64_t CalculateSeqWidth(const DDim& dims) {
  return dims.count(1, dims.size());
}

template <typename T>
class GRUCompute : public KernelLite<TARGET(kLoongArch), PRECISION(kFloat)> {
 public:
  void Run() override {
    auto& context = ctx_->As<LoongArchContext>();
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

    lite::loongarch::math::LoDTensor2BatchFunctor<TARGET(kLoongArch), T> to_batch;
    to_batch(context, *input, batch_gate, true, is_reverse);

    if (bias) {
      lite::loongarch::math::RowwiseAdd<TARGET(kLoongArch), T> add_bias;
      add_bias(context, *batch_gate, *bias, batch_gate);
    }

    int frame_size = hidden_dims[1];
    lite::loongarch::math::GRUMetaValue<T> gru_value;
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
        lite::loongarch::math::detail::GetActivationType(param.activation);
    auto active_gate =
        lite::loongarch::math::detail::GetActivationType(param.gate_activation);

      for (size_t n = 0; n < seq_len; n++) {
        int64_t bstart = static_cast<int64_t>(batch_starts[n]);
        int64_t bend = static_cast<int64_t>(batch_starts[n + 1]);
        int64_t cur_batch_size = bend - bstart;

        gru_value.output_value = batch_hidden_ptr + bstart * batch_hidden_width;
        gru_value.gate_value = batch_gate_ptr + bstart * batch_gate_width;
        gru_value.reset_output_value = batch_reset_hidden_prev_ptr +
                                       bstart * batch_reset_hidden_prev_width;

        lite::loongarch::math::GRUUnitFunctor<TARGET(kLoongArch), T>::compute(
            context,
            gru_value,
            frame_size,
            cur_batch_size,
            active_node,
            active_gate,
            origin_mode);

        gru_value.prev_out_value = gru_value.output_value;
      }

    lite::loongarch::math::Batch2LoDTensorFunctor<TARGET(kLoongArch), T> to_seq;
    batch_hidden->set_lod(batch_gate->lod());
    to_seq(context, *batch_hidden, hidden);
  }
};

}  // namespace loongarch
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
