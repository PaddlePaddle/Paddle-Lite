/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef GRU_OP
#pragma once

#include <operators/math/sequence2batch.h>
#include <vector>
#include "common/types.h"
#include "operators/math/gru_compute.h"
#include "operators/math/math_function.h"
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

template <typename Device, typename T>
inline void ReorderInitState(const framework::Tensor& src,
                             std::vector<size_t> index_lod,
                             framework::Tensor* dst, bool indexed_src) {
  math::CopyMatrixRowsFunctor<Device, T> row_shuffle;
  dst->mutable_data<T>(src.dims());
  row_shuffle(src, index_lod, dst, indexed_src);
}

template <typename T>
void GruCompute(const GruParam<CPU>& param) {
  auto* input = param.InputInput();
  auto* h0 = param.InputH0();
  auto* weight = param.InputWeight();
  const auto* weight_data = weight->data<float>();
  auto* bias = param.InputBias();
  auto* batch_gate = param.OutBatchGate();
  batch_gate->mutable_data<float>();
  auto* batch_reset_hidden_prev = param.OutBatchResetHiddenPrev();
  batch_reset_hidden_prev->mutable_data<float>();
  auto* batch_hidden = param.OutBatchHidden();
  batch_hidden->mutable_data<float>();
  auto* hidden = param.OutHidden();
  hidden->mutable_data<float>();

  auto hidden_dims = hidden->dims();

  bool is_reverse = param.IsReverse();
  math::LoDTensor2BatchFunctor<CPU, float> to_batch;
  to_batch(*input, batch_gate, true, is_reverse);
  if (bias) {
    math::RowwiseAdd<CPU, float> add_bias;
    add_bias(*batch_gate, *bias, batch_gate);
  }
  int frame_size = hidden_dims[1];
  math::GRUMetaValue<float> gru_value;
  gru_value.gate_weight = const_cast<float*>(weight_data);
  gru_value.state_weight =
      const_cast<float*>(weight_data + 2 * frame_size * frame_size);
  framework::Tensor ordered_h0;
  std::vector<size_t> order(batch_gate->lod()[2]);
  if (h0) {
    // Since the batch computing for GRU reorders the input sequences
    // according to their length. The initialized cell state also needs
    // to reorder.
    ReorderInitState<CPU, float>(*h0, order, &ordered_h0, true);
    gru_value.prev_out_value = ordered_h0.data<float>();
  } else {
    gru_value.prev_out_value = nullptr;
  }
  auto batch_starts = batch_gate->lod()[0];
  size_t seq_len = batch_starts.size() - 1;
  auto active_node = math::GetActivationType(param.Activation());
  auto active_gate = math::GetActivationType(param.GateActivation());
  for (size_t n = 0; n < seq_len; n++) {
    int bstart = static_cast<int>(batch_starts[n]);
    int bend = static_cast<int>(batch_starts[n + 1]);
    int cur_batch_size = bend - bstart;
    framework::Tensor gate_t = batch_gate->Slice(bstart, bend);
    framework::Tensor reset_hidden_prev_t =
        batch_reset_hidden_prev->Slice(bstart, bend);
    framework::Tensor hidden_t = batch_hidden->Slice(bstart, bend);
    gru_value.output_value = hidden_t.data<float>();
    gru_value.gate_value = gate_t.data<float>();
    gru_value.reset_output_value = reset_hidden_prev_t.data<float>();

    math::GRUUnitFunctor<CPU, float>::compute(
        gru_value, frame_size, cur_batch_size, active_node, active_gate);

    gru_value.prev_out_value = gru_value.output_value;
  }
  math::Batch2LoDTensorFunctor<CPU, float> to_seq;
  batch_hidden->set_lod(batch_gate->lod());
  to_seq(*batch_hidden, hidden);
}

}  // namespace operators
}  // namespace paddle_mobile

#endif  // GRU_OP
