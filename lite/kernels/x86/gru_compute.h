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
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/core/types.h"
#include "lite/fluid/eigen.h"
#include "lite/x86/math/blas.h"
#include "lite/x86/math/gru_compute.h"
#include "lite/x86/math/math_function.h"
#include "lite/x86/math/sequence2batch.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

using Tensor = lite::Tensor;

template <typename T>
inline void ReorderInitState(const Tensor& src,
                             const std::vector<uint64_t>& index_lod,
                             Tensor* dst,
                             bool indexed_src) {
  lite::x86::math::CopyMatrixRowsFunctor<T> row_shuffle;
  dst->Resize(src.dims());
  dst->mutable_data<T>();
  row_shuffle(src, index_lod, dst, indexed_src);
}

template <typename T>
class GRUCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  void Run() override {
    i
    auto& context = ctx_->As<X86Context>();
    auto& param = *param_.get_mutable<operators::MulParam>();
    CHECK(context.x86_device_context());

    bool origin_mode = param.origin_mode;
    bool is_reverse = param.is_reverse;

    auto* input = &param.x->raw_tensor();
    auto* h0 = &param.h0->raw_tensor();
    auto* weight = &param.weight->raw_tensor();
    const T* weight_data = wight->data<T>();
    auto* bias = &param.bias->raw_tensor();

    auto* batch_gate = &param.batch_gate->raw_tensor();
    batch_gate->mutable_data<T>();
    auto* batch_reset_hidden_prev = &param.batch_reset_hidden_prev->raw_tensor();
    batch_reset_hidden_prev->mutable_data<T>();
    auto* batch_hidden = &param.batch_hidden->raw_tensor();
    batch_hidden->mutable_data<T>();
    auto* hidden = &param.hidden->raw_tensor();
    hidden->mutable_data<T>();

    auto hidden_dims = hidden->dims();

    lite::x86::math::LoDTensor2BatchFunctor<TARGET(kX86), T> to_batch;
    auto& dev_ctx = *context.x86_device_context();
    to_batch(dev_ctx, *input, batch_gate, true, is_reverse);

    if (bias) {
      lite::x86::math::RowwiseAdd<T> add_bias;
      add_bias(dev_ctx, *batch_gate, *bias, batch_gate);
    }

    int frame_size = hidden_dims[1];
    lite::x86::math::GRUMetaValue<T> gru_value;
    gru_value.gate_weight = const_cast<T*>(weight_data);
    gru_value.state_weight = const_cast<T*>(weight_data + 2 * frame_size * frame_size);
    Tensor ordered_h0;

    std::vector<size_t> order(batch_gate->lod()[2]);

    if (h0) {
      // Since the batch computing for GRU reorders the input sequences
      // according to their length. The initialized cell state also needs
      // to reorder.
      ReorderInitState<DeviceContext, T>(
          *context.x86_device_context(), *h0, order, &ordered_h0, true);
      gru_value.prev_out_value = ordered_h0.data<T>();
    } else {
      gru_value.prev_out_value = nullptr;
    }
    auto batch_starts = batch_gate->lod()[0];
    size_t seq_len = batch_starts.size() - 1;
    auto active_node = lite::x86::math::detail::GetActivationType(param.activation);
    auto active_gate = lite::x86::math::detail::GetActivationType(param.gate_activation);

#ifdef PADDLE_WITH_MKLML
    // use MKL packed to speedup GEMM
    if (FLAGS_paddle_num_threads >= 4) {
      auto blas = lite::x86::math::GetBlas<TARGET(kX86), T>(dev_ctx);
      T* packed_gate = blas.GEMM_ALLOC(CblasBMatrix, 1 /*height of C*/,
                                       frame_size * 2 /*width of weight*/,
                                       frame_size /*height of height*/);
      CHECK_OR_FALSE(packed_gate);
      blas.GEMM_PACK(CblasBMatrix, CblasNoTrans, 1 /*cur bs?*/, frame_size * 2,
                     frame_size, T(1.0), gru_value.gate_weight, frame_size * 2,
                     packed_gate);
      T* packed_state = blas.GEMM_ALLOC(CblasBMatrix, 1 /*height of C*/,
                                        frame_size /*width of weight*/,
                                        frame_size /*height of height*/);
      CHECK_OR_FALSE(packed_state);
      blas.GEMM_PACK(CblasBMatrix, CblasNoTrans, 1 /*cur bs?*/, frame_size,
                     frame_size, T(1.0), gru_value.state_weight, frame_size,
                     packed_state);
      for (size_t n = 0; n < seq_len; n++) {
        int bstart = static_cast<int>(batch_starts[n]);
        int bend = static_cast<int>(batch_starts[n + 1]);
        int cur_batch_size = bend - bstart;

        Tensor gate_t = batch_gate->Slice(bstart, bend);
        Tensor reset_hidden_prev_t =
            batch_reset_hidden_prev->Slice(bstart, bend);
        Tensor hidden_t = batch_hidden->Slice(bstart, bend);
        gru_value.output_value = hidden_t.data<T>();
        gru_value.gate_value = gate_t.data<T>();
        gru_value.reset_output_value = reset_hidden_prev_t.data<T>();

        if (gru_value.prev_out_value) {
          blas.GEMM_COMPUTE(
              CblasNoTrans, CblasPacked, cur_batch_size, frame_size * 2,
              frame_size, gru_value.prev_out_value, frame_size, packed_gate,
              frame_size * 2, T(1), gru_value.gate_value, frame_size * 3);
        }

        lite::x86::math::detail::forward_final_output(
            lite::x86::math::detail::forward::gru_finalOutput<T>(), gru_value, frame_size,
            cur_batch_size, active_node, origin_mode);

        gru_value.prev_out_value = gru_value.output_value;
      }

      blas.GEMM_FREE(packed_gate);
      blas.GEMM_FREE(packed_state);
    } else {
#endif
      for (size_t n = 0; n < seq_len; n++) {
        int bstart = static_cast<int>(batch_starts[n]);
        int bend = static_cast<int>(batch_starts[n + 1]);
        int cur_batch_size = bend - bstart;

        Tensor gate_t = batch_gate->Slice(bstart, bend);
        Tensor reset_hidden_prev_t =
            batch_reset_hidden_prev->Slice(bstart, bend);
        Tensor hidden_t = batch_hidden->Slice(bstart, bend);
        gru_value.output_value = hidden_t.data<T>();
        gru_value.gate_value = gate_t.data<T>();
        gru_value.reset_output_value = reset_hidden_prev_t.data<T>();

        lite::x86::math::GRUUnitFunctor<T>::compute(
            dev_ctx, gru_value, frame_size, cur_batch_size, active_node,
            active_gate, origin_mode);

        gru_value.prev_out_value = gru_value.output_value;
      }
#ifdef PADDLE_WITH_MKLML
    }
#endif
    math::Batch2LoDTensorFunctor<TARGET(kX86), T> to_seq;
    batch_hidden->set_lod(batch_gate->lod());
    to_seq(dev_ctx, *batch_hidden, hidden);

  }
};

template <typename T>
class GRUGradCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  void Run() override {
    auto& context = ctx_->As<X86Context>();
    auto& param = *param_.get_mutable<operators::MulParam>();
    CHECK(context.x86_device_context());

    bool origin_mode = parram.origin_mode;
    bool is_reverse = param.is_reverse;

    auto* h0 = &param.h0->raw_tensor();
    auto* weight = &param.weight->raw_tensor();
    const T* weight_data = wight->data<T>();

    auto* batch_gate = &param.batch_gate->raw_tensor();
    auto* batch_reset_hidden_prev = &param.batch_reset_hidden_prev->raw_tensor();
    auto* batch_hidden = &param.batch_hidden->raw_tensor();
    auto* hidden = &param.hidden->raw_tensor();
    auto* hidden_grad = &param.hidden_grad->raw_tensor();
    auto* input_grad = &param.input_grad->raw_tensor();
    auto* h0_grad = &param.h0_grad->raw_tensor();
    auto* weight_grad = &param.weight_grad->raw_tensor();
    auto* bias_grad = &param.bias_grad->raw_tensor();

    auto gate_dims = batch_gate->dims();
    auto hidden_dims = hidden->dims();
    int frame_size = hidden_dims[1];

    lite::x86::math::LoDTensor2BatchFunctor<TARGET(kX86), T> to_batch;
    Tensor batch_hidden_grad, batch_gate_grad, batch_reset_hidden_prev_grad;
    batch_hidden_grad.Resize(hidden_dims);
    batch_hidden_grad.mutable_data<T>();
    batch_gate_grad.Resize(gate_dims);
    batch_gate_grad.mutable_data<T>();
    batch_reset_hidden_prev_grad.Resize(hidden_dims);
    batch_reset_hidden_prev_grad.mutable_data<T>();
    lite::x86::math::SetConstant<TARGET(kX86), T> zero;
    auto& dev_ctx = *context.x86_device_context();
    zero(dev_ctx, &batch_hidden_grad, static_cast<T>(0.0));
    zero(dev_ctx, &batch_gate_grad, static_cast<T>(0.0));
    zero(dev_ctx, &batch_reset_hidden_prev_grad, static_cast<T>(0.0));

    Tensor ordered_h0, ordered_h0_grad;

    std::ector<size_t> order(batch_gate->lod()[2]);

    if (h0) {
      ReorderInitState<T>(*h0, order, &ordered_h0, true);
    }
    if (h0_grad) {
      ordered_h0_grad.Resize(h0_grad->dims());
      ordered_h0_grad.mutable_data<T>();
      zero(*context.x86_device_context(), &ordered_h0_grad, static_cast<T>(0.0));
    }

    batch_hidden_grad.set_lod(batch_hidden->lod());
    to_batch(dev_ctx, *hidden_grad, &batch_hidden_grad, false, is_reverse);

    lite::x86::math::GRUMetaValue<T> gru_value;
    gru_value.gate_weight = const_cast<T*>(weight_data);
    gru_value.state_weight =
        const_cast<T*>(weight_data + 2 * frame_size * frame_size);

    lite::x86::math::GRUMetaGrad<T> gru_grad;
    if (weight_grad) {
      gru_grad.gate_weight_grad =
          weight_grad->mutable_data<T>();
      zero(dev_ctx, weight_grad, static_cast<T>(0.0));
      gru_grad.state_weight_grad =
          weight_grad->data<T>() + 2 * frame_size * frame_size;
    } else {
      gru_grad.gate_weight_grad = nullptr;
      gru_grad.state_weight_grad = nullptr;
    }

    auto batch_starts = batch_hidden_grad.lod()[0];
    size_t num_batch = batch_starts.size() - 1;
    auto active_node = lite::x86:math::detail::GetActivationType(parram.activation);
    auto active_gate = lite::x86:math::detail::GetActivationType(parram.gate_activation);
    for (int n = static_cast<int>(num_batch) - 1; n >= 0; n--) {
      int bstart = static_cast<int>(batch_starts[n]);
      int bend = static_cast<int>(batch_starts[n + 1]);
      int cur_batch_size = bend - bstart;

      Tensor gate_t = batch_gate->Slice(bstart, bend);
      gru_value.gate_value = gate_t.data<T>();
      Tensor reset_hidden_prev_t = batch_reset_hidden_prev->Slice(bstart, bend);
      gru_value.reset_output_value = reset_hidden_prev_t.data<T>();

      Tensor hidden_grad_t = batch_hidden_grad.Slice(bstart, bend);
      gru_grad.output_grad = hidden_grad_t.data<T>();
      Tensor gate_grad_t = batch_gate_grad.Slice(bstart, bend);
      gru_grad.gate_grad = gate_grad_t.data<T>();
      Tensor reset_hidden_prev_grad_t =
          batch_reset_hidden_prev_grad.Slice(bstart, bend);
      gru_grad.reset_output_grad = reset_hidden_prev_grad_t.data<T>();
      if (n == 0) {
        gru_value.prev_out_value = h0 ? ordered_h0.data<T>() : nullptr;
        gru_grad.prev_out_grad =
            h0 && h0_grad ? ordered_h0_grad.data<T>() : nullptr;
      } else {
        int bstart_pre = static_cast<int>(batch_starts[n - 1]);
        Tensor hidden_prev_t = batch_hidden->Slice(bstart_pre, bstart);
        gru_value.prev_out_value = hidden_prev_t.data<T>();
        Tensor hidden_prev_grad_t = batch_hidden_grad.Slice(bstart_pre, bstart);
        gru_grad.prev_out_grad = hidden_prev_grad_t.data<T>();
      }
      gru_value.output_value = nullptr;
      lite::x86::math::GRUUnitGradFunctor<TARGET(kX86), T>::compute(
          dev_ctx, gru_value, gru_grad, frame_size, cur_batch_size, active_node,
          active_gate, origin_mode);
    }
    if (input_grad) {
      input_grad->mutable_data<T>();
      lite::x86::math::Batch2LoDTensorFunctor<TARGET(kX86), T> to_seq;
      batch_gate_grad.set_lod(batch_gate->lod());
      to_seq(dev_ctx, batch_gate_grad, input_grad);
    }
    if (bias_grad) {
      bias_grad->mutable_data<T>();
      lite::x86::math::ColwiseSum<TARGET(kX86), T> col_sum;
      col_sum(dev_ctx, batch_gate_grad, bias_grad);
    }
    if (h0 && h0_grad) {
      ReorderInitState<T>(ordered_h0_grad, order, h0_grad, false);
    }
  }
};

}  // x86
}  // kernels
}  // lite
}  // paddle
