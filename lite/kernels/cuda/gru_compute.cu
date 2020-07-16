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
#include <string>

#include "lite/backends/cuda/cuda_utils.h"
#include "lite/backends/cuda/math/bias.h"
#include "lite/backends/cuda/math/gru_forward.h"
#include "lite/backends/cuda/math/sequence2batch.h"
#include "lite/backends/cuda/target_wrapper.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/cuda/gru_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

template <typename T>
struct GRUMetaValue {
  T* gate_weight;
  T* state_weight;
  T* gate_value;
  T* reset_output_value;
  T* output_value;
  T* prev_out_value;
};

template <typename T>
struct GRUUnitFunctor {
  static void compute(GRUMetaValue<T> value,
                      int frame_size,
                      int batch_size,
                      const lite::cuda::math::ActivationType& active_node,
                      const lite::cuda::math::ActivationType& active_gate,
                      bool origin_mode,
                      lite::cuda::math::Gemm<T, T>* blas,
                      CUDAContext* context) {
    dim3 threads, grids;
    if (batch_size == 1) {
      int frame_per_block = frame_size <= 1024 ? frame_size : 1024;
      int frame_blocks = (frame_size + 1024 - 1) / 1024;
      threads = dim3(frame_per_block, 1);
      grids = dim3(frame_blocks, 1);
    } else {
      threads = dim3(32, 32);
      grids = dim3((frame_size + 32 - 1) / 32, (batch_size + 32 - 1) / 32);
    }

    if (value.prev_out_value) {
      CHECK(blas->init(false,
                       false,
                       batch_size,
                       frame_size * 2,
                       frame_size,
                       frame_size,
                       frame_size * 2,
                       frame_size * 3,
                       context));
      blas->run(1.0f,
                1.0f,
                value.prev_out_value,
                value.gate_weight,
                value.gate_value,
                context);
    }
    CUDA_POST_KERNEL_CHECK;

    lite::cuda::math::GruForwardResetOutput<
        T><<<grids, threads, 0, context->exec_stream()>>>(
        value.gate_value,
        value.reset_output_value,
        value.prev_out_value,
        frame_size,
        batch_size,
        active_gate,
        batch_size == 1);
    CUDA_POST_KERNEL_CHECK;

    if (value.prev_out_value) {
      CHECK(blas->init(false,
                       false,
                       batch_size,
                       frame_size,
                       frame_size,
                       frame_size,
                       frame_size,
                       frame_size * 3,
                       context));
      blas->run(1.0f,
                1.0f,
                value.reset_output_value,
                value.state_weight,
                value.gate_value + frame_size * 2,
                context);
    }
    CUDA_POST_KERNEL_CHECK;

    lite::cuda::math::GruForwardFinalOutput<
        T><<<grids, threads, 0, context->exec_stream()>>>(value.gate_value,
                                                          value.prev_out_value,
                                                          value.output_value,
                                                          frame_size,
                                                          batch_size,
                                                          active_node,
                                                          origin_mode,
                                                          batch_size == 1);
    CUDA_POST_KERNEL_CHECK;
  }
};

template struct GRUUnitFunctor<float>;

template <typename T, PrecisionType PType>
void GRUCompute<T, PType>::PrepareForRun() {
  gemm_impl_.reset(new lite::cuda::math::Gemm<T, T>);
}

template <typename T, PrecisionType PType>
void GRUCompute<T, PType>::Run() {
  auto& context = this->ctx_->template As<CUDAContext>();
  auto stream = context.exec_stream();
  auto& param = this->template Param<param_t>();

  auto* input = param.input;
  lite::Tensor* h0{nullptr};
  if (param.h0) {
    h0 = const_cast<lite::Tensor*>(param.h0);
  }
  lite::Tensor* bias{nullptr};
  if (param.bias) {
    bias = const_cast<lite::Tensor*>(param.bias);
  }
  auto* weight = param.weight;
  auto* weight_data = const_cast<T*>(weight->template data<T>());
  auto* batch_gate = param.batch_gate;
  auto* batch_reset_hidden_prev = param.batch_reset_hidden_prev;
  auto* batch_hidden = param.batch_hidden;
  auto* hidden = param.hidden;
  auto* batch_reset_hidden_prev_data =
      batch_reset_hidden_prev->template mutable_data<T>(TARGET(kCUDA));
  hidden->template mutable_data<T>(TARGET(kCUDA));
  auto* batch_gate_data = batch_gate->template mutable_data<T>(TARGET(kCUDA));
  auto* batch_hidden_data =
      batch_hidden->template mutable_data<T>(TARGET(kCUDA));
  bool is_reverse = param.is_reverse;
  auto active_node = lite::cuda::math::GetActiveType(param.activation);
  auto active_gate = lite::cuda::math::GetActiveType(param.gate_activation);
  bool origin_mode = param.origin_mode;

  auto hidden_dims = hidden->dims();
  int frame_size = hidden_dims[1];

  lite::cuda::math::LoDTensor2BatchFunctor<T> batch_func;
  batch_func(*input, batch_gate, is_reverse, stream);

  if (bias) {
    lite::cuda::math::RowwiseAdd<T> add_bias;
    add_bias(batch_gate_data,
             bias->template data<T>(),
             batch_gate_data,
             frame_size,
             batch_gate->numel(),
             stream);
  }
  GRUMetaValue<T> gru_value;
  gru_value.gate_weight = weight_data;
  gru_value.state_weight = weight_data + 2 * frame_size * frame_size;

  if (h0) {
    // Since the batch computing for GRU reorders the input sequences
    // according to their length. The initialized cell state also needs
    // to reorder.
    ordered_h0_.Resize(h0->dims());
    lite::cuda::math::CopyMatrixRowsFunctor<T> row_shuffle;
    row_shuffle(*h0, &ordered_h0_, batch_gate->lod()[2], true, stream);
    gru_value.prev_out_value = ordered_h0_.mutable_data<T>(TARGET(kCUDA));
  } else {
    gru_value.prev_out_value = nullptr;
  }
  auto batch_starts = batch_gate->lod()[0];
  size_t num_batch = batch_starts.size() - 1;
  for (size_t n = 0; n < num_batch; ++n) {
    int bstart = static_cast<int>(batch_starts[n]);
    int bend = static_cast<int>(batch_starts[n + 1]);
    int cur_batch_size = bend - bstart;

    gru_value.output_value = batch_hidden_data + bstart * frame_size;
    gru_value.gate_value = batch_gate_data + bstart * frame_size * 3;
    gru_value.reset_output_value =
        batch_reset_hidden_prev_data + bstart * frame_size;

    GRUUnitFunctor<T>::compute(gru_value,
                               frame_size,
                               cur_batch_size,
                               active_node,
                               active_gate,
                               origin_mode,
                               gemm_impl_.get(),
                               &context);
    gru_value.prev_out_value = gru_value.output_value;
  }

  lite::cuda::math::Batch2LoDTensorFunctor<T> to_seq;
  batch_hidden->set_lod(batch_gate->lod());
  to_seq(*batch_hidden, hidden, stream);
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using GRUFp32 =
    paddle::lite::kernels::cuda::GRUCompute<float, PRECISION(kFloat)>;

REGISTER_LITE_KERNEL(gru, kCUDA, kFloat, kNCHW, GRUFp32, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindInput("H0", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindInput("Weight", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("BatchGate", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("BatchResetHiddenPrev", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("BatchHidden", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("Hidden", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .Finalize();
