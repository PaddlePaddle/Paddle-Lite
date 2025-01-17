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

#include "lite/backends/loongarch/math/gru_compute.h"
#include "lite/backends/loongarch/math/blas.h"
#include "lite/backends/loongarch/math/gru_cpu_kernel.h"
#include "lite/backends/loongarch/math/gru_kernel.h"

namespace paddle {
namespace lite {
namespace loongarch {
namespace math {

template <typename T>
struct GRUUnitFunctor<lite::TargetType::kLoongArch, T> {
  static void compute(const lite::LoongArchContext &context,
                      GRUMetaValue<T> value,
                      int frame_size,
                      int batch_size,
                      const detail::ActivationType active_node,
                      const detail::ActivationType active_gate,
                      bool origin_mode) {
    auto blas = math::GetBlas<lite::TargetType::kLoongArch, T>(context);
    if (value.prev_out_value) {
      blas.GEMM(false,
                false,
                batch_size,
                frame_size * 2,
                frame_size,
                1,
                value.prev_out_value,
                frame_size,
                value.gate_weight,
                frame_size * 2,
                1,
                value.gate_value,
                frame_size * 3);
    }

    detail::forward_reset_output(detail::forward::gru_resetOutput<T>(),
                                 value,
                                 frame_size,
                                 batch_size,
                                 active_gate);

    if (value.prev_out_value) {
      blas.GEMM(false,
                false,
                batch_size,
                frame_size,
                frame_size,
                1,
                value.reset_output_value,
                frame_size,
                value.state_weight,
                frame_size,
                1,
                value.gate_value + frame_size * 2,
                frame_size * 3);
    }

    detail::forward_final_output(detail::forward::gru_finalOutput<T>(),
                                 value,
                                 frame_size,
                                 batch_size,
                                 active_node,
                                 origin_mode);
  }
};

template <typename T>
struct GRUUnitGradFunctor<lite::TargetType::kLoongArch, T> {
  static void compute(const lite::LoongArchContext &context,
                      GRUMetaValue<T> value,
                      GRUMetaGrad<T> grad,
                      int frame_size,
                      int batch_size,
                      const detail::ActivationType active_node,
                      const detail::ActivationType active_gate,
                      bool origin_mode) {
    detail::backward_state_grad(detail::backward::gru_stateGrad<T>(),
                                value,
                                grad,
                                frame_size,
                                batch_size,
                                active_node,
                                origin_mode);
    auto blas = math::GetBlas<lite::TargetType::kLoongArch, T>(context);
    if (value.prev_out_value && grad.prev_out_grad) {
      blas.GEMM(false,
                true,
                batch_size,
                frame_size,
                frame_size,
                1,
                grad.gate_grad + frame_size * 2,
                frame_size * 3,
                value.state_weight,
                frame_size,
                0,
                grad.reset_output_grad,
                frame_size);

      if (grad.state_weight_grad) {
        blas.GEMM(true,
                  false,
                  frame_size,
                  frame_size,
                  batch_size,
                  1,
                  value.reset_output_value,
                  frame_size,
                  grad.gate_grad + frame_size * 2,
                  frame_size * 3,
                  1,
                  grad.state_weight_grad,
                  frame_size);
      }
    }

    detail::backward_reset_grad(detail::backward::gru_resetGrad<T>(),
                                value,
                                grad,
                                frame_size,
                                batch_size,
                                active_gate);
    if (grad.prev_out_grad && value.prev_out_value) {
      blas.GEMM(false,
                true,
                batch_size,
                frame_size,
                frame_size * 2,
                1,
                grad.gate_grad,
                frame_size * 3,
                value.gate_weight,
                frame_size * 2,
                1,
                grad.prev_out_grad,
                frame_size);

      if (grad.gate_weight_grad) {
        blas.GEMM(true,
                  false,
                  frame_size,
                  frame_size * 2,
                  batch_size,
                  1,
                  value.prev_out_value,
                  frame_size,
                  grad.gate_grad,
                  frame_size * 3,
                  1,
                  grad.gate_weight_grad,
                  frame_size * 2);
      }
    }
  }
};

template struct GRUUnitFunctor<lite::TargetType::kLoongArch, float>;
template struct GRUUnitFunctor<lite::TargetType::kLoongArch, double>;
template struct GRUUnitGradFunctor<lite::TargetType::kLoongArch, float>;
template struct GRUUnitGradFunctor<lite::TargetType::kLoongArch, double>;

}  // namespace math
}  // namespace loongarch
}  // namespace lite
}  // namespace paddle
