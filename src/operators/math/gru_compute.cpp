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

#include "operators/math/gru_compute.h"
#include "common/types.h"
#include "operators/math/activation.h"
#include "operators/math/gemm/cblas.h"
#include "operators/math/gru_cpu_kernel.h"

namespace paddle_mobile {
namespace operators {
namespace math {

template <typename T>
struct GRUUnitFunctor<CPU, T> {
  static void compute(GRUMetaValue<T> value, int frame_size, int batch_size,
                      const ActivationType active_node,
                      const ActivationType active_gate) {
    if (value.prev_out_value) {
      cblas_sgemm(false, false, batch_size, frame_size * 2, frame_size, 1.f,
                  value.prev_out_value, frame_size, value.gate_weight,
                  frame_size * 2, 1.f, value.gate_value, frame_size * 3);
    }

    forward_reset_output(value, frame_size, batch_size, active_gate);

    if (value.prev_out_value) {
      cblas_sgemm(false, false, batch_size, frame_size, frame_size, 1.f,
                  value.reset_output_value, frame_size, value.state_weight,
                  frame_size, 1.f, value.gate_value + frame_size * 2,
                  frame_size * 3);
    }

    forward_final_output(value, frame_size, batch_size, active_node);
  }
};

template struct GRUUnitFunctor<CPU, float>;

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile
#endif
