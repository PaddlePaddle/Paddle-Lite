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

#ifdef GRU_UNIT_OP

#pragma once

#include <operators/math/gru_compute.h>
#include "operators/kernel/activation_kernel.h"
#include "operators/math/gemm.h"
#include "operators/math/math_function.h"
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

template <typename P>
void GruUnitCompute(const GruUnitParam<CPU>& param) {
  // inputs
  auto* input = param.InputInput();
  auto* hidden_prev = param.InputHiddenPrev();
  auto* weight = param.InputWeight();
  auto* bias = param.InputBias();
  // outputs
  auto* gate = param.OutGate();
  gate->mutable_data<P>();
  auto* reset_hidden_prev = param.OutResetHiddenPrev();
  reset_hidden_prev->mutable_data<P>();
  auto* hidden = param.OutHidden();
  hidden->mutable_data<P>();

  // add bias
  if (bias) {
    math::RowwiseAdd<CPU, float> add_bias;
    add_bias(*input, *bias, gate);
  }

  int batch_size = input->dims()[0];
  int frame_size = hidden_prev->dims()[1];
  const P* weight_data = weight->data<P>();

  math::GRUMetaValue<P> gru_value;
  gru_value.gate_weight = const_cast<P*>(weight_data);
  gru_value.state_weight =
      const_cast<P*>(weight_data + 2 * frame_size * frame_size);
  gru_value.prev_out_value = const_cast<P*>(hidden_prev->data<P>());

  gru_value.output_value = hidden->data<P>();
  gru_value.gate_value = gate->data<P>();
  gru_value.reset_output_value = reset_hidden_prev->data<P>();

  auto active_node = math::GetActivationType(param.Activation());
  auto active_gate = math::GetActivationType(param.GateActivation());
  math::GRUUnitFunctor<CPU, float>::compute(gru_value, frame_size, batch_size,
                                            active_node, active_gate);
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
