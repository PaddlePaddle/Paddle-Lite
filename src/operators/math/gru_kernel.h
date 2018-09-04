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
#ifdef GRU_OP
#pragma once
#include <type_traits>
#include "operators/math/activation_functions.h"

namespace paddle_mobile {
namespace operators {
namespace math {

namespace forward {

template <typename T>
class gru_resetOutput {
 public:
  void operator()(T *value_update_gate, T *value_reset_gate, T *prev_out,
                  T *value_reset_output, ActivationType act_gate) {
    *value_update_gate = activation(*value_update_gate, act_gate);
    *value_reset_gate = activation(*value_reset_gate, act_gate);
    *value_reset_output = (*prev_out) * (*value_reset_gate);
  }
};

template <typename T>
class gru_finalOutput {
 public:
  void operator()(T *value_update_gate, T *value_frame_state, T *prev_out,
                  T *value_output, ActivationType act_input) {
    *value_frame_state = activation(*value_frame_state, act_input);
    *value_output = *prev_out - ((*value_update_gate) * (*prev_out)) +
                    ((*value_update_gate) * (*value_frame_state));
  }
};
}  // namespace forward

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile
#endif
