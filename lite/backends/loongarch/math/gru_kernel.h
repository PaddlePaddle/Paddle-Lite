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

#pragma once
#include <type_traits>
#include "lite/backends/loongarch/math/activation_functions.h"
#include "lite/utils/macros.h"

namespace paddle {
namespace lite {
namespace loongarch {
namespace math {
namespace detail {

namespace forward {

template <typename T>
class gru_resetOutput {
 public:
  HOSTDEVICE void operator()(T *value_update_gate,
                             T *value_reset_gate,
                             T *prev_out,
                             T *value_reset_output,
                             ActivationType act_gate) {
    *value_update_gate = activation(*value_update_gate, act_gate);
    *value_reset_gate = activation(*value_reset_gate, act_gate);
    *value_reset_output = (*prev_out) * (*value_reset_gate);
  }
#ifndef __loongarch_asx
  static const bool lasx = false;
#else
  static const bool lasx = true;
  HOSTDEVICE void operator()(__m256 *value_update_gate,
                             __m256 *value_reset_gate,
                             __m256 *prev_out,
                             __m256 *value_reset_output,
                             ActivationType act_gate) {
    *value_update_gate = activation(*value_update_gate, act_gate);
    *value_reset_gate = activation(*value_reset_gate, act_gate);
    *value_reset_output = lasx_mul_f32(*prev_out, *value_reset_gate);
  }
#endif
};

template <typename T>
class gru_finalOutput {
 public:
  HOSTDEVICE void operator()(T *value_update_gate,
                             T *value_frame_state,
                             T *prev_out,
                             T *value_output,
                             ActivationType act_input,
                             bool origin_mode) {
    *value_frame_state = activation(*value_frame_state, act_input);
    if (origin_mode) {
      *value_output = ((*value_update_gate) * (*prev_out)) +
                      *value_frame_state -
                      ((*value_update_gate) * (*value_frame_state));
    } else {
      *value_output = *prev_out - ((*value_update_gate) * (*prev_out)) +
                      ((*value_update_gate) * (*value_frame_state));
    }
  }
#ifndef __loongarch_asx
  static const bool lasx = false;
#else
  static const bool lasx = true;
  HOSTDEVICE void operator()(__m256 *value_update_gate,
                             __m256 *value_frame_state,
                             __m256 *prev_out,
                             __m256 *value_output,
                             ActivationType act_input,
                             bool origin_mode) {
    *value_frame_state = activation(*value_frame_state, act_input);
    if (origin_mode) {
      *value_output = lasx_sub_f32(
          lasx_add_f32(lasx_mul_f32(*value_update_gate, *prev_out),
                        *value_frame_state),
          lasx_mul_f32(*value_update_gate, *value_frame_state));
    } else {
      *value_output = lasx_add_f32(
          lasx_sub_f32(*prev_out,
                        lasx_mul_f32(*value_update_gate, *prev_out)),
          lasx_mul_f32(*value_update_gate, *value_frame_state));
    }
  }
#endif
};
}  // namespace forward

namespace backward {

template <typename T>
class gru_stateGrad {
 public:
  HOSTDEVICE void operator()(T *value_update_gate,
                             T *grad_update_gate,
                             T *value_frame_state,
                             T *grad_frame_state,
                             T *value_prev_out,
                             T *grad_prev_out,
                             T *grad_output,
                             ActivationType act_input,
                             bool origin_mode) {
    if (origin_mode) {
      *grad_update_gate =
          (*grad_output) * ((*value_prev_out) - (*value_frame_state));
      *grad_prev_out += (*grad_output * (*value_update_gate));
      *grad_frame_state = activation(
          *grad_output * (static_cast<T>(1.0) - (*value_update_gate)),
          *value_frame_state,
          act_input);
    } else {
      *grad_update_gate =
          (*grad_output) * ((*value_frame_state) - (*value_prev_out));
      *grad_prev_out +=
          (*grad_output * (static_cast<T>(1.0) - *value_update_gate));
      *grad_frame_state = activation(
          *grad_output * (*value_update_gate), *value_frame_state, act_input);
    }
  }
#ifndef __loongarch_asx
  static const bool lasx = false;
#else
  static const bool lasx = true;
  HOSTDEVICE void operator()(__m256 *value_update_gate,
                             __m256 *grad_update_gate,
                             __m256 *value_frame_state,
                             __m256 *grad_frame_state,
                             __m256 *value_prev_out,
                             __m256 *grad_prev_out,
                             __m256 *grad_output,
                             ActivationType act_input,
                             bool origin_mode) {
    if (origin_mode) {
      *grad_update_gate = lasx_mul_f32(
          *grad_output, lasx_sub_f32(*value_prev_out, *value_frame_state));
      *grad_prev_out = lasx_add_f32(
          *grad_prev_out, lasx_mul_f32(*grad_output, *value_update_gate));
      *grad_frame_state = activation(
          lasx_mul_f32(
              *grad_output,
              lasx_sub_f32(lasx_set1_f32(1.0f), *value_update_gate)),
          *value_frame_state,
          act_input);
    } else {
      *grad_update_gate = lasx_mul_f32(
          *grad_output, lasx_sub_f32(*value_frame_state, *value_prev_out));
      *grad_prev_out = lasx_add_f32(
          *grad_prev_out,
          lasx_mul_f32(
              *grad_output,
              lasx_sub_f32(lasx_set1_f32(1.0f), *value_update_gate)));
      *grad_frame_state =
          activation(lasx_mul_f32(*grad_output, *value_update_gate),
                     *value_frame_state,
                     act_input);
    }
  }
#endif
};

template <typename T>
class gru_resetGrad {
 public:
  HOSTDEVICE void operator()(T *value_update_gate,
                             T *grad_update_gate,
                             T *value_reset_gate,
                             T *grad_reset_gate,
                             T *value_prev_out,
                             T *grad_prev_out,
                             T *grad_reset_output,
                             ActivationType act_gate) {
    *grad_reset_gate = (*grad_reset_output * (*value_prev_out));
    *grad_prev_out += (*grad_reset_output * (*value_reset_gate));
    *grad_update_gate =
        activation(*grad_update_gate, *value_update_gate, act_gate);
    *grad_reset_gate =
        activation(*grad_reset_gate, *value_reset_gate, act_gate);
  }
#ifndef __loongarch_asx
  static const bool lasx = false;
#else
  static const bool lasx = true;
  HOSTDEVICE void operator()(__m256 *value_update_gate,
                             __m256 *grad_update_gate,
                             __m256 *value_reset_gate,
                             __m256 *grad_reset_gate,
                             __m256 *value_prev_out,
                             __m256 *grad_prev_out,
                             __m256 *grad_reset_output,
                             ActivationType act_gate) {
    *grad_reset_gate = lasx_mul_f32(*grad_reset_output, *value_prev_out);
    *grad_prev_out = lasx_add_f32(
        *grad_prev_out, lasx_mul_f32(*grad_reset_output, *value_reset_gate));
    *grad_update_gate =
        activation(*grad_update_gate, *value_update_gate, act_gate);
    *grad_reset_gate =
        activation(*grad_reset_gate, *value_reset_gate, act_gate);
  }
#endif
};

}  // namespace backward

}  // namespace detail
}  // namespace math
}  // namespace loongarch
}  // namespace lite
}  // namespace paddle
