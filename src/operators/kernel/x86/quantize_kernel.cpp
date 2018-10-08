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

#ifdef PADDLE_MOBILE_X86

#include "operators/kernel/quantize_kernel.h"
#include <cmath>
#include <limits>

namespace paddle_mobile {
namespace operators {

static float find_abs_max(const Tensor *input) {
  float max_abs = float(0);
  const float *x = input->data<const float>();
  for (size_t i = 0; i < input->numel(); ++i) {
    float value = std::abs(x[i]);
    if (value > max_abs) {
      max_abs = value;
    }
  }
  return max_abs;
}

static void quantize_round_to_even(const Tensor *input,
                            const float scale,
                            Tensor *output) {
  const float *x = input->data<const float>();
  int8_t *y = output->data<int8_t>();
  for (size_t i = 0; i < input->numel(); ++i) {
    float value = x[i] * scale;
    long long quant = llround(value);
    if (abs(abs(round(value) - value) - 0.5) > 0) {
      y[i] = quant;
    } else {
      if (abs(quant) % 2 == 0) {
        y[i] = quant;
      } else {
        y[i] = quant + (quant > 0) ? -1 : 1;
      }
    }
  }
}

static void quantize_round_to_zero(const Tensor *input,
                            const float scale,
                            Tensor *output) {
  const float *x = input->data<const float>();
  int8_t *y = output->data<int8_t>();
  for (size_t i = 0; i < input->numel(); ++i) {
    y[i] = trunc(x[i] * scale);
  }
}

static void quantize_round_to_nearest(const Tensor *input,
                               const float scale,
                               Tensor *output) {
  const float *x = input->data<const float>();
  int8_t *y = output->data<int8_t>();
  for (size_t i = 0; i < input->numel(); ++i) {
    y[i] = round(x[i] * scale);
  }
}

template<>
bool QuantizeKernel<X86, float>::Init(QuantizeParam<X86> *param) {
  return true;
}

template<>
void QuantizeKernel<X86, float>::Compute(
    const QuantizeParam<X86> &param) const {
  // TODO
  float max_abs = 0.f;
  const Tensor *input = param.input_;
  Tensor *output = param.out_;
  Tensor *output_scale = param.online_scale_;
  if (param.is_static_) {
    max_abs = param.static_scale_;
  } else {
    max_abs = find_abs_max(input);
  }
  if (max_abs < std::numeric_limits<float>::min()) {
    max_abs = std::numeric_limits<float>::min();
  }
  // only support int8 currently
  float online_scale = 127 / max_abs;
  param.online_scale_->mutable_data<float>()[0] = online_scale;
  switch (param.round_type_) {
    case ROUND_NEAREST_TO_EVEN:
      quantize_round_to_even(input, online_scale, output);
      break;
    case ROUND_NEAREST_TOWARDS_ZERO:
      quantize_round_to_zero(input, online_scale, output);
      break;
    case ROUND_NEAREST_AWAY_ZERO:
      quantize_round_to_nearest(input, online_scale, output);
    default:
      LOG(kLOG_ERROR) << "round type is not supported.";
      break;
  }
}

}  // namespace paddle_mobile
}  // namespace operators

#endif
