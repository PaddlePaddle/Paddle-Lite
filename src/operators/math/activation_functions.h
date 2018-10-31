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
#include <math.h>
#include <string>
#include "common/enforce.h"
namespace paddle_mobile {
namespace operators {
namespace math {

#define SIGMOID_THRESHOLD_MIN -40.0
#define SIGMOID_THRESHOLD_MAX 13.0
#define EXP_MAX_INPUT 40.0

enum ActivationType {
  kSigmoid,
  kReLU,
  kTanh,
  kIdentity,
};

inline ActivationType GetActivationType(const std::string &type) {
  if (type == "sigmoid") {
    return ActivationType::kSigmoid;
  } else if (type == "relu") {
    return ActivationType::kReLU;
  } else if (type == "tanh") {
    return ActivationType::kTanh;
  } else if (type == "identity" || type == "") {
    return ActivationType::kIdentity;
  }
  PADDLE_MOBILE_THROW_EXCEPTION("Not support activation type.");
}

namespace forward {

template <typename T>
T Identity(const T a) {
  return a;
}

template <typename T>
T Relu(const T a) {
  return a > static_cast<T>(0.0) ? a : static_cast<T>(0.0);
}

template <typename T>
T Sigmoid(const T a) {
  const T min = SIGMOID_THRESHOLD_MIN;
  const T max = SIGMOID_THRESHOLD_MAX;
  T tmp = (a < min) ? min : ((a > max) ? max : a);
  return static_cast<T>(1.0) / (static_cast<T>(1.0) + exp(-tmp));
}

template <typename T>
T Tanh(const T a) {
  T tmp = -2.0 * a;
  tmp = (tmp > EXP_MAX_INPUT) ? EXP_MAX_INPUT : tmp;
  return (2.0 / (1.0 + exp(tmp))) - 1.0;
}

}  // namespace forward

template <typename T>
struct Active {
  typedef T (*Act)(T);
};

static Active<float>::Act kActFloat[] = {
    &forward::Sigmoid<float>, &forward::Relu<float>, &forward::Tanh<float>,
    &forward::Identity<float>};

namespace forward {
inline float activation(float a, int index) { return kActFloat[index](a); }

}  // namespace forward

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile
