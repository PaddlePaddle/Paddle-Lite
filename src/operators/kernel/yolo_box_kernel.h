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

#pragma once

#include <algorithm>
#include <cmath>
#include <vector>
#include "framework/operator.h"
#include "operators/math/transform.h"
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

#ifdef YOLOBOX_OP
inline void ExpandAspectRatios(const std::vector<float> &input_aspect_ratior,
                               bool flip,
                               std::vector<float> *output_aspect_ratior) {
  constexpr float epsilon = 1e-6;
  output_aspect_ratior->clear();
  output_aspect_ratior->push_back(1.0f);
  for (size_t i = 0; i < input_aspect_ratior.size(); ++i) {
    float ar = input_aspect_ratior[i];
    bool already_exist = false;
    for (size_t j = 0; j < output_aspect_ratior->size(); ++j) {
      if (fabs(ar - output_aspect_ratior->at(j)) < epsilon) {
        already_exist = true;
        break;
      }
    }
    if (!already_exist) {
      output_aspect_ratior->push_back(ar);
      if (flip) {
        output_aspect_ratior->push_back(1.0f / ar);
      }
    }
  }
}

template <typename DeviceType, typename T>
class YoloBoxKernel
    : public framework::OpKernelBase<DeviceType, YoloBoxParam<DeviceType>> {
 public:
  void Compute(const YoloBoxParam<DeviceType> &param);
  bool Init(YoloBoxParam<DeviceType> *param);
};

#endif  // YOLOBOX_OP
}  // namespace operators
}  // namespace paddle_mobile
