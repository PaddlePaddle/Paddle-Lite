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

#ifdef CONV_OP

#pragma once

#include <vector>
#include "framework/operator.h"
#include "operators/math/im2col.h"
#include "operators/math/math_function.h"
#include "operators/math/vol2col.h"
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

using framework::OpKernelBase;

template <typename DeviceType, typename T>
class ConvKernel : public OpKernelBase<DeviceType, ConvParam> {
 public:
  void Compute(const ConvParam &param) const;
};

inline bool IsExpand(const std::vector<int64_t> &filter_dim,
                     const std::vector<int> &strides,
                     const std::vector<int> &paddings,
                     const std::vector<int> &dilations) {
  bool filter_1 = true, strides_1 = true, padding_0 = true, dilation_1 = true;
  for (size_t j = 0; j < strides.size(); ++j) {
    filter_1 = filter_1 && (static_cast<int>(filter_dim[j + 2]) == 1);
    strides_1 = strides_1 && (strides[j] == 1);
    padding_0 = padding_0 && (paddings[j] == 0);
    dilation_1 = dilation_1 && (dilations[j] == 1);
  }

  return !(filter_1 && strides_1 && padding_0 && dilation_1);
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
