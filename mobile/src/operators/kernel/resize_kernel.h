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

#ifdef RESIZE_OP

#pragma once

#include <vector>
#include "framework/operator.h"

#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

template <typename DeviceType>
inline framework::DDim CalOutputShape(const ResizeParam<DeviceType> &param) {
  const auto *input_x = param.InputX();
  const auto &input_x_dims = input_x->dims();
  auto *out = param.Out();
  framework::DDim out_dims = out->dims();
  const auto *input_shape = param.InputShape();

  if (input_shape) {
    input_x->dims()[0];
    auto *shape_data = input_shape->template data<int>();
    framework::Tensor cpu_shape_tensor;
    auto shape =
        std::vector<int>(shape_data, shape_data + input_shape->numel());
    const int in_batch_size = input_x->dims()[0];
    const int in_chan_size = input_x->dims()[1];
    const int in_height = input_x->dims()[2];
    const int in_width = input_x->dims()[3];

    int out_height = 0;
    int out_width = 0;
    bool is_pyramid_test = param.IsPyramidTest();
    if (is_pyramid_test == false) {
      out_height = param.Height();
      out_width = param.Width();
      PADDLE_MOBILE_ENFORCE(out_height > 0, "output height is required");
      PADDLE_MOBILE_ENFORCE(out_width > 0, "output width is required");

    } else {
      float out_height_scale = param.OutHeightScale();
      float out_width_scale = param.OutWidthScale();
      PADDLE_MOBILE_ENFORCE(out_height_scale > 0,
                            "output height scale is required");
      PADDLE_MOBILE_ENFORCE(out_width_scale > 0,
                            "output width scale is required");

      out_height = int(out_height_scale * in_height);
      out_width = int(out_width_scale * in_width);
    }

    out_dims = framework::make_ddim(
        {in_batch_size, in_chan_size, in_height, in_width});
  }
  return out_dims;
}

template <typename DeviceType, typename T>
class ResizeKernel
    : public framework::OpKernelBase<DeviceType, ResizeParam<DeviceType>> {
 public:
  void Compute(const ResizeParam<DeviceType> &param);
};
}  // namespace operators
}  // namespace paddle_mobile

#endif
