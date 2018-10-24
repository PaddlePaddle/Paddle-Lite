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

#ifdef POLYGONBOXTRANSFORM_OP
#pragma once

#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

template <typename P>
void PolygonBoxTransformCompute(const PolygonBoxTransformParam<CPU>& param) {
  const auto* input = param.Input();
  const auto& input_dims = input->dims();
  const auto* input_data = input->data<float>();
  auto* output = param.Output();
  auto* output_data = output->mutable_data<float>(input_dims);

  int64_t batch_size = input_dims[0];
  int64_t geo_channel = input_dims[1];
  int64_t height = input_dims[2];
  int64_t width = input_dims[3];
  int64_t id = 0;
  for (int64_t id_n = 0; id_n < batch_size * geo_channel; ++id_n) {
    for (int64_t id_h = 0; id_h < height; ++id_h) {
      for (int64_t id_w = 0; id_w < width; ++id_w) {
        id = id_n * height * width + width * id_h + id_w;
        if (id_n % 2 == 0) {
          output_data[id] = id_w * 4 - input_data[id];
        } else {
          output_data[id] = id_h * 4 - input_data[id];
        }
      }
    }
  }
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
