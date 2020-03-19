// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include <Eigen/Core>
#include "lite/core/kernel.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"
#include "lite/operators/interpolate_op.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

inline void nearest_interp(const float* src,
                           int w_in,
                           int h_in,
                           float* dst,
                           int w_out,
                           int h_out,
                           bool with_align) {
  float scale_w_new = (with_align)
                          ? (static_cast<float>(w_in - 1) / (w_out - 1))
                          : (static_cast<float>(w_in) / (w_out));
  float scale_h_new = (with_align)
                          ? (static_cast<float>(h_in - 1) / (h_out - 1))
                          : (static_cast<float>(h_in) / (h_out));
  if (with_align) {
    for (int h = 0; h < h_out; ++h) {
      float* dst_p = dst + h * w_out;
      int near_y = static_cast<int>(scale_h_new * h + 0.5);
      for (int w = 0; w < w_out; ++w) {
        int near_x = static_cast<int>(scale_w_new * w + 0.5);
        *dst_p++ = src[near_y * w_in + near_x];
      }
    }
  } else {
    for (int h = 0; h < h_out; ++h) {
      float* dst_p = dst + h * w_out;
      int near_y = static_cast<int>(scale_h_new * h);
      for (int w = 0; w < w_out; ++w) {
        int near_x = static_cast<int>(scale_w_new * w);
        *dst_p++ = src[near_y * w_in + near_x];
      }
    }
  }
}

inline std::vector<int> get_new_shape(
    std::vector<const lite::Tensor*> list_new_shape_tensor) {
  std::vector<int> vec_new_shape;
  for (size_t i = 0; i < list_new_shape_tensor.size(); ++i) {
    auto tensor = list_new_shape_tensor[i];
    vec_new_shape.push_back(static_cast<int32_t>(*tensor->data<int32_t>()));
  }

  return vec_new_shape;
}

class InterpolateCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::InterpolateParam;

  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    int in_h = param.X->dims()[2];
    int in_w = param.X->dims()[3];
    if (param.SizeTensor.size() > 0) {
      auto new_size = get_new_shape(param.SizeTensor);
      param.out_h = new_size[0];
      param.out_w = new_size[1];
    } else {
      auto scale_tensor = param.Scale;
      if (scale_tensor != nullptr) {
        auto* scale_data = param.Scale->mutable_data<float>();
        param.scale = scale_data[0];
      }
      if (param.scale > 0) {
        param.out_h = static_cast<int>(in_h * param.scale);
        param.out_w = static_cast<int>(in_w * param.scale);
      }
      if (param.OutSize != nullptr) {
        auto* outsize_data = param.OutSize->mutable_data<float>();
        param.out_h = outsize_data[0];
        param.out_w = outsize_data[1];
      }
    }

    int num_cout = param.X->dims()[0];
    int c_cout = param.X->dims()[1];
    param.Out->Resize({num_cout, c_cout, param.out_h, param.out_w});

    float* dout = param.Out->mutable_data<float>();
    const float* din = param.X->data<float>();
    int out_num = param.Out->dims()[0];
    int out_c = param.Out->dims()[1];
    int count = out_num * out_c;
    int out_h = param.Out->dims()[2];
    int out_w = param.Out->dims()[3];
    int spatial_in = in_h * in_w;
    int spatial_out = out_h * out_w;

#pragma omp parallel for
    for (int i = 0; i < count; ++i) {
      nearest_interp(din + spatial_in * i,
                     in_w,
                     in_h,
                     dout + spatial_out * i,
                     out_w,
                     out_h,
                     param.align_corners);
    }
  }

  virtual ~InterpolateCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
