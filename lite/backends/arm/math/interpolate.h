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
#include <string>
#include <vector>
#include "lite/core/parallel_defines.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

inline std::vector<int> get_new_shape(
    std::vector<const lite::Tensor*> list_new_shape_tensor) {
  // get tensor from
  std::vector<int> vec_new_shape;
  for (size_t i = 0; i < list_new_shape_tensor.size(); ++i) {
    auto tensor = list_new_shape_tensor[i];
    vec_new_shape.push_back(static_cast<int32_t>(*tensor->data<int32_t>()));
  }

  return vec_new_shape;
}

template <typename T>
inline std::vector<T> get_new_data_from_tensor(const Tensor* new_data_tensor) {
  std::vector<T> vec_new_data;
  auto* new_data = new_data_tensor->data<T>();
  lite::Tensor cpu_starts_tensor;
  vec_new_data =
      std::vector<T>(new_data, new_data + new_data_tensor->dims().production());
  return vec_new_data;
}

void bilinear_interp(const float* src,
                     int w_in,
                     int h_in,
                     float* dst,
                     int w_out,
                     int h_out,
                     float scale_x,
                     float scale_y,
                     bool with_align);
void nearest_interp(const float* src,
                    int w_in,
                    int h_in,
                    float* dst,
                    int w_out,
                    int h_out,
                    float scale_x,
                    float scale_y,
                    bool with_align);
template <typename T>
void nearest_interp_v2_compute(const T* src,
                               int w_in,
                               int h_in,
                               T* dst,
                               int w_out,
                               int h_out,
                               float scale_x,
                               float scale_y,
                               bool with_align) {
  if (with_align) {
    for (int h = 0; h < h_out; ++h) {
      T* dst_p = dst + h * w_out;
      int near_y = static_cast<int>(scale_y * h + 0.5);
      for (int w = 0; w < w_out; ++w) {
        int near_x = static_cast<int>(scale_x * w + 0.5);
        *dst_p++ = src[near_y * w_in + near_x];
      }
    }
  } else {
    for (int h = 0; h < h_out; ++h) {
      T* dst_p = dst + h * w_out;
      int near_y = static_cast<int>(scale_y * h);
      for (int w = 0; w < w_out; ++w) {
        int near_x = static_cast<int>(scale_x * w);
        *dst_p++ = src[near_y * w_in + near_x];
      }
    }
  }
}

void interpolate(lite::Tensor* X,
                 lite::Tensor* OutSize,
                 std::vector<const lite::Tensor*> SizeTensor,
                 lite::Tensor* Scale,
                 lite::Tensor* Out,
                 int out_height,
                 int out_width,
                 float scale,
                 bool with_align,
                 int align_mode,
                 std::string interpolate_type,
                 std::vector<float> scale_data);

template <typename T>
void nearest_interp_v2(lite::Tensor* X,
                       lite::Tensor* OutSize,
                       std::vector<const lite::Tensor*> SizeTensor,
                       lite::Tensor* Scale,
                       lite::Tensor* Out,
                       int out_height,
                       int out_width,
                       float scale,
                       bool with_align,
                       int align_mode,
                       std::string interpolate_type,
                       std::vector<float> scale_data) {
  int in_h = X->dims()[2];
  int in_w = X->dims()[3];
  float height_scale = -1;
  float width_scale = -1;

  if (SizeTensor.size() > 0) {
    // have size tensor
    auto new_size = get_new_shape(SizeTensor);
    out_height = new_size[0];
    out_width = new_size[1];
  } else {
    if (Scale != nullptr) {
      auto scale_data1 = get_new_data_from_tensor<float>(Scale);
      if (scale_data1.size() > 1) {
        height_scale = scale_data1[0];
        width_scale = scale_data1[1];
      } else {
        height_scale = scale_data1[0];
        width_scale = scale_data1[0];
      }
    } else {
      if (scale_data.size() > 1 && scale_data[0] > 0 && scale_data[1] > 0) {
        height_scale = scale_data[0];
        width_scale = scale_data[1];
      }
    }
    if (height_scale > 0. && width_scale > 0.) {
      out_height = static_cast<int>(in_h * height_scale);
      out_width = static_cast<int>(in_w * width_scale);
    }
    if (OutSize != nullptr) {
      auto out_size_data = get_new_data_from_tensor<int>(OutSize);
      out_height = out_size_data[0];
      out_width = out_size_data[1];
    }
  }

  float ratio_h = 0.f;
  float ratio_w = 0.f;
  if (out_height > 1) {
    float new_scale_h = 0.f;
    new_scale_h = (height_scale > 0) ? static_cast<float>(1. / height_scale)
                                     : static_cast<float>(in_h) / out_height;
    ratio_h = (with_align) ? static_cast<float>(in_h - 1) / (out_height - 1)
                           : static_cast<float>(new_scale_h);
  }
  if (out_width > 1) {
    float new_scale_w = 0.f;
    new_scale_w = (width_scale > 0) ? static_cast<float>(1. / width_scale)
                                    : static_cast<float>(in_w) / out_width;
    ratio_w = (with_align) ? static_cast<float>(in_w - 1) / (out_width - 1)
                           : static_cast<float>(new_scale_w);
  }

  int num_cout = X->dims()[0];
  int c_cout = X->dims()[1];
  Out->Resize({num_cout, c_cout, out_height, out_width});

  T* dout = Out->mutable_data<T>();
  const T* din = X->data<T>();
  int out_num = Out->dims()[0];
  int out_c = Out->dims()[1];
  int count = out_num * out_c;
  int out_h = Out->dims()[2];
  int out_w = Out->dims()[3];
  int spatial_in = in_h * in_w;
  int spatial_out = out_h * out_w;

  LITE_PARALLEL_BEGIN(i, tid, count) {
    nearest_interp_v2_compute<T>(din + spatial_in * i,
                                 in_w,
                                 in_h,
                                 dout + spatial_out * i,
                                 out_w,
                                 out_h,
                                 ratio_w,
                                 ratio_h,
                                 with_align);
  }
  LITE_PARALLEL_END()
}

} /* namespace math */
} /* namespace arm */
} /* namespace lite */
} /* namespace paddle */
