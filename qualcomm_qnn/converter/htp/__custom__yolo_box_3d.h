// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"

namespace nnadapter {
namespace qualcomm_qnn {
namespace htp {

// op execute function declarations
template <typename TensorType>
int CustomYoloBox3dImpl(TensorType& out_boxes,     // NOLINT
                        TensorType& out_scores,    // NOLINT
                        TensorType& out_location,  // NOLINT
                        TensorType& out_dim,       // NOLINT
                        TensorType& out_alpha,     // NOLINT
                        const TensorType& input,
                        const TensorType& img_size,
                        const Tensor& anchors,
                        const Tensor& class_num,
                        const Tensor& conf_thresh,
                        const Tensor& downsample_ratio,
                        const Tensor& scale_x_y);

template <typename T>
void CustomYoloBox3dKernel(T* out_boxes_data,
                           T* out_scores_data,
                           T* out_location_data,
                           T* out_dim_data,
                           T* out_alpha_data,
                           const T* input_data,
                           const int32_t* img_size_data,
                           const int32_t* anchors_data,
                           const int32_t class_num_data,
                           const float conf_thresh_data,
                           const int32_t downsample_ratio_data,
                           const float scale_x_y_data,
                           const int32_t batch_size,
                           const int32_t h,
                           const int32_t w,
                           const int32_t box_num,
                           const int32_t anchor_num);

}  // namespace htp
}  // namespace qualcomm_qnn
}  // namespace nnadapter
