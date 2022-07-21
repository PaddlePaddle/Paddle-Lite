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
int CustomYoloBox3dNmsFuserImpl(TensorType& out_boxes,         // NOLINT
                                TensorType& out_nms_rois_num,  // NOLINT
                                TensorType& out_index,         // NOLINT
                                TensorType& out_locations,     // NOLINT
                                TensorType& out_dims,          // NOLINT
                                TensorType& out_alphas,        // NOLINT
                                const TensorType& input0,
                                const TensorType& input1,
                                const TensorType& input2,
                                const TensorType& imgsize,
                                const Tensor& anchors0,
                                const Tensor& anchors1,
                                const Tensor& anchors2,
                                const Tensor& class_num,
                                const Tensor& conf_thresh,
                                const Tensor& downsample_ratio0,
                                const Tensor& downsample_ratio1,
                                const Tensor& downsample_ratio2,
                                const Tensor& scale_x_y,
                                const Tensor& background_label,
                                const Tensor& score_threshold,
                                const Tensor& nms_top_k,
                                const Tensor& nms_threshold,
                                const Tensor& nms_eta,
                                const Tensor& keep_top_k,
                                const Tensor& normalized);
}  // namespace htp
}  // namespace qualcomm_qnn
}  // namespace nnadapter
