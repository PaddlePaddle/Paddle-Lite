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
int MulticlassNmsImpl(TensorType& output_box,           // NOLINT
                      TensorType& output_nms_rois_num,  // NOLINT
                      TensorType& output_index,         // NOLINT
                      const TensorType& bboxes,
                      const TensorType& scores,
                      const Tensor& background_label,
                      const Tensor& score_threshold,
                      const Tensor& nms_top_k,
                      const Tensor& nms_threshold,
                      const Tensor& nms_eta,
                      const Tensor& keep_top_k,
                      const Tensor& normalized);

template <typename T>
void MulticlassNmsKernel(T* output_box_data,
                         int32_t* output_nms_rois_num_data,
                         int32_t* output_index_data,
                         const T* bboxes_data,
                         const T* scores_data,
                         const int32_t background_label_data,
                         const float score_threshold_data,
                         const int32_t nms_top_k_data,
                         const float nms_threshold_data,
                         const float nms_eta_data,
                         const int32_t keep_top_k_data,
                         const bool normalized_data,
                         const int32_t batch_size,
                         const int32_t class_num,
                         const int32_t box_num,
                         const int32_t box_size,
                         const std::vector<uint32_t>& batch_starts);

}  // namespace htp
}  // namespace qualcomm_qnn
}  // namespace nnadapter
