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

#include "lite/backends/arm/math/box_coder.h"
#include "lite/backends/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

void box_coder(lite::Tensor* proposals,
               const lite::Tensor* anchors,
               const lite::Tensor* variances,
               const lite::Tensor* bbox_deltas,
               const std::string code_type,
               bool box_normalized,
               int axis) {
  if (code_type == "decode_center_size") {
    float normalized = !box_normalized ? 1.f : 0;

    const float* anchor_data = anchors->data<float>();
    const float* bbox_deltas_data = bbox_deltas->data<float>();
    float* proposals_data = proposals->mutable_data<float>();
    const float* variances_data = variances->data<float>();

    int N = bbox_deltas->dims()[0];
    int M = bbox_deltas->dims()[1];
    int len = bbox_deltas->dims()[2];

    for (int64_t row_id = 0; row_id < N; ++row_id) {
      for (int64_t col_id = 0; col_id < M; ++col_id) {
        size_t offset = row_id * M * len + col_id * len;
        int prior_box_offset = axis == 0 ? col_id * len : row_id * len;
        int var_offset = axis == 0 ? col_id * len : row_id * len;

        auto anchor_data_tmp = anchor_data + prior_box_offset;
        auto bbox_deltas_data_tmp = bbox_deltas_data + offset;
        auto proposals_data_tmp = proposals_data + offset;

        auto anchor_width =
            anchor_data_tmp[2] - anchor_data_tmp[0] + normalized;
        auto anchor_height =
            anchor_data_tmp[3] - anchor_data_tmp[1] + normalized;
        auto anchor_center_x = anchor_data_tmp[0] + 0.5 * anchor_width;
        auto anchor_center_y = anchor_data_tmp[1] + 0.5 * anchor_height;

        float bbox_center_x = 0, bbox_center_y = 0;
        float bbox_width = 0, bbox_height = 0;

        auto variances_data_tmp = variances_data + var_offset;

        bbox_center_x =
            variances_data_tmp[0] * bbox_deltas_data_tmp[0] * anchor_width +
            anchor_center_x;
        bbox_center_y =
            variances_data_tmp[1] * bbox_deltas_data_tmp[1] * anchor_height +
            anchor_center_y;
        bbox_width = std::exp(variances_data_tmp[2] * bbox_deltas_data_tmp[2]) *
                     anchor_width;
        bbox_height =
            std::exp(variances_data_tmp[3] * bbox_deltas_data_tmp[3]) *
            anchor_height;

        proposals_data_tmp[0] = bbox_center_x - bbox_width / 2;
        proposals_data_tmp[1] = bbox_center_y - bbox_height / 2;
        proposals_data_tmp[2] = bbox_center_x + bbox_width / 2 - normalized;
        proposals_data_tmp[3] = bbox_center_y + bbox_height / 2 - normalized;
      }
    }
  } else if (code_type == "encode_center_size") {
    LOG(FATAL) << "not implemented type: " << code_type;
  } else {
    LOG(FATAL) << "not supported type: " << code_type;
  }
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
