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

#ifdef ANCHOR_GENERATOR_OP

#include <string.h>
#include <iostream>
#include <utility>
#include <vector>
#include "operators/kernel/detection_kernel.h"

namespace paddle_mobile {
namespace operators {

template <>
bool AnchorGeneratorKernel<FPGA, float>::Init(
    AnchorGeneratorParam<FPGA> *param) {
  auto input = param->input_;
  auto anchors = param->output_anchors_;
  auto anchor_ptr = anchors->mutable_data<float>();
  auto stride = param->stride_;
  auto feature_width = input->dims()[3], feature_height = input->dims()[2];
  auto stride_width = stride[0], stride_height = stride[1];
  auto offset = param->offset_;

  int anchors_offset[] = {-2,  -2,   18,   18,  -10, -9,   26,   25,   -23,
                          -20, 39,   36,   -43, -34, 59,   49,   -63,  -54,
                          79,  69,   -96,  -77, 112, 93,   -137, -118, 153,
                          134, -204, -188, 220, 204, -281, -395, 296,  441};

  int anchors_offset2[] = {0, 0, 51, 77, 0, 0, 30, 35, 0, 0, 81, 103,
                           0, 0, 20, 21, 0, 0, 36, 44, 0, 0, 43, 58,
                           0, 0, 34, 68, 0, 0, 24, 28, 0, 0, 19, 46};

  if (offset > 0.6) {
    memcpy(anchors_offset, anchors_offset2, sizeof(anchors_offset));
    std::cout << "anchor generator marker" << std::endl;
  } else {
    std::cout << "anchor generator rfcn" << std::endl;
  }
  int num_anchors = sizeof(anchors_offset) / (sizeof(int) * 4);

  //  DLOG << "feature_height: " << feature_height;
  //  DLOG << "feature_width: " << feature_width;
  //  DLOG << "num_anchors: " << num_anchors;
  //  DLOG << "stride_width: " << stride_width;
  //  DLOG << "stride_height: " << stride_height;

  for (int h_idx = 0; h_idx < feature_height; ++h_idx) {
    int offset0 = h_idx * feature_width * num_anchors * 4;
    for (int w_idx = 0; w_idx < feature_width; ++w_idx) {
      int offset1 = w_idx * num_anchors * 4;
      for (int idx = 0; idx < num_anchors; idx++) {
        int offset = offset0 + offset1 + idx * 4;
        anchor_ptr[offset + 0] =
            anchors_offset[idx * 4 + 0] + w_idx * stride_width;
        anchor_ptr[offset + 1] =
            anchors_offset[idx * 4 + 1] + h_idx * stride_height;
        anchor_ptr[offset + 2] =
            anchors_offset[idx * 4 + 2] + w_idx * stride_width;
        anchor_ptr[offset + 3] =
            anchors_offset[idx * 4 + 3] + h_idx * stride_height;
      }
    }
  }
  return true;
}

template <>
void AnchorGeneratorKernel<FPGA, float>::Compute(
    const AnchorGeneratorParam<FPGA> &param) {}

}  // namespace operators
}  // namespace paddle_mobile

#endif  // ANCHOR_GENERATOR_OP
