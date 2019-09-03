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

#include "lite/backends/arm/math/sequence_expand.h"
#include <string.h>
#include <vector>
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

template <>
void SequenceExpandImpl<float>(const float* x_data,
                               const LoD& x_lod,
                               int width,
                               const std::vector<uint64_t>& ref_lod,
                               lite::Tensor* output) {
  float* output_data = output->mutable_data<float>();
  if (x_lod.size() == 0) {
    for (int i = 0; i < ref_lod.size() - 1; i++) {
      for (int j = ref_lod[i]; j < ref_lod[i + 1]; j++) {
        memcpy(
            output_data + j * width, x_data + i * width, sizeof(float) * width);
      }
    }
    (output->mutable_lod())->push_back(ref_lod);
  } else {
    std::vector<uint64_t> out_lod;
    out_lod.push_back(0);
    uint64_t out_offset = 0;
    uint64_t len = 0;
    for (int i = 0; i < ref_lod.size() - 1; i++) {
      auto x_seq_len = x_lod[0][i + 1] - x_lod[0][i];
      for (int j = ref_lod[i]; j < ref_lod[i + 1]; j++) {
        memcpy(output_data + out_offset * width,
               x_data + len * width,
               width * sizeof(float) * x_seq_len);
        out_offset += x_seq_len;
        out_lod.push_back(out_offset);
      }
      len += x_seq_len;
    }
    (output->mutable_lod())->push_back(out_lod);
  }
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
