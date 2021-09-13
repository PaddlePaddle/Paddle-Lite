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

#include "lite/backends/arm/math/slice.h"
#include <arm_neon.h>
#include <algorithm>
#include <cmath>
#include "lite/utils/log/cp_logging.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

template <typename Dtype>
void slice(const Dtype* input,
           std::vector<int64_t> in_dims,
           std::vector<int> axes,
           std::vector<int> starts,
           std::vector<int> ends,
           Dtype* out,
           Context<TARGET(kARM)>* ctx) {
  auto out_dims = in_dims;
  std::vector<int> real_starts(in_dims.size(), 0);
  std::vector<int> real_ends(in_dims.size(), 0);
  std::vector<int> real_step(in_dims.size(), 0);
  for (int i = 0; i < in_dims.size(); i++) {
    real_ends[i] = in_dims[i];
  }
  for (int i = 0; i < axes.size(); i++) {
    int dim_value = in_dims[axes[i]];
    if (dim_value > 0) {
      int start = starts[i] < 0 ? (starts[i] + dim_value) : starts[i];
      int end = ends[i] < 0 ? (ends[i] + dim_value) : ends[i];
      start = std::max(start, 0);
      end = std::max(end, 0);
      end = std::min(end, dim_value);
      out_dims[axes[i]] = end - start;
      real_starts[axes[i]] = start;
      real_ends[axes[i]] = end;
    }
  }
  const int LEN = in_dims.size();
  int dst_step[LEN];
  for (int i = 0; i < in_dims.size(); ++i) {
    dst_step[i] = 1;
  }
  int src_step[LEN];
  for (int i = 0; i < in_dims.size(); ++i) {
    src_step[i] = 1;
  }
  int out_num = out_dims[in_dims.size() - 1];
  for (int i = in_dims.size() - 2; i >= 0; i--) {
    dst_step[i] = out_dims[i + 1] * dst_step[i + 1];
    src_step[i] = in_dims[i + 1] * src_step[i + 1];
    out_num *= out_dims[i];
  }

  for (int dst_id = 0; dst_id < out_num; dst_id++) {
    int src_id = 0;
    int index_id = dst_id;
    for (int j = 0; j < out_dims.size(); j++) {
      int cur_id = index_id / dst_step[j];
      index_id = index_id % dst_step[j];
      src_id += (cur_id + real_starts[j]) * src_step[j];
    }
    out[dst_id] = input[src_id];
  }
}

template void slice(const float* input,
                    std::vector<int64_t> dims,
                    std::vector<int> axes,
                    std::vector<int> starts,
                    std::vector<int> ends,
                    float* out,
                    Context<TARGET(kARM)>* ctx);
template void slice(const int* input,
                    std::vector<int64_t> dims,
                    std::vector<int> axes,
                    std::vector<int> starts,
                    std::vector<int> ends,
                    int* out,
                    Context<TARGET(kARM)>* ctx);
template void slice(const int64_t* input,
                    std::vector<int64_t> dims,
                    std::vector<int> axes,
                    std::vector<int> starts,
                    std::vector<int> ends,
                    int64_t* out,
                    Context<TARGET(kARM)>* ctx);
template void slice(const bool* input,
                    std::vector<int64_t> dims,
                    std::vector<int> axes,
                    std::vector<int> starts,
                    std::vector<int> ends,
                    bool* out,
                    Context<TARGET(kARM)>* ctx);

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
