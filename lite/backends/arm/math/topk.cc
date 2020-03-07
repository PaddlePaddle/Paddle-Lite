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

#include "lite/backends/arm/math/topk.h"
#include <utility>
#include <vector>
#include "lite/backends/arm/math/funcs.h"
namespace paddle {
namespace lite {
namespace arm {
namespace math {
bool comp_func(std::pair<float, int> a, std::pair<float, int> b) {
  return (a.first > b.first);
}

void topk(const float* in_data,
          float* out_val,
          int64_t* out_ind,
          int m,
          int n,
          int k,
          Context<TARGET(kARM)>* ctx) {
  for (int i = 0; i < m; i++) {
    const float* in_tmp = in_data + i * n;
    float* out_val_tmp = out_val + i * k;
    int64_t* out_ind_tmp = out_ind + i * k;
    std::vector<std::pair<float, int>> vec;
    for (int j = 0; j < n; j++) {
      vec.push_back(std::make_pair(in_tmp[j], j));
    }
    std::partial_sort(vec.begin(), vec.begin() + k, vec.end(), comp_func);
    for (int q = 0; q < k; q++) {
      out_val_tmp[q] = vec[q].first;
      out_ind_tmp[q] = vec[q].second;
    }
  }
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
