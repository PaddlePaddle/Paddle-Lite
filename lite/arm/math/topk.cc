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

#include "lite/arm/math/topk.h"
#include <utility>
#include <vector>
#include "lite/arm/math/funcs.h"
namespace paddle {
namespace lite {
namespace arm {
namespace math {
template <typename T>
bool comp_func(std::pair<T, int> a, std::pair<T, int> b) {
  return (a.first > b.first);
}

template <typename T>
void topk(const T* in_data, T* out_val, int* out_ind, int m, int n, int k) {
  for (int i = 0; i < m; i++) {
    const T* in_tmp = in_data + i * n;
    T* out_val_tmp = out_val + i * k;
    int* out_ind_tmp = out_ind + i * k;
    std::vector<std::pair<T, int>> vec;
    for (int j = 0; j < n; j++) {
      vec.push_back(std::make_pair(in_tmp[j], j));
    }
    std::partial_sort(vec, vec + k, vec + n, comp_func);
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
