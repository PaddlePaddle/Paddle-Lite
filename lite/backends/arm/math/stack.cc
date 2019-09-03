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

#include "lite/backends/arm/math/stack.h"
#include <cstddef>
#include <utility>
#include <vector>
#include "lite/backends/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

void stack(std::vector<lite::Tensor *> x, lite::Tensor *y, int axis) {
  if (axis < 0) axis += (x[0]->dims().size() + 1);
  int n = x.size();
  auto *y_data = y->mutable_data<float>();
  std::vector<const float *> x_datas(n);
  for (int i = 0; i < n; i++) x_datas[i] = x[i]->data<float>();

  int pre = 1, post = 1;
  auto &dim = x[0]->dims();
  for (auto i = 0; i < axis; ++i) pre *= dim[i];
  for (auto i = axis; i < dim.size(); ++i) post *= dim[i];

  auto x_data_arr = x_datas.data();

  size_t x_offset = 0;
  size_t y_offset = 0;
  for (int i = 0; i < pre; i++) {
    for (int j = 0; j < n; j++) {
      std::memcpy(
          y_data + y_offset, x_data_arr[j] + x_offset, post * sizeof(float));
      y_offset += post;
    }
    x_offset += post;
  }
}

} /* namespace math */
} /* namespace arm */
} /* namespace lite */
} /* namespace paddle */
