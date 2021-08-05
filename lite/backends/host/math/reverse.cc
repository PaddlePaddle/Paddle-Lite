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

#include "lite/backends/host/math/reverse.h"
#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace paddle {
namespace lite {
namespace host {
namespace math {

int cal(int i, int a, int b) {
  int b_gr_num = i / a;
  int b_in_gr = i % a;
  int s_gr_num = b_in_gr / b;
  int s_in_gr = b_in_gr % b;
  s_gr_num = a / b - 1 - s_gr_num;
  int i_now = b_gr_num * a + s_gr_num * b + s_in_gr;
  return i_now;
}

template <typename T>
void reverse_func(const lite::Tensor *input,
                  std::vector<int> axis,
                  lite::Tensor *output) {
  std::sort(axis.begin(), axis.end());
  auto input_ddim = input->dims();
  int count = input_ddim.count(0, input_ddim.size());

  const T *in_ptr = input->data<T>();
  T *out_ptr = output->mutable_data<T>();

  std::vector<int> stride = axis;
  for (int i = 0; i < stride.size(); i++) {
    stride[i] = input_ddim.count(axis[i], input_ddim.size());
  }
  for (int i = 0; i < count; i++) {
    int now_i = i;
    for (int j = stride.size() - 1; j >= 0; j--) {
      int a = stride[j];
      int b = a / input_ddim[axis[j]];
      now_i = cal(now_i, a, b);
    }
    out_ptr[now_i] = in_ptr[i];
  }
}

template void reverse_func<float>(const lite::Tensor *input,
                                  std::vector<int> axis,
                                  lite::Tensor *output);
}  // namespace math
}  // namespace host
}  // namespace lite
}  // namespace paddle
