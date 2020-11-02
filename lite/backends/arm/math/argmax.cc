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

#include "lite/backends/arm/math/argmax.h"
#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <utility>
#include <vector>
#include "lite/backends/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

template <typename InType, typename OutType>
void argmax_func(const lite::Tensor *input,
                 const int axis,
                 lite::Tensor *output) {
  auto input_ddim = input->dims();
  auto output_ddim = output->dims();

  const int size = input_ddim[axis];
  const int in_channel = input_ddim.count(axis, input_ddim.size());
  const int out_channel = output_ddim.count(axis, output_ddim.size());
  const int in_stride = input_ddim.count(axis + 1, input_ddim.size());
  const int out_stride = input_ddim.count(0, axis);

  for (int n = 0; n < out_stride; n++) {
    for (int k = 0; k < in_stride; k++) {
      const InType *in_ptr = input->data<InType>() + n * in_channel + k;
      std::vector<std::pair<InType, OutType>> vec;
      vec.resize(size);
      for (int i = 0; i < size; i++) {
        vec[i] = std::make_pair(in_ptr[i * in_stride], i);
      }
      // sort
      std::partial_sort(vec.begin(),
                        vec.begin() + 1,
                        vec.end(),
                        std::greater<std::pair<InType, OutType>>());

      // out
      OutType *out_ptr = output->mutable_data<OutType>() + n * out_channel + k;
      *out_ptr = vec[0].second;
    }
  }
}

template void argmax_func<float, int32_t>(const lite::Tensor *input,
                                          const int axis,
                                          lite::Tensor *output);
template void argmax_func<float, int64_t>(const lite::Tensor *input,
                                          const int axis,
                                          lite::Tensor *output);
#ifdef LITE_BUILD_EXTRA
template void argmax_func<double, int32_t>(const lite::Tensor *input,
                                           const int axis,
                                           lite::Tensor *output);
template void argmax_func<double, int64_t>(const lite::Tensor *input,
                                           const int axis,
                                           lite::Tensor *output);
template void argmax_func<int64_t, int32_t>(const lite::Tensor *input,
                                            const int axis,
                                            lite::Tensor *output);
template void argmax_func<int64_t, int64_t>(const lite::Tensor *input,
                                            const int axis,
                                            lite::Tensor *output);
template void argmax_func<int32_t, int32_t>(const lite::Tensor *input,
                                            const int axis,
                                            lite::Tensor *output);
template void argmax_func<int32_t, int64_t>(const lite::Tensor *input,
                                            const int axis,
                                            lite::Tensor *output);
template void argmax_func<int16_t, int32_t>(const lite::Tensor *input,
                                            const int axis,
                                            lite::Tensor *output);
template void argmax_func<int16_t, int64_t>(const lite::Tensor *input,
                                            const int axis,
                                            lite::Tensor *output);
template void argmax_func<uint8_t, int32_t>(const lite::Tensor *input,
                                            const int axis,
                                            lite::Tensor *output);
template void argmax_func<uint8_t, int64_t>(const lite::Tensor *input,
                                            const int axis,
                                            lite::Tensor *output);
#endif
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
