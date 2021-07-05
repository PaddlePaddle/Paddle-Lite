/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "lite/backends/x86/math/sequence_scale.h"
#include "lite/fluid/lod.h"

namespace paddle {
namespace lite_metal {
namespace x86 {
namespace math {

template <typename T>
class ScaleLoDTensorFunctor<lite_metal::TargetType::kX86, T> {
 public:
  void operator()(const lite_metal::Context<lite_metal::TargetType::kX86>& context,
                  const T* scales,
                  lite_metal::Tensor* seq) {
    const size_t level = 0;
    auto lod = seq->lod();
    const size_t num_seq = lod[level].size() - 1;
    size_t seq_width = seq->dims()[1];
    lite_metal::LoD abs_offset_lod = lite_metal::fluid::ToAbsOffset(lod);

    T* seq_data = seq->template mutable_data<T>(lite_metal::TargetType::kX86);
    for (size_t i = 0; i < num_seq; ++i) {
      for (size_t j = lod[level][i] * seq_width;
           j < lod[level][i + 1] * seq_width;
           ++j) {
        seq_data[j] *= scales[i];
      }
    }
  }
};

template class ScaleLoDTensorFunctor<lite_metal::TargetType::kX86, float>;

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
