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

#include "lite/backends/x86/math/sequence2batch.h"

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

template <typename T>
class CopyMatrixRowsFunctor<lite::TargetType::kX86, T> {
 public:
  void operator()(const lite::Context<lite::TargetType::kX86>& context,
                  const lite::Tensor& src,
                  const std::vector<uint64_t>& index_lod,
                  lite::Tensor* dst,
                  bool is_src_index) {
    const uint64_t* index = index_lod.data();
    const auto& src_dims = src.dims();
    const auto& dst_dims = dst->dims();
    CHECK_EQ(src_dims.size(), 2UL) << "The src must be matrix with rank 2.";
    CHECK_EQ(dst_dims.size(), 2UL) << "The dst must be matrix with rank 2.";
    CHECK_EQ(src_dims[1], dst_dims[1])
        << "The width of src and dst must be same.";
    auto height = dst_dims[0];
    auto width = dst_dims[1];
    auto* src_data = src.data<T>();
    auto* dst_data = dst->template mutable_data<T>();
    const int sz = width * sizeof(T);
    if (is_src_index) {
      for (int i = 0; i < height; ++i) {
        memcpy(dst_data + i * width, src_data + index[i] * width, sz);
      }
    } else {
      for (int i = 0; i < height; ++i) {
        memcpy(dst_data + index[i] * width, src_data + i * width, sz);
      }
    }
  }
};

template class CopyMatrixRowsFunctor<lite::TargetType::kX86, float>;
template class CopyMatrixRowsFunctor<lite::TargetType::kX86, double>;

template class LoDTensor2BatchFunctor<lite::TargetType::kX86, float>;
template class LoDTensor2BatchFunctor<lite::TargetType::kX86, double>;
template class Batch2LoDTensorFunctor<lite::TargetType::kX86, float>;
template class Batch2LoDTensorFunctor<lite::TargetType::kX86, double>;

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
