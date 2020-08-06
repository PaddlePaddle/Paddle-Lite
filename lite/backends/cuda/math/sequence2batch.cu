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

#include <algorithm>

#include "lite/backends/cuda/cuda_utils.h"
#include "lite/backends/cuda/math/sequence2batch.h"
#include "lite/backends/cuda/math/utils.h"

namespace paddle {
namespace lite {
namespace cuda {
namespace math {

template <typename T>
__global__ void CopyMatrixRowsKernel(const T* src,
                                     T* dst,
                                     const uint64_t* index,
                                     int height,
                                     int width,
                                     bool is_src_index) {
  CUDA_KERNEL_LOOP(tid, height * width) {
    int row = tid / width;
    int idx = tid % width;
    int src_row = is_src_index ? index[row] : row;
    int dst_row = is_src_index ? row : index[row];
    dst[dst_row * width + idx] = src[src_row * width + idx];
  }
}

template <typename T>
void CopyMatrixRowsFunctor<T>::operator()(
    const lite::Tensor& src,
    lite::Tensor* dst,
    const std::vector<uint64_t>& index_lod,
    bool is_src_index,
    const cudaStream_t& stream) {
  auto src_dims = src.dims();
  auto dst_dims = dst->dims();
  CHECK_EQ(src_dims.size(), 2) << "The src must be matrix with rank 2.";
  CHECK_EQ(dst_dims.size(), 2) << "The dst must be matrix with rank 2.";
  CHECK_EQ(src_dims[1], dst_dims[1])
      << "The width of src and dst must be same.";
  int height = dst_dims[0];
  int width = dst_dims[1];
  const auto* src_data = src.data<T>();
  auto* dst_data = dst->template mutable_data<T>(TARGET(kCUDA));

  index_tensor_.Resize({static_cast<int64_t>(index_lod.size())});
  auto* index_tensor_data = index_tensor_.mutable_data<uint64_t>(TARGET(kCUDA));
  TargetWrapperCuda::MemcpyAsync(index_tensor_data,
                                 index_lod.data(),
                                 sizeof(uint64_t) * index_lod.size(),
                                 IoDirection::HtoD,
                                 stream);
  CopyMatrixRowsKernel<
      T><<<CUDA_GET_BLOCKS(height * width), CUDA_NUM_THREADS, 0, stream>>>(
      src_data, dst_data, index_tensor_data, height, width, is_src_index);
  CUDA_POST_KERNEL_CHECK;
}

template class CopyMatrixRowsFunctor<float>;
template class CopyMatrixRowsFunctor<half>;

template class LoDTensor2BatchFunctor<float>;
template class LoDTensor2BatchFunctor<half>;

template class Batch2LoDTensorFunctor<float>;
template class Batch2LoDTensorFunctor<half>;

}  // namespace math
}  // namespace cuda
}  // namespace lite
}  // namespace paddle
