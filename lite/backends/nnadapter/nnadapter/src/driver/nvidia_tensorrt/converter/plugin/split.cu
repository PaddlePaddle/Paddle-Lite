// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "driver/nvidia_tensorrt/converter/plugin/split.h"

namespace nnadapter {
namespace nvidia_tensorrt {

template <typename T>
__device__ int UpperBound(T const* vals, int n, T const& key) {
  int i = 0;
  while (n > 0) {
    int m = n / 2;
    int j = i + m;
    if (!(key < vals[j])) {
      i = j + 1;
      n -= (m + 1);
    } else {
      n = m;
    }
  }
  return i;
}

// The following part of the code refers to onnx-tensorrt
// https://github.com/onnx/onnx-tensorrt/blob/master/Split.cu
template <typename T>
__global__ void SplitKernel(int nsegment,
                            int const* __restrict__ segment_offsets,
                            T const* __restrict__ idata,
                            T* const* odatas,
                            int inner_cols,
                            int axis_shape,
                            int outer_rows) {
  int x0 = threadIdx.x + blockIdx.x * blockDim.x;
  int src_y0 = threadIdx.y + blockIdx.y * blockDim.y;
  int z0 = threadIdx.z + blockIdx.z * blockDim.z;
  for (int z = z0; z < outer_rows; z += blockDim.z * gridDim.z) {
    for (int src_y = src_y0; src_y < axis_shape;
         src_y += blockDim.y * gridDim.y) {
      for (int x = x0; x < inner_cols; x += blockDim.x * gridDim.x) {
        int segment = UpperBound(segment_offsets, nsegment, src_y) - 1;
        int dst_y = src_y - segment_offsets[segment];
        int dst_ny = segment_offsets[segment + 1] - segment_offsets[segment];
        odatas[segment][x + inner_cols * (dst_y + dst_ny * z)] =
            idata[x + inner_cols * (src_y + axis_shape * z)];
      }
    }
  }
}

template <typename T>
cudaError_t Split(const T* input,
                  T* const* outputs,
                  int* segment_offsets,
                  int nsegment,
                  int inner_cols,
                  int axis_shape,
                  int outer_rows,
                  int batch_size,
                  cudaStream_t stream) {
  int batch_outer_rows = outer_rows * batch_size;

  dim3 block(32, 16);
  dim3 grid(std::min((inner_cols - 1) / block.x + 1, 65535u),
            std::min((axis_shape - 1) / block.y + 1, 65535u),
            std::min((outer_rows - 1) / block.z + 1, 65535u));
  SplitKernel<<<grid, block, 0, stream>>>(nsegment,
                                          segment_offsets,
                                          input,
                                          outputs,
                                          inner_cols,
                                          axis_shape,
                                          batch_outer_rows);
  return cudaGetLastError();
}

template cudaError_t Split(const float* input,
                           float* const* outputs,
                           int* segment_offsets,
                           int nsegment,
                           int inner_cols,
                           int axis_shape,
                           int outer_rows,
                           int batch_size,
                           cudaStream_t stream);

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
