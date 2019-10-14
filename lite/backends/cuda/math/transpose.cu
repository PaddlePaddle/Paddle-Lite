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

#include "lite/backends/cuda/cuda_utils.h"
#include "lite/backends/cuda/math/transpose.h"
#include "lite/backends/cuda/math/utils.h"

namespace paddle {
namespace lite {
namespace cuda {
namespace math {

constexpr int kTileDim = 32;
constexpr int kBlockRows = 8;
constexpr int CUDA_NUM_THREADS = 128;

// Splits the original matrix into submatrices with size 32 * 32.
// Reference https://devblogs.nvidia.com/efficient-matrix-transpose-cuda-cc/
template <typename T>
__global__ void BatchTranspose2DCUDAKernel(const int N,
                                           const int H,
                                           const int W,
                                           const int dh,
                                           const int dw,
                                           const T* input,
                                           T* out) {
  __shared__ T tile[kTileDim][kTileDim + 1];  // plus 1 to prevent bank confict.
  const int n = blockIdx.x / (dh * dw);
  const int k = blockIdx.x % (dh * dw);
  const int r = k / dw;
  const int c = k % dw;
  const int offset = n * H * W;
  int x = c * kTileDim + threadIdx.x;
  int y = r * kTileDim + threadIdx.y;
  if (x < W) {
    for (int i = 0; threadIdx.y + i < kTileDim && y + i < H; i += kBlockRows) {
#if __CUDA_ARCH__ >= 350 || defined(__HIP_PLATFORM_HCC__)
      tile[threadIdx.y + i][threadIdx.x] =
          __ldg(input + offset + (y + i) * W + x);
#else
      tile[threadIdx.y + i][threadIdx.x] = input[offset + (y + i) * W + x];
#endif
    }
  }
  __syncthreads();
  x = r * kTileDim + threadIdx.x;
  y = c * kTileDim + threadIdx.y;
  if (x < H) {
    for (int i = 0; threadIdx.y + i < kTileDim && y + i < W; i += kBlockRows) {
      out[offset + (y + i) * H + x] = tile[threadIdx.x][threadIdx.y + i];
    }
  }
}

template <typename T>
void BatchTranspose2DCUDAImpl(const int N,
                              const int H,
                              const int W,
                              const T* input,
                              T* out,
                              CUDAContext* ctx) {
  const int dh = (H + kTileDim - 1) / kTileDim;
  const int dw = (W + kTileDim - 1) / kTileDim;
  BatchTranspose2DCUDAKernel<
      T><<<N * dh * dw, dim3(kTileDim, kBlockRows), 0, ctx->exec_stream()>>>(
      N, H, W, dh, dw, input, out);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) LOG(INFO) << cudaGetErrorString(error);
}

#define TYPE_SPECIALIZED_CUDA_NCHW2NHWC(T)             \
  template <>                                          \
  void NCHW2NHWC<T>(const int N,                       \
                    const int C,                       \
                    const int HxW,                     \
                    const T* X,                        \
                    T* Y,                              \
                    CUDAContext* ctx) {                \
    BatchTranspose2DCUDAImpl<T>(N, C, HxW, X, Y, ctx); \
  }
TYPE_SPECIALIZED_CUDA_NCHW2NHWC(float)
TYPE_SPECIALIZED_CUDA_NCHW2NHWC(int8_t)
#undef TYPE_SPECIALIZED_CUDA_NCHW2NHWC

#define TYPE_SPECIALIZED_CUDA_NHWC2NCHW(T)             \
  template <>                                          \
  void NHWC2NCHW<T>(const int N,                       \
                    const int C,                       \
                    const int HxW,                     \
                    const T* X,                        \
                    T* Y,                              \
                    CUDAContext* ctx) {                \
    BatchTranspose2DCUDAImpl<T>(N, HxW, C, X, Y, ctx); \
  }
TYPE_SPECIALIZED_CUDA_NHWC2NCHW(float)
TYPE_SPECIALIZED_CUDA_NHWC2NCHW(int8_t)
#undef TYPE_SPECIALIZED_CUDA_NHWC2NCHW

template <typename T>
__global__ void TransposeCUDAKernel(const int size,
                                    const int ndim,
                                    const int* X_strides,
                                    const int* Y_dims,
                                    const T* X,
                                    T* Y) {
  const int Y_index = blockIdx.x * CUDA_NUM_THREADS + threadIdx.x;
  if (Y_index < size) {
    int X_index = 0;
    int v = Y_index;
#pragma unroll
    for (int i = ndim - 1; i >= 0; --i) {
      X_index += v % Y_dims[i] * X_strides[i];
      v /= Y_dims[i];
    }
#if __CUDA_ARCH__ >= 350 || defined(__HIP_PLATFORM_HCC__)
    Y[Y_index] = __ldg(X + X_index);
#else
    Y[Y_index] = X[X_index];
#endif
  }
}

template <typename T>
void TransposeCUDAImpl(const std::vector<int64_t>& X_dims,
                       const std::vector<int>& axes,
                       const T* X,
                       T* Y,
                       CUDAContext* ctx) {
  CHECK_EQ(X_dims.size(), axes.size()) << "dimension size should be equal";
  int ndim = X_dims.size();
  std::vector<int> strides(ndim, 0);
  std::vector<int> Y_dims(ndim, 0);
  std::vector<int> buf(ndim, 0);
  int cur_stride = 1;
  for (int i = ndim - 1; i >= 0; --i) {
    buf[i] = cur_stride;
    cur_stride *= X_dims[i];
  }
  for (int i = 0; i < ndim; ++i) {
    strides[i] = buf[axes[i]];
  }
  int size = 1;
  for (int i = 0; i < ndim; ++i) {
    Y_dims[i] = static_cast<int>(X_dims[axes[i]]);
    size *= X_dims[i];
  }

  lite::Tensor Y_dims_, strides_;
  Y_dims_.Resize(std::vector<int64_t>({ndim}));
  int* d_y_dims = Y_dims_.mutable_data<int>(TARGET(kCUDA));
  CopySync<TARGET(kCUDA)>(
      d_y_dims, Y_dims.data(), sizeof(int) * Y_dims.size(), IoDirection::HtoD);

  strides_.Resize(std::vector<int64_t>({ndim}));
  int* d_strides = strides_.mutable_data<int>(TARGET(kCUDA));
  CopySync<TARGET(kCUDA)>(d_strides,
                          strides.data(),
                          sizeof(int) * strides.size(),
                          IoDirection::HtoD);

  const int M = (size + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
  TransposeCUDAKernel<<<M, CUDA_NUM_THREADS, 0, ctx->exec_stream()>>>(
      size, ndim, d_strides, d_y_dims, X, Y);
  auto e = cudaGetLastError();
  CHECK_EQ(e, cudaSuccess) << " CUDA: " << cudaGetErrorString(e);
}

#define TYPE_SPECIALIZED_CUDA_TRANSPOSE(T)              \
  template <>                                           \
  void Transpose<T>(const std::vector<int64_t>& X_dims, \
                    const std::vector<int>& axes,       \
                    const T* X,                         \
                    T* Y,                               \
                    CUDAContext* ctx) {                 \
    TransposeCUDAImpl<T>(X_dims, axes, X, Y, ctx);      \
  }
TYPE_SPECIALIZED_CUDA_TRANSPOSE(float)
#undef TYPE_SPECIALIZED_CUDA_TRANSPOSEF

}  // namespace math
}  // namespace cuda
}  // namespace lite
}  // namespace paddle
