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

#pragma once
#include <cublasXt.h>
#include <cublas_api.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include "lite/utils/log/cp_logging.h"

#if (CUBLAS_VER_MAJOR * 10 + CUBLAS_VER_MINOR) >= 101
#include <cublasLt.h>
#endif

/*
 * This file contains some CUDA specific utils.
 */

// For quickly implementing the prototype, some of the following code snippets
// are borrowed from project MXNet, great thanks for the original developers.

#define CHECK_CUDA_ERROR(msg)                                                \
  {                                                                          \
    auto e = cudaGetLastError();                                             \
    CHECK_EQ(e, cudaSuccess) << (msg) << " CUDA: " << cudaGetErrorString(e); \
  }

#define CUDA_CALL(func)                                      \
  {                                                          \
    auto e = (func);                                         \
    CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading) \
        << "CUDA: " << cudaGetErrorString(e);                \
  }

#define CUDA_POST_KERNEL_CHECK CUDA_CALL(cudaPeekAtLastError())

#define CUBLAS_CALL(func)                                        \
  {                                                              \
    auto e = (func);                                             \
    CHECK_EQ(e, CUBLAS_STATUS_SUCCESS)                           \
        << "cuBlas: " << paddle::lite::cuda::CublasErrorInfo(e); \
  }

#define CUDNN_VERSION_MIN(major, minor, patch) \
  (CUDNN_VERSION >= (major * 1000 + minor * 100 + patch))

#define CUDNN_CHECK(condition)                                           \
  {                                                                      \
    cudnnStatus_t status = condition;                                    \
    CHECK_EQ(status, CUDNN_STATUS_SUCCESS) << CudnnGetErrorInfo(status); \
  }

const int CUDA_NUM_THREADS = 512;
// CUDA: number of blocks for threads.
inline int CUDA_GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}
inline int CUDA_GET_BLOCKS(const int N, const int base) {
  return (N + base - 1) / base;
}
#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

namespace paddle {
namespace lite {
namespace cuda {

static const char* CublasErrorInfo(int error) {
  switch (error) {
#define LITE_CUBLAS_ERROR_INFO(xx) \
  case xx:                         \
    return #xx;                    \
    break;
    LITE_CUBLAS_ERROR_INFO(CUBLAS_STATUS_NOT_INITIALIZED);
    LITE_CUBLAS_ERROR_INFO(CUBLAS_STATUS_ALLOC_FAILED);
    LITE_CUBLAS_ERROR_INFO(CUBLAS_STATUS_INVALID_VALUE);
    LITE_CUBLAS_ERROR_INFO(CUBLAS_STATUS_ARCH_MISMATCH);
    LITE_CUBLAS_ERROR_INFO(CUBLAS_STATUS_MAPPING_ERROR);
    LITE_CUBLAS_ERROR_INFO(CUBLAS_STATUS_EXECUTION_FAILED);
    LITE_CUBLAS_ERROR_INFO(CUBLAS_STATUS_INTERNAL_ERROR);
    LITE_CUBLAS_ERROR_INFO(CUBLAS_STATUS_NOT_SUPPORTED);
    LITE_CUBLAS_ERROR_INFO(CUBLAS_STATUS_LICENSE_ERROR);
#undef LITE_CUBLAS_ERROR_INFO
    default:
      return "unknown error";
  }
}

static const char* CudnnGetErrorInfo(cudnnStatus_t status) {
  switch (status) {
    case CUDNN_STATUS_SUCCESS:
      return "CUDNN_STATUS_SUCCESS";
    case CUDNN_STATUS_NOT_INITIALIZED:
      return "CUDNN_STATUS_NOT_INITIALIZED";
    case CUDNN_STATUS_ALLOC_FAILED:
      return "CUDNN_STATUS_ALLOC_FAILED";
    case CUDNN_STATUS_BAD_PARAM:
      return "CUDNN_STATUS_BAD_PARAM";
    case CUDNN_STATUS_INTERNAL_ERROR:
      return "CUDNN_STATUS_INTERNAL_ERROR";
    case CUDNN_STATUS_INVALID_VALUE:
      return "CUDNN_STATUS_INVALID_VALUE";
    case CUDNN_STATUS_ARCH_MISMATCH:
      return "CUDNN_STATUS_ARCH_MISMATCH";
    case CUDNN_STATUS_MAPPING_ERROR:
      return "CUDNN_STATUS_MAPPING_ERROR";
    case CUDNN_STATUS_EXECUTION_FAILED:
      return "CUDNN_STATUS_EXECUTION_FAILED";
    case CUDNN_STATUS_NOT_SUPPORTED:
      return "CUDNN_STATUS_NOT_SUPPORTED";
    case CUDNN_STATUS_LICENSE_ERROR:
      return "CUDNN_STATUS_LICENSE_ERROR";
#if CUDNN_VERSION_MIN(6, 0, 0)
    case CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING:
      return "CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING";
#endif
#if CUDNN_VERSION_MIN(7, 0, 0)
    case CUDNN_STATUS_RUNTIME_IN_PROGRESS:
      return "CUDNN_STATUS_RUNTIME_IN_PROGRESS";
    case CUDNN_STATUS_RUNTIME_FP_OVERFLOW:
      return "CUDNN_STATUS_RUNTIME_FP_OVERFLOW";
#endif
#if CUDNN_VERSION_MIN(8, 0, 0)
    case CUDNN_STATUS_VERSION_MISMATCH:
      return "CUDNN_STATUS_VERSION_MISMATCH";
#endif
  }
  return "Unknown cudnn status";
}

}  // namespace cuda
}  // namespace lite
}  // namespace paddle
