// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/cuda/topk_pooling_compute.h"

#include <limits>
#include <vector>

#include "lite/backends/cuda/target_wrapper.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

template <typename Dtype>
__global__ void top_k_pooling_batch_kernel_reduction(Dtype *output_data,
                                                     const Dtype *input,
                                                     const int *height_offset,
                                                     const int *width_offset,
                                                     const int batch_size,
                                                     const int channel_num,
                                                     const int height_stride,
                                                     const int width_stride,
                                                     const int k) {
  const Dtype *input_start =
      input +
      (blockIdx.x * channel_num + blockIdx.y) * height_stride * width_stride;
  Dtype *output_start =
      output_data + (blockIdx.x * channel_num + blockIdx.y) * k;

  int width = width_offset[blockIdx.x + 1] - width_offset[blockIdx.x];
  int height = height_offset[blockIdx.x + 1] - height_offset[blockIdx.x];
  int real_k = k < height * width ? k : height * width;

  extern __shared__ Dtype smem[];

  Dtype min_val = -100000.0f;
  for (int j = threadIdx.x; j < height * width; j += blockDim.x) {
    int index_tmp = (j / width) * width_stride + j % width;
    smem[j] = input_start[index_tmp];
  }
  __syncthreads();

  // get max val
  int t = 0;
  for (; t < real_k; ++t) {
    // reduction
    for (int gap = height * width; gap > 1;) {
      if (threadIdx.x == 0) {  // edge cond
        if (gap % 2 != 0) {
          Dtype value_first = smem[0];
          Dtype value_gap = smem[gap - 1];
          if (value_first < value_gap) {
            smem[0] = value_gap;
            smem[gap - 1] = value_first;
          }
        }
      }
      gap >>= 1;
      for (int j = threadIdx.x; j < gap; j += blockDim.x) {
        Dtype value_first = smem[j];
        Dtype value_gap = smem[j + gap];
        if (value_first < value_gap) {
          smem[j] = value_gap;
          smem[j + gap] = value_first;
        }
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      output_start[t] = smem[0];
      smem[0] = min_val;
    }
    __syncthreads();
  }
  for (int i = threadIdx.x; i < (k - t); i += blockDim.x) {
    // output_start[t + i] = 0.0f;
  }
}

template <typename T>
void TopkPoolingCompute<T>::PrepareForRun() {
  int device_id = lite::TargetWrapperCuda::GetCurDevice();
  cudaDeviceProp deviceProp;
  CUDA_CALL(cudaGetDeviceProperties(&deviceProp, device_id));
  _shared_mem_size = deviceProp.sharedMemPerBlock;
}

template <typename T>
void TopkPoolingCompute<T>::Run() {
  auto &param = this->Param<param_t>();
  auto &ctx = this->ctx_->template As<CUDAContext>();
  auto cuda_stream = ctx.exec_stream();

  CHECK(param.X->lod().size() > 0 && param.X->lod()[0].size() > 0)
      << "X sequence offset is not valid";
  CHECK(param.Y->lod().size() > 0 && param.Y->lod()[0].size() > 0)
      << "Y sequence offset is not valid";

  int width_offset_len = param.X->lod()[0].size();
  lite::DDim width_offset_shape(std::vector<int64_t>{width_offset_len});
  _width_offset.Resize(width_offset_shape);
  std::vector<int> width_lod_0(width_offset_len, 0);
  for (size_t i = 0; i < param.X->lod()[0].size(); ++i) {
    width_lod_0[i] = static_cast<int>(param.X->lod()[0][i]);
  }
  lite::TargetWrapperCuda::MemcpyAsync(
      _width_offset.mutable_data<int>(TARGET(kCUDA)),
      width_lod_0.data(),
      sizeof(int) * width_offset_len,
      lite::IoDirection::HtoD,
      cuda_stream);

  int height_offset_len = param.Y->lod()[0].size();
  lite::DDim height_offset_shape(std::vector<int64_t>{height_offset_len});
  _height_offset.Resize(height_offset_shape);
  std::vector<int> height_lod_0(height_offset_len, 0);
  for (size_t i = 0; i < param.Y->lod()[0].size(); ++i) {
    height_lod_0[i] = static_cast<int>(param.Y->lod()[0][i]);
  }
  lite::TargetWrapperCuda::MemcpyAsync(
      _height_offset.mutable_data<int>(TARGET(kCUDA)),
      height_lod_0.data(),
      sizeof(int) * height_offset_len,
      lite::IoDirection::HtoD,
      cuda_stream);

  const Tensor *x_tensor = param.X;
  Tensor *out_tensor = param.Out;
  const T *in_data = x_tensor->data<T>();
  T *out_data = out_tensor->mutable_data<T>(TARGET(kCUDA));

  int num = x_tensor->dims()[0];
  int channel = x_tensor->dims()[1];
  int height = x_tensor->dims()[2];
  int width = x_tensor->dims()[3];

  const int *height_offset = _height_offset.data<int>();
  const int *width_offset = _width_offset.data<int>();

  int feat_map_size = height * width;

  if (feat_map_size * sizeof(T) <= _shared_mem_size) {
    dim3 blocks(num, channel);
    dim3 threads(32, 1);

    top_k_pooling_batch_kernel_reduction<
        T><<<blocks, threads, feat_map_size * sizeof(T), cuda_stream>>>(
        out_data,
        in_data,
        height_offset,
        width_offset,
        num,
        channel,
        height,
        width,
        param.top_k);
  } else {
    LOG(FATAL) << "Not implemented. Exceeded the shared memory limit.";
  }
  CUDA_POST_KERNEL_CHECK;
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(topk_pooling,
                     kCUDA,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::cuda::TopkPoolingCompute<float>,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .Finalize();
