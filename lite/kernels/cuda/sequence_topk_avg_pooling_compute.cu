/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <limits>
#include <vector>
#include "lite/core/op_registry.h"
#include "lite/kernels/cuda/sequence_topk_avg_pooling_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

template <typename Dtype>
__global__ void topk_avg_pooling_kernel_by_row_improve(
    Dtype *output_data,
    const Dtype *input,
    const int *gpu_input_offset_l,
    const int *gpu_input_offset_r,
    const int row_max,
    const int col_max,
    const int topk_size,
    const int *topks,
    const int feat_map_num) {
  int row =
      gpu_input_offset_l[blockIdx.x + 1] - gpu_input_offset_l[blockIdx.x];  // 8
  int col = gpu_input_offset_r[blockIdx.x + 1] -
            gpu_input_offset_r[blockIdx.x];  // 30

  int max_k = topks[topk_size - 1];
  max_k = max_k < col ? max_k : col;

  extern __shared__ Dtype smem[];  // H*W

  const Dtype *fm_row_in_data = input +
                                blockIdx.x * row_max * feat_map_num * col_max +
                                blockIdx.y * row_max * col_max;

  for (int i = threadIdx.x; i < row * col_max; i += blockDim.x) {
    smem[i] = fm_row_in_data[i];
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < row; idx += blockDim.x) {
    Dtype *fm_row_out_data =
        output_data +
        (gpu_input_offset_l[blockIdx.x] + idx) * feat_map_num * topk_size +
        blockIdx.y * topk_size;
    for (int i = 0; i < topk_size; ++i) {
      fm_row_out_data[i] = 0;
    }
    Dtype *smem_start_col = smem + idx * col_max;

    int counter = max_k;  // topk_size;
    Dtype last_max_val = -20000.0;
    while (counter) {
      Dtype max_val = -10000.0;
      int max_pos = 0;  // -1;
      int m = 0;
      for (; m < col; m++) {
        Dtype cur_data = smem_start_col[m];
        if (cur_data > max_val) {
          max_val = cur_data;
          max_pos = m;
          last_max_val = max_val;
        }
      }
      if (max_val < -9999.0) {  // == -10000.0
        max_val = last_max_val;
      }
      smem_start_col[max_pos] = -10000000.0;

      int i = max_k - counter;
      for (int c = 0; c < topk_size; c++) {
        if (i <= topks[c] - 1) {
          fm_row_out_data[c] += max_val;
        }
      }
      counter--;
    }
    __syncthreads();
    // compute avg
    for (int i = 0; i < topk_size; i++) {
      fm_row_out_data[i] = fm_row_out_data[i] / topks[i];
    }
  }
}

template <typename Dtype>
__global__ void topk_avg_pooling_kernel_for_big_data(
    Dtype *output_data,
    const Dtype *input_data,
    const int *gpu_input_offset_l,
    const int *gpu_input_offset_r,
    const int row_max,
    const int col_max,
    const int topk_size,
    const int *topks,
    const int feat_map_num,
    const int actual_row_in_shared_mem) {
  int row = gpu_input_offset_l[blockIdx.x + 1] -
            gpu_input_offset_l[blockIdx.x];  // 75
  int col = gpu_input_offset_r[blockIdx.x + 1] -
            gpu_input_offset_r[blockIdx.x];  // 300

  int max_k = topks[topk_size - 1];
  max_k = max_k < col ? max_k : col;

  extern __shared__ Dtype smem[];  // H1*W or H2*W ...

  int filled_z = row / actual_row_in_shared_mem;
  int remain_row = row - filled_z * actual_row_in_shared_mem;

  if (blockIdx.z > filled_z || (blockIdx.z == filled_z && remain_row == 0)) {
    return;
  }

  const Dtype *fm_row_in_data = input_data +
                                blockIdx.x * row_max * feat_map_num * col_max +
                                blockIdx.y * row_max * col_max +
                                blockIdx.z * actual_row_in_shared_mem * col_max;
  if (blockIdx.z == filled_z) {
    for (int i = threadIdx.x; i < remain_row * col_max; i += blockDim.x) {
      smem[i] = fm_row_in_data[i];
    }
  } else {
    for (int i = threadIdx.x; i < actual_row_in_shared_mem * col_max;
         i += blockDim.x) {
      smem[i] = fm_row_in_data[i];
    }
  }
  __syncthreads();

  int cur_row;
  if (blockIdx.z == filled_z) {
    cur_row = remain_row;
  } else {
    cur_row = actual_row_in_shared_mem;
  }

  for (int idx = threadIdx.x; idx < cur_row; idx += blockDim.x) {
    Dtype *fm_row_out_data = output_data +
                             (gpu_input_offset_l[blockIdx.x] +
                              blockIdx.z * actual_row_in_shared_mem + idx) *
                                 feat_map_num * topk_size +
                             blockIdx.y * topk_size;
    for (int i = 0; i < topk_size; ++i) {
      fm_row_out_data[i] = 0;
    }

    Dtype *smem_start_col = smem + idx * col_max;

    int counter = max_k;  // topk_size;
    Dtype last_max_val = -20000.0;
    while (counter) {
      Dtype max_val = -10000.0;
      int max_pos = 0;  // -1;
      int m = 0;
      for (; m < col; m++) {
        Dtype cur_data = smem_start_col[m];
        if (cur_data > max_val) {
          max_val = cur_data;
          max_pos = m;
          last_max_val = max_val;
        }
      }
      if (max_val < -9999.0) {  // == -10000.0
        max_val = last_max_val;
      }
      smem_start_col[max_pos] = -10000000.0;

      int i = max_k - counter;
      for (int c = 0; c < topk_size; c++) {
        if (i <= topks[c] - 1) {
          fm_row_out_data[c] += max_val;
        }
      }
      counter--;
    }
    __syncthreads();
    // compute avg
    for (int i = 0; i < topk_size; i++) {
      fm_row_out_data[i] = fm_row_out_data[i] / topks[i];
    }
  }
}

template <typename T>
void SequenceTopkAvgPoolingCompute<T>::PrepareForRun() {
  int device_id;
  cudaGetDevice(&device_id);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device_id);
  _shared_mem_size = deviceProp.sharedMemPerBlock;
}

template <typename T>
void SequenceTopkAvgPoolingCompute<T>::Run() {
  auto &param = this->Param<param_t>();
  auto &ctx = this->ctx_->template As<CUDAContext>();
  auto cuda_stream = ctx.exec_stream();

  CHECK(param.X->lod().size() > 0 && param.X->lod()[0].size() > 0)
      << "X sequence offset is not valid";
  CHECK(param.ROW->lod().size() > 0 && param.ROW->lod()[0].size() > 0)
      << "ROW sequence offset is not valid";

  int width_offset_len = param.X->lod()[0].size();
  lite::DDim width_offset_shape(std::vector<int64_t>{width_offset_len});
  _width_offset.Resize(width_offset_shape);
  std::vector<int> width_lod_0(width_offset_len, 0);
  for (size_t i = 0; i < param.X->lod()[0].size(); ++i) {
    width_lod_0[i] = static_cast<int>(param.X->lod()[0][i]);
  }
  cudaMemcpyAsync(_width_offset.mutable_data<int>(TARGET(kCUDA)),
                  &width_lod_0[0],
                  sizeof(int) * width_offset_len,
                  cudaMemcpyHostToDevice,
                  cuda_stream);

  int height_offset_len = param.ROW->lod()[0].size();
  lite::DDim height_offset_shape(std::vector<int64_t>{height_offset_len});
  _height_offset.Resize(height_offset_shape);
  std::vector<int> height_lod_0(height_offset_len, 0);
  for (size_t i = 0; i < param.ROW->lod()[0].size(); ++i) {
    height_lod_0[i] = static_cast<int>(param.ROW->lod()[0][i]);
  }
  cudaMemcpyAsync(_height_offset.mutable_data<int>(TARGET(kCUDA)),
                  &height_lod_0[0],
                  sizeof(int) * height_offset_len,
                  cudaMemcpyHostToDevice,
                  cuda_stream);

  const Tensor *x_tensor = param.X;
  Tensor *out_tensor = param.Out;
  const T *in_data = x_tensor->data<T>();
  T *out_data = out_tensor->mutable_data<T>(TARGET(kCUDA));

  int topk_num = param.topks.size();
  lite::DDim top_ks_shape(std::vector<int64_t>{topk_num, 1, 1, 1});
  _top_ks.Resize(top_ks_shape);
  cudaMemcpyAsync(_top_ks.mutable_data<int>(TARGET(kCUDA)),
                  &param.topks[0],
                  sizeof(int) * topk_num,
                  cudaMemcpyHostToDevice,
                  cuda_stream);

  int num = param.X->dims()[0];
  int channel = param.X->dims()[1];
  int height = param.X->dims()[2];
  int width = param.X->dims()[3];

  const int *height_offset = _height_offset.data<int>();
  const int *width_offset = _width_offset.data<int>();

  int feat_map_size = height * width;

  if (feat_map_size * sizeof(T) <= _shared_mem_size) {
    dim3 blocks(num, channel);
    dim3 threads(32, 1);

    topk_avg_pooling_kernel_by_row_improve<
        T><<<blocks, threads, feat_map_size * sizeof(T), cuda_stream>>>(
        out_data,
        in_data,
        height_offset,
        width_offset,
        height,
        width,
        param.topks.size(),
        _top_ks.data<int>(),
        param.channel_num);
  } else {
    int actual_row = _shared_mem_size / width / sizeof(T);
    int num_z = (height + actual_row - 1) / actual_row;
    dim3 blocks(num, channel, num_z);
    dim3 threads(32, 1);

    topk_avg_pooling_kernel_for_big_data<
        T><<<blocks, threads, actual_row * width * sizeof(T), cuda_stream>>>(
        out_data,
        in_data,
        height_offset,
        width_offset,
        height,
        width,
        param.topks.size(),
        _top_ks.data<int>(),
        param.channel_num,
        actual_row);
  }
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    sequence_topk_avg_pooling,
    kCUDA,
    kFloat,
    kNCHW,
    paddle::lite::kernels::cuda::SequenceTopkAvgPoolingCompute<float>,
    def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindInput("ROW",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindInput("COLUMN",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .BindOutput("pos",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .Finalize();
