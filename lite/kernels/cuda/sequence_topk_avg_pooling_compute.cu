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

  const Dtype *fm_row_in_data = input;
  for (int i = 0; i < blockIdx.x; ++i) {
    int tmp_row = gpu_input_offset_l[i + 1] - gpu_input_offset_l[i];
    int tmp_col = gpu_input_offset_r[i + 1] - gpu_input_offset_r[i];
    fm_row_in_data += tmp_row * feat_map_num * tmp_col;
  }
  fm_row_in_data += blockIdx.y * row * col;

  for (int i = threadIdx.x; i < row * col; i += blockDim.x) {
    smem[i] = fm_row_in_data[i];
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < row; idx += blockDim.x) {
    Dtype *fm_row_out_data =
        output_data +
        (gpu_input_offset_l[blockIdx.x] + idx) * feat_map_num * topk_size +
        blockIdx.y * topk_size;

    Dtype *smem_start_col = smem + idx * col;

    int counter = max_k;  // topk_size;
    Dtype last_max_val = -20000.0;
    while (counter) {
      Dtype max_val = -10000.0;
      int max_pos = 0;
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
void SequenceTopkAvgPoolingCompute<T>::Run() {
  auto &param = this->Param<param_t>();
  auto &ctx = this->ctx_->template As<CUDAContext>();
  auto cuda_stream = ctx.exec_stream();
  int topk_num = param.topks.size();
  lite::DDim top_ks_shape(std::vector<int64_t>{topk_num, 1, 1, 1});
  _top_ks.Resize(top_ks_shape);
  cudaMemcpyAsync(_top_ks.mutable_data<int>(TARGET(kCUDA)),
                  &param.topks[0],
                  sizeof(int) * topk_num,
                  cudaMemcpyHostToDevice,
                  cuda_stream);

  int width_offset_len = param.COLUMN->lod()[0].size();
  lite::DDim width_offset_shape(
      std::vector<int64_t>{width_offset_len, 1, 1, 1});
  _width_offset.Resize(width_offset_shape);
  std::vector<int> width_lod_0(width_offset_len, 0);
  for (size_t i = 0; i < param.COLUMN->lod()[0].size(); ++i) {
    width_lod_0[i] = static_cast<int>(param.COLUMN->lod()[0][i]);
  }
  cudaMemcpyAsync(_width_offset.mutable_data<int>(TARGET(kCUDA)),
                  &width_lod_0[0],
                  sizeof(int) * width_offset_len,
                  cudaMemcpyHostToDevice,
                  cuda_stream);

  int height_offset_len = param.ROW->lod()[0].size();
  lite::DDim height_offset_shape(
      std::vector<int64_t>{height_offset_len, 1, 1, 1});
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
  TargetWrapperCuda::MemsetAsync(out_tensor->mutable_data<T>(TARGET(kCUDA)),
                                 0,
                                 sizeof(T) * out_tensor->numel(),
                                 cuda_stream);

  int num = param.ROW->lod()[0].size() - 1;
  int channel = param.channel_num;

  const int *height_offset = _height_offset.data<int>();
  const int *width_offset = _width_offset.data<int>();

  int feat_map_size = 0;
  for (size_t i = 0; i < height_lod_0.size() - 1; ++i) {
    int height = height_lod_0[i + 1] - height_lod_0[i];
    int width = width_lod_0[i + 1] - width_lod_0[i];
    if (height * width > feat_map_size) {
      feat_map_size = height * width;
    }
  }
  dim3 blocks(num, channel);
  dim3 threads(32, 1);
  topk_avg_pooling_kernel_by_row_improve<
      T><<<blocks, threads, feat_map_size * sizeof(T), cuda_stream>>>(
      out_data,
      in_data,
      height_offset,
      width_offset,
      param.topks.size(),
      _top_ks.data<int>(),
      param.channel_num);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) LOG(ERROR) << cudaGetErrorString(error);
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
