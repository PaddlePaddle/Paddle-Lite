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
#include <algorithm>
#include <vector>
#include "lite/core/op_registry.h"
#include "lite/kernels/cuda/concat_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {
using Tensor = lite::Tensor;

template <typename T>
__global__ void ConcatKernel(const T** inputs,
                             const int* input_cols,
                             int col_size,
                             const int output_rows,
                             const int output_cols,
                             T* output) {
  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  int curr_segment = 0;
  int curr_offset = input_cols[0];
  for (; tid_x < output_cols; tid_x += blockDim.x * gridDim.x) {
    int curr_col_offset = input_cols[curr_segment + 1];
    while (curr_col_offset <= tid_x) {
      curr_offset = curr_col_offset;
      ++curr_segment;
      curr_col_offset = input_cols[curr_segment + 1];
    }

    int local_col = tid_x - curr_offset;
    int segment_width = curr_col_offset - curr_offset;

    const T* input_ptr = inputs[curr_segment];
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    for (; tid_y < output_rows; tid_y += blockDim.y * gridDim.y)
      output[tid_y * output_cols + tid_x] =
          input_ptr[tid_y * segment_width + local_col];
  }
}

template <typename T>
__device__ void ConcatKernelDetail(const T** inputs_data,
                                   const int fixed_in_col,
                                   const int out_rows,
                                   const int out_cols,
                                   T* output_data) {
  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  for (; tid_x < out_cols; tid_x += blockDim.x * gridDim.x) {
    int split = tid_x * 1.0 / fixed_in_col;
    int in_offset = tid_x - split * fixed_in_col;
    const T* input_ptr = inputs_data[split];
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    for (; tid_y < out_rows; tid_y += blockDim.y * gridDim.y) {
      output_data[tid_y * out_cols + tid_x] =
          input_ptr[tid_y * fixed_in_col + in_offset];
    }
  }
  // for (int i = 0; i < 4; i++){
  //   printf("input[0][%d] = %.1f\n", i, inputs_data[0][i]);
  //   printf("output[%d] = %.1f\n", i, output_data[i]);
  // }
}

template <typename T>
__global__ void ConcatKernel(const T* input_addr0,
                             const T* input_addr1,
                             const int fixed_in_col,
                             const int out_rows,
                             const int out_cols,
                             T* output_data) {
  const T* inputs_data[2];
  inputs_data[0] = input_addr0;
  inputs_data[1] = input_addr1;
  ConcatKernelDetail<T>(
      inputs_data, fixed_in_col, out_rows, out_cols, output_data);
}

template <typename T>
__global__ void ConcatKernel(const T* input_addr0,
                             const T* input_addr1,
                             const T* input_addr2,
                             const int fixed_in_col,
                             const int out_rows,
                             const int out_cols,
                             T* output_data) {
  const T* inputs_data[3];
  inputs_data[0] = input_addr0;
  inputs_data[1] = input_addr1;
  inputs_data[2] = input_addr2;
  ConcatKernelDetail<T>(
      inputs_data, fixed_in_col, out_rows, out_cols, output_data);
}

template <typename T>
__global__ void ConcatKernel(const T* input_addr0,
                             const T* input_addr1,
                             const T* input_addr2,
                             const T* input_addr3,
                             const int fixed_in_col,
                             const int out_rows,
                             const int out_cols,
                             T* output_data) {
  const T* inputs_data[4];
  inputs_data[0] = input_addr0;
  inputs_data[1] = input_addr1;
  inputs_data[2] = input_addr2;
  inputs_data[3] = input_addr3;
  ConcatKernelDetail<T>(
      inputs_data, fixed_in_col, out_rows, out_cols, output_data);
}

template <typename T>
__global__ void ConcatKernel(const T** inputs_data,
                             const int in_num,
                             const int fixed_in_col,
                             const int out_rows,
                             const int out_cols,
                             T* output_data) {
  ConcatKernelDetail<T>(
      inputs_data, fixed_in_col, out_rows, out_cols, output_data);
}

static inline void GetBlockDims(const CUDAContext& context,
                                int num_rows,
                                int num_cols,
                                dim3* block_dims,
                                dim3* grid_dims) {
  // Set the thread block and grid according to CurrentDeviceId
  const int kThreadsPerBlock = 1024;
  int block_cols = kThreadsPerBlock;
  if (num_cols < kThreadsPerBlock) {  // block_cols is aligned by 32.
    block_cols = ((num_cols + 31) >> 5) << 5;
  }
  int block_rows = kThreadsPerBlock / block_cols;
  *block_dims = dim3(block_cols, block_rows, 1);

  int grid_cols = (num_cols + block_cols - 1) / block_cols;
  int grid_rows = std::max(num_rows / block_rows, 1);
  *grid_dims = dim3(grid_cols, grid_rows, 1);
}

void ConcatCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<CUDAContext>();
  auto stream = ctx.exec_stream();

  std::vector<Tensor*> input = param.x;
  Tensor* output = param.output;
  int axis = param.axis;

  int in_num = input.size();
  int in_row = 1;
  auto dim_0 = input[0]->dims();
  for (int i = 0; i < axis; ++i) {
    in_row *= dim_0[i];
  }
  int in_col = input[0]->numel() / in_row;
  int out_row = in_row, out_col = 0;

  std::vector<const float*> inputs_data(in_num);
  std::vector<int> inputs_col(in_num + 1);
  inputs_col[0] = 0;
  bool has_same_shape = true;
  for (int i = 0; i < in_num; ++i) {
    int t_cols = input[i]->numel() / in_row;
    if (has_same_shape) {
      if (t_cols != in_col) has_same_shape = false;
    }
    out_col += t_cols;
    inputs_col[i + 1] = out_col;
    inputs_data[i] = input[i]->data<float>();
  }
  dim3 block_dims;
  dim3 grid_dims;
  GetBlockDims(ctx, out_row, out_col, &block_dims, &grid_dims);
  const float** dev_ins_data = nullptr;
  if (!has_same_shape || in_num < 2 || in_num > 4) {
    float* tmp_dev_ins_data = nullptr;
    CHECK(cudaSuccess ==
          cudaMalloc(&tmp_dev_ins_data, inputs_data.size() * sizeof(float*)));
    CHECK(cudaSuccess == cudaMemcpy(tmp_dev_ins_data,
                                    static_cast<void*>(inputs_data.data()),
                                    inputs_data.size() * sizeof(float*),
                                    cudaMemcpyHostToDevice));
    dev_ins_data = reinterpret_cast<const float**>(tmp_dev_ins_data);
  }
  if (has_same_shape) {
    if (in_num == 2) {
      ConcatKernel<float><<<grid_dims, block_dims, 0, stream>>>(
          inputs_data[0],
          inputs_data[1],
          in_col,
          out_row,
          out_col,
          output->mutable_data<float>());
    } else if (in_num == 3) {
      ConcatKernel<float><<<grid_dims, block_dims, 0, stream>>>(
          inputs_data[0],
          inputs_data[1],
          inputs_data[2],
          in_col,
          out_row,
          out_col,
          output->mutable_data<float>());
    } else if (in_num == 4) {
      ConcatKernel<float><<<grid_dims, block_dims, 0, stream>>>(
          inputs_data[0],
          inputs_data[1],
          inputs_data[2],
          inputs_data[3],
          in_col,
          out_row,
          out_col,
          output->mutable_data<float>());
    } else {
      ConcatKernel<float><<<grid_dims, block_dims, 0, stream>>>(
          dev_ins_data,
          in_num,
          in_col,
          out_row,
          out_col,
          output->mutable_data<float>());
      cudaFree(dev_ins_data);
    }
  } else {
    int* tmp_dev_ins_col_data = nullptr;

    CHECK(cudaSuccess ==
          cudaMalloc(&tmp_dev_ins_col_data, inputs_col.size() * sizeof(int)));
    CHECK(cudaSuccess == cudaMemcpy(tmp_dev_ins_col_data,
                                    static_cast<void*>(inputs_col.data()),
                                    inputs_col.size() * sizeof(int),
                                    cudaMemcpyHostToDevice));
    int* dev_ins_col_data = static_cast<int*>(tmp_dev_ins_col_data);
    ConcatKernel<float><<<grid_dims, block_dims, 0, stream>>>(
        dev_ins_data,
        dev_ins_col_data,
        static_cast<int>(inputs_col.size()),
        out_row,
        out_col,
        output->mutable_data<float>());
    cudaFree(dev_ins_data);
    cudaFree(dev_ins_col_data);
  }

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) LOG(INFO) << cudaGetErrorString(error);
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(concat,
                     kCUDA,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::cuda::ConcatCompute,
                     def)
    .BindInput("x", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindInput("axis", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("output", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .Finalize();
