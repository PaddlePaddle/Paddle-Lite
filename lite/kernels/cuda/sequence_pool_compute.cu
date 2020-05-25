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

#include <vector>
#include "lite/backends/cuda/cuda_utils.h"
#include "lite/core/op_registry.h"
#include "lite/core/target_wrapper.h"
#include "lite/kernels/cuda/sequence_pool_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

template <typename Dtype>
__global__ void seq_pool_average_kernel(Dtype* dst,
                                        const Dtype* src_in,
                                        const int batch_size,
                                        const uint64_t* seq_offset,
                                        const int slice_size) {
  int total = slice_size * batch_size;
  CUDA_KERNEL_LOOP(tid, total) {
    int out_batch_id = tid / slice_size;
    int out_id = tid % slice_size;
    int in_slice_num = static_cast<int>(seq_offset[out_batch_id + 1] -
                                        seq_offset[out_batch_id]);
    int in_offset = static_cast<int>(seq_offset[out_batch_id] * slice_size);
    src_in += in_offset + out_id;
    Dtype sum = (Dtype)0;
    for (int i = 0; i < in_slice_num; ++i) {
      sum += src_in[i * slice_size];
    }
    dst[out_batch_id * slice_size + out_id] = sum / in_slice_num;
  }
}

template <typename Dtype>
__global__ void seq_pool_sum_kernel(Dtype* dst,
                                    const Dtype* src_in,
                                    const int batch_size,
                                    const uint64_t* seq_offset,
                                    const int slice_size) {
  int total = slice_size * batch_size;
  CUDA_KERNEL_LOOP(tid, total) {
    int out_batch_id = tid / slice_size;
    int out_id = tid % slice_size;
    int in_slice_num = static_cast<int>(seq_offset[out_batch_id + 1] -
                                        seq_offset[out_batch_id]);
    int in_offset = static_cast<int>(seq_offset[out_batch_id] * slice_size);
    src_in += in_offset + out_id;
    Dtype sum = (Dtype)0;
    for (int i = 0; i < in_slice_num; ++i) {
      sum += src_in[i * slice_size];
    }
    dst[out_batch_id * slice_size + out_id] = sum;
  }
}

template <typename Dtype>
__global__ void seq_pool_sqrt_kernel(Dtype* dst,
                                     const Dtype* src_in,
                                     const int batch_size,
                                     const uint64_t* seq_offset,
                                     const int slice_size) {
  int total = slice_size * batch_size;
  CUDA_KERNEL_LOOP(tid, total) {
    int out_batch_id = tid / slice_size;
    int out_id = tid % slice_size;
    int in_slice_num = static_cast<int>(seq_offset[out_batch_id + 1] -
                                        seq_offset[out_batch_id]);
    int in_offset = static_cast<int>(seq_offset[out_batch_id] * slice_size);
    src_in += in_offset + out_id;
    Dtype sum = (Dtype)0;
    for (int i = 0; i < in_slice_num; ++i) {
      sum += src_in[i * slice_size];
    }
    dst[out_batch_id * slice_size + out_id] = sum * rsqrtf(in_slice_num);
  }
}

template <typename Dtype>
__global__ void seq_pool_max_kernel(Dtype* dst,
                                    const Dtype* src_in,
                                    const int batch_size,
                                    const uint64_t* seq_offset,
                                    const int slice_size) {
  int total = slice_size * batch_size;
  CUDA_KERNEL_LOOP(tid, total) {
    int out_batch_id = tid / slice_size;
    int out_id = tid % slice_size;
    int in_slice_num = static_cast<int>(seq_offset[out_batch_id + 1] -
                                        seq_offset[out_batch_id]);
    int in_offset = static_cast<int>(seq_offset[out_batch_id] * slice_size);
    src_in += in_offset + out_id;
    Dtype max = src_in[0];
    for (int i = 1; i < in_slice_num; ++i) {
      Dtype val = src_in[i * slice_size];
      if (val > max) {
        max = val;
      }
    }
    dst[out_batch_id * slice_size + out_id] = max;
  }
}

template <typename Dtype>
__global__ void seq_pool_last_kernel(Dtype* dst,
                                     const Dtype* src_in,
                                     const int batch_size,
                                     const uint64_t* seq_offset,
                                     const int slice_size) {
  int total = slice_size * batch_size;
  CUDA_KERNEL_LOOP(tid, total) {
    int out_batch_id = tid / slice_size;
    int out_id = tid % slice_size;
    int in_offset =
        (static_cast<int>(seq_offset[out_batch_id + 1]) - 1) * slice_size;
    dst[tid] = src_in[in_offset + out_id];
  }
}

template <typename Dtype>
__global__ void seq_pool_first_kernel(Dtype* dst,
                                      const Dtype* src_in,
                                      const int batch_size,
                                      const uint64_t* seq_offset,
                                      const int slice_size) {
  int total = slice_size * batch_size;
  CUDA_KERNEL_LOOP(tid, total) {
    int out_batch_id = tid / slice_size;
    int out_id = tid % slice_size;
    int in_offset = static_cast<int>(seq_offset[out_batch_id] * slice_size);
    dst[tid] = src_in[in_offset + out_id];
  }
}

void SequencePoolCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<CUDAContext>();
  auto stream = ctx.exec_stream();

  std::vector<uint64_t> seq_offset = param.X->lod()[0];
  int batch_size = param.X->lod()[0].size() - 1;
  int slice_size = param.Out->dims().production() / batch_size;

  float* out_data = param.Out->mutable_data<float>(TARGET(kCUDA));
  const float* in_data = param.X->data<float>();

  seq_offset_D.Resize({static_cast<int64_t>(seq_offset.size())});
  TargetWrapperCuda::MemcpyAsync(
      seq_offset_D.mutable_data<uint64_t>(TARGET(kCUDA)),
      seq_offset.data(),
      sizeof(uint64_t) * seq_offset.size(),
      IoDirection::HtoD,
      stream);

  if (param.pool_type == "MAX") {
    seq_pool_max_kernel<float><<<CUDA_GET_BLOCKS(batch_size * slice_size),
                                 CUDA_NUM_THREADS,
                                 0,
                                 stream>>>(out_data,
                                           in_data,
                                           batch_size,
                                           seq_offset_D.data<uint64_t>(),
                                           slice_size);
  } else if (param.pool_type == "AVERAGE") {
    seq_pool_average_kernel<float><<<CUDA_GET_BLOCKS(batch_size * slice_size),
                                     CUDA_NUM_THREADS,
                                     0,
                                     stream>>>(out_data,
                                               in_data,
                                               batch_size,
                                               seq_offset_D.data<uint64_t>(),
                                               slice_size);
  } else if (param.pool_type == "SUM") {
    seq_pool_sum_kernel<float><<<CUDA_GET_BLOCKS(batch_size * slice_size),
                                 CUDA_NUM_THREADS,
                                 0,
                                 stream>>>(out_data,
                                           in_data,
                                           batch_size,
                                           seq_offset_D.data<uint64_t>(),
                                           slice_size);
  } else if (param.pool_type == "SQRT") {
    seq_pool_sqrt_kernel<float><<<CUDA_GET_BLOCKS(batch_size * slice_size),
                                  CUDA_NUM_THREADS,
                                  0,
                                  stream>>>(out_data,
                                            in_data,
                                            batch_size,
                                            seq_offset_D.data<uint64_t>(),
                                            slice_size);
  } else if (param.pool_type == "FIRST") {
    seq_pool_first_kernel<float><<<CUDA_GET_BLOCKS(batch_size * slice_size),
                                   CUDA_NUM_THREADS,
                                   0,
                                   stream>>>(out_data,
                                             in_data,
                                             batch_size,
                                             seq_offset_D.data<uint64_t>(),
                                             slice_size);
  } else if (param.pool_type == "LAST") {
    seq_pool_last_kernel<float><<<CUDA_GET_BLOCKS(batch_size * slice_size),
                                  CUDA_NUM_THREADS,
                                  0,
                                  stream>>>(out_data,
                                            in_data,
                                            batch_size,
                                            seq_offset_D.data<uint64_t>(),
                                            slice_size);
  } else {
    LOG(ERROR) << "pool type " << param.pool_type << " is not supoorted.";
  }

  std::vector<uint64_t> offset_new(static_cast<uint64_t>(batch_size + 1));

  for (int i = 0; i <= batch_size; ++i) {
    offset_new[i] = i;
  }
  std::vector<std::vector<uint64_t>> voffset_new;
  voffset_new.push_back(offset_new);
  param.Out->set_lod(voffset_new);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) LOG(INFO) << cudaGetErrorString(error);
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(sequence_pool,
                     kCUDA,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::cuda::SequencePoolCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("MaxIndex", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .Finalize();
