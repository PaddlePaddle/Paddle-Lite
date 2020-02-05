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
#include "lite/core/op_registry.h"
#include "lite/core/target_wrapper.h"
#include "lite/kernels/cuda/sequence_arithmetic_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

const int CUDA_NUM_THREADS = 512;

#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

inline int CUDA_GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

template <typename Dtype>
__global__ void ker_arithmetic_sum(Dtype* out_data,
                                   const Dtype* in_data_0,
                                   const Dtype* in_data_1,
                                   const int* offset_0,
                                   const int* offset_1,
                                   const int* word_id_to_seq_id,
                                   const int seq_num,
                                   const int inner_size,
                                   const int count) {
  CUDA_KERNEL_LOOP(tid, count) {
    int emb_id = tid % inner_size;
    int word_id = tid / inner_size;
    int seq_id = word_id_to_seq_id[word_id];
    int word_id_in_cur_seq = word_id - offset_0[seq_id];
    int seq_len_1 = offset_1[seq_id + 1] - offset_1[seq_id];
    if (word_id_in_cur_seq < seq_len_1) {
      out_data[tid] =
          in_data_0[tid] +
          in_data_1[(offset_1[seq_id] + word_id_in_cur_seq) * inner_size +
                    emb_id];
    } else {
      out_data[tid] = in_data_0[tid];
    }
  }
}

template <typename Dtype>
__global__ void ker_arithmetic_sub(Dtype* out_data,
                                   const Dtype* in_data_0,
                                   const Dtype* in_data_1,
                                   const int* offset_0,
                                   const int* offset_1,
                                   const int* word_id_to_seq_id,
                                   const int seq_num,
                                   const int inner_size,
                                   const int count) {
  CUDA_KERNEL_LOOP(tid, count) {
    int emb_id = tid % inner_size;
    int word_id = tid / inner_size;
    int seq_id = word_id_to_seq_id[word_id];
    int word_id_in_cur_seq = word_id - offset_0[seq_id];
    int seq_len_1 = offset_1[seq_id + 1] - offset_1[seq_id];
    if (word_id_in_cur_seq < seq_len_1) {
      out_data[tid] =
          in_data_0[tid] -
          in_data_1[(offset_1[seq_id] + word_id_in_cur_seq) * inner_size +
                    emb_id];
    } else {
      out_data[tid] = in_data_0[tid];
    }
  }
}

template <typename Dtype>
__global__ void ker_arithmetic_mul(Dtype* out_data,
                                   const Dtype* in_data_0,
                                   const Dtype* in_data_1,
                                   const int* offset_0,
                                   const int* offset_1,
                                   const int* word_id_to_seq_id,
                                   const int seq_num,
                                   const int inner_size,
                                   const int count) {
  CUDA_KERNEL_LOOP(tid, count) {
    int emb_id = tid % inner_size;
    int word_id = tid / inner_size;
    int seq_id = word_id_to_seq_id[word_id];
    int word_id_in_cur_seq = word_id - offset_0[seq_id];
    int seq_len_1 = offset_1[seq_id + 1] - offset_1[seq_id];
    if (word_id_in_cur_seq < seq_len_1) {
      out_data[tid] =
          in_data_0[tid] *
          in_data_1[(offset_1[seq_id] + word_id_in_cur_seq) * inner_size +
                    emb_id];
    } else {
      out_data[tid] = in_data_0[tid];
    }
  }
}

void SequenceArithmeticCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<CUDAContext>();
  auto stream = ctx.exec_stream();

  auto x_data = param.X->data<float>();
  auto x_lod = param.X->lod()[0];
  auto y_data = param.Y->data<float>();
  auto y_lod = param.Y->lod()[0];
  auto out_data = param.Out->mutable_data<float>(TARGET(kCUDA));

  offset_x.Resize({static_cast<int64_t>(x_lod.size())});
  auto offset_x_data = offset_x.mutable_data<int>(TARGET(kCUDA));

  offset_y.Resize({static_cast<int64_t>(y_lod.size())});
  auto offset_y_data = offset_y.mutable_data<int>(TARGET(kCUDA));

  word_id_to_seq_id.Resize({param.X->numel()});
  auto word_id_to_seq_id_data =
      word_id_to_seq_id.mutable_data<int>(TARGET(kCUDA));

  std::vector<int> word_seq_map;
  for (int i = 0; i < x_lod.size() - 1; i++) {
    for (int j = x_lod[i]; j < x_lod[i + 1]; j++) {
      word_seq_map.push_back(i);
    }
  }

  std::vector<int> offset_x_data_cpu(x_lod.size(), 0);
  auto x_lod_data = x_lod.data();
  for (int i = 0; i < offset_x_data_cpu.size(); i++) {
    offset_x_data_cpu[i] = x_lod_data[i];
  }

  std::vector<int> offset_y_data_cpu(y_lod.size(), 0);
  auto y_lod_data = y_lod.data();
  for (int i = 0; i < offset_y_data_cpu.size(); i++) {
    offset_y_data_cpu[i] = y_lod_data[i];
  }

  TargetWrapperCuda::MemcpyAsync(offset_x_data,
                                 offset_x_data_cpu.data(),
                                 sizeof(int) * x_lod.size(),
                                 IoDirection::HtoD,
                                 stream);

  TargetWrapperCuda::MemcpyAsync(offset_y_data,
                                 offset_y_data_cpu.data(),
                                 sizeof(int) * y_lod.size(),
                                 IoDirection::HtoD,
                                 stream);

  TargetWrapperCuda::MemcpyAsync(word_id_to_seq_id_data,
                                 word_seq_map.data(),
                                 sizeof(int) * word_seq_map.size(),
                                 IoDirection::HtoD,
                                 stream);

  int seq_num = x_lod.size() - 1;
  int count = param.X->numel();
  int inner_size = param.X->dims()[1];
  switch (param.op_type) {
    case 1:  // sum
      ker_arithmetic_sum<
          float><<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(
          out_data,
          x_data,
          y_data,
          offset_x_data,
          offset_y_data,
          word_id_to_seq_id_data,
          seq_num,
          inner_size,
          count);
      break;
    case 2:  // sub
      ker_arithmetic_sub<
          float><<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(
          out_data,
          x_data,
          y_data,
          offset_x_data,
          offset_y_data,
          word_id_to_seq_id_data,
          seq_num,
          inner_size,
          count);
      break;
    case 3:  // mul
      ker_arithmetic_mul<
          float><<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(
          out_data,
          x_data,
          y_data,
          offset_x_data,
          offset_y_data,
          word_id_to_seq_id_data,
          seq_num,
          inner_size,
          count);
      break;
    default:
      break;
  }

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) LOG(INFO) << cudaGetErrorString(error);
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(sequence_arithmetic,
                     kCUDA,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::cuda::SequenceArithmeticCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .Finalize();
REGISTER_LITE_KERNEL(search_seq_arithmetic,
                     kCUDA,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::cuda::SequenceArithmeticCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .Finalize();
