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
#include "lite/kernels/cuda/sequence_concat_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

const int CUDA_NUM_THREADS = 512;

template <typename T>
inline LoD ConcatLoD(const std::vector<lite::Tensor*>& xs) {
  std::vector<size_t> result;
  result.resize(xs[0]->lod()[0].size());

  for (size_t i = 1; i < result.size(); ++i) {
    size_t sum = 0;
    for (size_t j = 0; j < xs.size(); ++j) {
      auto& x_lod = xs[j]->lod()[0];
      sum += x_lod[i];
    }
    result[i] = sum;
  }
  LoD lod;
  lod.emplace_back(result);
  return lod;
}

template <typename Dtype>
__global__ void ker_sequence_concat(Dtype* out_data,
                                    const uint64_t* in_locate_data,
                                    const int* o2i_map,
                                    const int* o2i_w_map,
                                    const int seq_num,
                                    const int emb_size,
                                    const int count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int tid = idx; tid < count; tid += blockDim.x * gridDim.x) {
    int emb_id = tid % emb_size;
    int word_id = tid / emb_size;
    int input_id = o2i_map[word_id];
    int cur_work_id = o2i_w_map[word_id];
    const Dtype* in_data = reinterpret_cast<const Dtype*>(
        reinterpret_cast<uintptr_t>(in_locate_data[input_id]));
    out_data[tid] = in_data[cur_work_id * emb_size + emb_id];
  }
}

void SequenceConcatCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<CUDAContext>();
  auto stream = ctx.exec_stream();
  float* out_data = param.Out->mutable_data<float>(TARGET(kCUDA));

  int seq_num = param.X[0]->lod()[0].size() - 1;
  const int emb_size = param.X[0]->numel() / param.X[0]->dims()[0];
  std::vector<uint64_t> in_locate_vec;
  for (size_t i = 0; i < param.X.size(); ++i) {
    in_locate_vec.push_back(
        reinterpret_cast<uintptr_t>(param.X[i]->data<float>()));
  }
  in_locate_tensor.Resize({static_cast<int64_t>(in_locate_vec.size())});

  std::vector<int> out2in_map;
  std::vector<int> out2in_word_map;
  for (int i = 0; i < seq_num; ++i) {
    for (int j = 0; j < param.X.size(); ++j) {
      auto offset = param.X[j]->lod()[0];
      int cur_len = offset[i + 1] - offset[i];
      for (int k = 0; k < cur_len; ++k) {
        out2in_map.push_back(j);
        out2in_word_map.push_back(offset[i] + k);
      }
    }
  }
  int word_num = out2in_map.size();
  out2in_map_tensor.Resize({word_num});
  out2in_word_map_tensor.Resize({word_num});
  int* gpu_o2i_map_data = out2in_map_tensor.mutable_data<int>(TARGET(kCUDA));
  int* gpu_o2i_w_map_data =
      out2in_word_map_tensor.mutable_data<int>(TARGET(kCUDA));
  uint64_t* gpu_in_locate_data =
      in_locate_tensor.mutable_data<uint64_t>(TARGET(kCUDA));

  TargetWrapperCuda::MemcpyAsync(gpu_o2i_map_data,
                                 out2in_map.data(),
                                 sizeof(int) * out2in_map.size(),
                                 IoDirection::HtoD,
                                 stream);
  TargetWrapperCuda::MemcpyAsync(gpu_o2i_w_map_data,
                                 out2in_word_map.data(),
                                 sizeof(int) * out2in_word_map.size(),
                                 IoDirection::HtoD,
                                 stream);
  TargetWrapperCuda::MemcpyAsync(gpu_in_locate_data,
                                 in_locate_vec.data(),
                                 sizeof(uint64_t) * in_locate_vec.size(),
                                 IoDirection::HtoD,
                                 stream);

  param.Out->set_lod(ConcatLoD<float>(param.X));

  int count = param.X[0]->numel();
  for (int i = 1; i < param.X.size(); ++i) {
    count += param.X[i]->numel();
  }

  int blocks = (count + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
  ker_sequence_concat<float><<<blocks, CUDA_NUM_THREADS, 0, stream>>>(
      out_data,
      gpu_in_locate_data,
      gpu_o2i_map_data,
      gpu_o2i_w_map_data,
      seq_num,
      emb_size,
      count);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) LOG(INFO) << cudaGetErrorString(error);
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(sequence_concat,
                     kCUDA,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::cuda::SequenceConcatCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .Finalize();
