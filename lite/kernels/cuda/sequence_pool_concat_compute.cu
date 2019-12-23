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
#include "lite/kernels/cuda/sequence_pool_concat_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

template <typename Dtype>
__global__ void sequence_pool_concat(const uint64_t* input_locate_data,
                                     const int* pool_type_list,
                                     Dtype* output_data,
                                     const int* offset,
                                     int batch,
                                     int in_num,
                                     int in_dim) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int em_id = tid % in_dim;
  int in_id = (tid / in_dim) % in_num;
  int seq_id = tid / (in_dim * in_num);

  if (seq_id >= batch) {
    return;
  }
  Dtype* out_data = output_data + tid;
  int offset_id = in_id * (batch + 1) + seq_id;
  if (pool_type_list[in_id] == 4) {  // last
    const Dtype* in_data =
        reinterpret_cast<const Dtype*>(
            reinterpret_cast<uintptr_t>(input_locate_data[in_id])) +
        em_id;
    output_data[tid] = in_data[(offset[offset_id + 1] - 1) * in_dim];
  } else if (pool_type_list[in_id] == 6) {  // max
    const Dtype* in_data =
        reinterpret_cast<const Dtype*>(
            reinterpret_cast<uintptr_t>(input_locate_data[in_id])) +
        em_id + offset[offset_id] * in_dim;
    Dtype max = in_data[0];
    for (int i = 1; i < offset[offset_id + 1] - offset[offset_id]; i++) {
      Dtype cur_data = in_data[i * in_dim];
      max = cur_data > max ? cur_data : max;
    }
    output_data[tid] = max;
  } else {
    return;
  }
}

template <typename Dtype>
__global__ void sequence_pool_concat(const uint64_t* input_locate_data,
                                     const int* pool_type_list,
                                     Dtype* output_data,
                                     const int* offset,
                                     int batch,
                                     int in_num,
                                     const int* out_offset,
                                     const int* out_id_seq_map_data,
                                     int out_dim) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int em_id = tid % out_dim;
  int seq_id = tid / out_dim;
  int in_id = out_id_seq_map_data[em_id];
  em_id = em_id - out_offset[in_id];
  int in_dim = out_offset[in_id + 1] - out_offset[in_id];

  if (seq_id >= batch) {
    return;
  }
  Dtype* out_data = output_data + tid;
  int offset_id = in_id * (batch + 1) + seq_id;
  if (pool_type_list[in_id] == 4) {  // last
    const Dtype* in_data =
        reinterpret_cast<const Dtype*>(
            reinterpret_cast<uintptr_t>(input_locate_data[in_id])) +
        em_id;
    output_data[tid] = in_data[(offset[offset_id + 1] - 1) * in_dim];
  } else if (pool_type_list[in_id] == 6) {  // max
    const Dtype* in_data =
        reinterpret_cast<const Dtype*>(
            reinterpret_cast<uintptr_t>(input_locate_data[in_id])) +
        em_id + offset[offset_id] * in_dim;
    Dtype max = in_data[0];
    for (int i = 1; i < offset[offset_id + 1] - offset[offset_id]; i++) {
      Dtype cur_data = in_data[i * in_dim];
      max = cur_data > max ? cur_data : max;
    }
    output_data[tid] = max;
  } else {
    return;
  }
}

void SequencePoolConcatCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<CUDAContext>();
  auto stream = ctx.exec_stream();

  int in_num = param.X.size();
  std::vector<int64_t> shape({in_num, 1, 1, 1});
  _in_offset_tensor.Resize(shape);
  _in_ptr_tensor.Resize(shape);
  _in_pool_type_tensor.Resize(shape);
  int* in_pool_type_data =
      _in_pool_type_tensor.mutable_data<int>(TARGET(kCUDA));
  std::vector<int> pool_type_list;
  for (auto type : param.pool_type) {
    if (type == "AVERAGE") {
      pool_type_list.push_back(1);
    } else if (type == "SUM") {
      pool_type_list.push_back(2);
    } else if (type == "SQRT") {
      pool_type_list.push_back(3);
    } else if (type == "LAST") {
      pool_type_list.push_back(4);
    } else if (type == "FIRST") {
      pool_type_list.push_back(5);
    } else if (type == "MAX") {
      pool_type_list.push_back(6);
    } else {
      LOG(ERROR) << "pool type " << type << " is not supoorted.";
    }
  }
  _is_in_same_len = true;
  int in_len = param.X[0]->dims().count(1, param.X[0]->dims().size());
  std::vector<int> out_id_seq_map_list;
  std::vector<int> out_offset_list;
  int total_len = 0;
  out_offset_list.push_back(total_len);
  for (int i = 0; i < in_num; ++i) {
    int cur_len = param.X[i]->dims().count(1, param.X[i]->dims().size());
    _is_in_same_len = _is_in_same_len && in_len == cur_len;
    for (int k = 0; k < cur_len; ++k) {
      out_id_seq_map_list.push_back(i);
    }
    total_len += cur_len;
    out_offset_list.push_back(total_len);
  }
  std::vector<int64_t> out_id_seq_map_shape({total_len, 1, 1, 1});
  std::vector<int64_t> out_offset_shape({in_num + 1, 1, 1, 1});
  _out_offset_tensor.Resize(out_offset_shape);
  _out_id_seq_map_tensor.Resize(out_id_seq_map_shape);
  int* out_offset_data = _out_offset_tensor.mutable_data<int>(TARGET(kCUDA));
  int* out_id_seq_map_data =
      _out_id_seq_map_tensor.mutable_data<int>(TARGET(kCUDA));

  TargetWrapperCuda::MemcpyAsync(in_pool_type_data,
                                 &pool_type_list[0],
                                 sizeof(int) * param.X.size(),
                                 IoDirection::HtoD,
                                 stream);
  TargetWrapperCuda::MemcpyAsync(out_offset_data,
                                 &out_offset_list[0],
                                 sizeof(int) * out_offset_list.size(),
                                 IoDirection::HtoD,
                                 stream);
  TargetWrapperCuda::MemcpyAsync(out_id_seq_map_data,
                                 &out_id_seq_map_list[0],
                                 sizeof(int) * out_id_seq_map_list.size(),
                                 IoDirection::HtoD,
                                 stream);
  cudaStreamSynchronize(stream);
}

void SequencePoolConcatCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<CUDAContext>();
  auto stream = ctx.exec_stream();
  auto& inputs = param.X;

  auto offset = inputs[0]->lod()[0];
  int batch = offset.size() - 1;
  CHECK_GE(offset.size(), 1);
  std::vector<int> all_offset;
  for (int i = 0; i < inputs.size(); ++i) {
    auto it = all_offset.end();
    auto cur_offset = inputs[i]->lod()[0];
    all_offset.insert(it, cur_offset.begin(), cur_offset.end());
  }
  int total_size = all_offset.size();
  std::vector<int64_t> offset_shape({total_size, 1, 1, 1});
  _in_offset_tensor.Resize(offset_shape);
  int* offset_data = _in_offset_tensor.mutable_data<int>(TARGET(kCUDA));
  TargetWrapperCuda::MemcpyAsync(offset_data,
                                 &all_offset[0],
                                 sizeof(int) * all_offset.size(),
                                 IoDirection::HtoD,
                                 stream);

  std::vector<uint64_t> in_locate_vec;
  for (int i = 0; i < inputs.size(); ++i) {
    in_locate_vec.push_back(
        reinterpret_cast<uintptr_t>(inputs[i]->data<float>()));
  }
  uint64_t* in_locate_data =
      _in_ptr_tensor.mutable_data<uint64_t>(TARGET(kCUDA));
  TargetWrapperCuda::MemcpyAsync(in_locate_data,
                                 &in_locate_vec[0],
                                 sizeof(uint64_t) * inputs.size(),
                                 IoDirection::HtoD,
                                 stream);

  const int* in_pool_type_data = _in_pool_type_tensor.data<int>();
  const int* out_id_seq_map_data = _out_id_seq_map_tensor.data<int>();
  const int* out_offset_data = _out_offset_tensor.data<int>();
  int count = param.Out->numel();

  int in_dim = inputs[0]->numel() / inputs[0]->dims()[0];
  float* out_data = param.Out->mutable_data<float>(TARGET(kCUDA));
  int in_num = inputs.size();
  if (_is_in_same_len) {
    sequence_pool_concat<
        float><<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(
        in_locate_data,
        in_pool_type_data,
        out_data,
        offset_data,
        batch,
        in_num,
        in_dim);
  } else {
    int out_dim = param.Out->numel() / param.Out->dims()[0];
    sequence_pool_concat<
        float><<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(
        in_locate_data,
        in_pool_type_data,
        out_data,
        offset_data,
        batch,
        in_num,
        out_offset_data,
        out_id_seq_map_data,
        out_dim);
  }
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(sequence_pool_concat,
                     kCUDA,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::cuda::SequencePoolConcatCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .Finalize();
