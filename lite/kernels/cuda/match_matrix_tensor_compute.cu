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
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>

#include <algorithm>
#include <vector>

#include "lite/backends/cuda/cuda_utils.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/cuda/match_matrix_tensor_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

template <typename dtype>
__global__ void ReorganizeOutput(const dtype* src,
                                 dtype* dst,
                                 const int count,
                                 const int len_l,
                                 const int len_r,
                                 const int dim_t) {
  CUDA_KERNEL_LOOP(tid, count) {
    int r_id = tid % len_r;
    int l_id = tid / len_r % len_l;
    int t_id = tid / len_r / len_l % dim_t;
    dst[tid] = src[r_id * len_l * dim_t + l_id * dim_t + t_id];
  }
}

template <typename dtype>
__global__ void ReorganizeOutputFuseRelu(const dtype* src,
                                         dtype* dst,
                                         const int count,
                                         const int len_l,
                                         const int len_r,
                                         const int dim_t) {
  CUDA_KERNEL_LOOP(tid, count) {
    int r_id = tid % len_r;
    int l_id = tid / len_r % len_l;
    int t_id = tid / len_r / len_l % dim_t;
    dst[tid] = src[r_id * len_l * dim_t + l_id * dim_t + t_id] > 0
                   ? src[r_id * len_l * dim_t + l_id * dim_t + t_id]
                   : 0;
  }
}

template <typename dtype>
__global__ void PaddingOutNotSameL(const dtype* src,
                                   const int* offset,
                                   const int seq_num_r,
                                   const int max_len_r,
                                   const int dim_t,
                                   const int len_l,
                                   const int count,
                                   dtype* dst) {
  CUDA_KERNEL_LOOP(tid, count) {
    int batch_id = tid / (dim_t * len_l * max_len_r);
    int r_id = tid % max_len_r;
    int l_id = tid / max_len_r % len_l;
    int t_id = tid / max_len_r / len_l % dim_t;

    int cur_len = offset[batch_id + 1] - offset[batch_id];
    if (r_id < cur_len) {
      dst[tid] = src[(offset[batch_id] + r_id) * (len_l * dim_t) + t_id +
                     l_id * dim_t];
    } else {
      dst[tid] = 0.f;
    }
  }
}

template <typename dtype>
__global__ void PaddingOutNotSameLFuseRelu(const dtype* src,
                                           const int* offset,
                                           const int seq_num_r,
                                           const int max_len_r,
                                           const int dim_t,
                                           const int len_l,
                                           const int count,
                                           dtype* dst) {
  CUDA_KERNEL_LOOP(tid, count) {
    int batch_id = tid / (dim_t * len_l * max_len_r);
    int r_id = tid % max_len_r;
    int l_id = tid / max_len_r % len_l;
    int t_id = tid / max_len_r / len_l % dim_t;

    int cur_len = offset[batch_id + 1] - offset[batch_id];
    if (r_id < cur_len) {
      dtype val = src[(offset[batch_id] + r_id) * (len_l * dim_t) +
                      l_id * dim_t + t_id];
      dst[tid] = val > 0.f ? val : 0.f;
    } else {
      dst[tid] = 0.f;
    }
  }
}

template <typename dtype>
__global__ void PaddingOut(const dtype* src,
                           const int* offset,
                           const int batch_r,
                           const int max_len_r,
                           const int tl,
                           const int count,
                           const bool fuse_relu,
                           dtype* dst) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int thread_num = blockDim.x * gridDim.x;
  for (tid = threadIdx.x + blockIdx.x * blockDim.x; tid < count;
       tid += thread_num) {
    int seq_id = tid / (tl * max_len_r);
    int tl_id = (tid / (max_len_r)) % tl;
    int r_id = tid % max_len_r;
    int cur_len = offset[seq_id + 1] - offset[seq_id];
    if (r_id < cur_len) {
      if (fuse_relu) {
        dst[tid] = src[(offset[seq_id] + r_id) * tl + tl_id] > 0
                       ? src[(offset[seq_id] + r_id) * tl + tl_id]
                       : 0;
      } else {
        dst[tid] = src[(offset[seq_id] + r_id) * tl + tl_id];
      }
    } else {
      dst[tid] = 0.f;
    }
  }
}

template <typename T, PrecisionType PType>
void MatchMatrixTensorCompute<T, PType>::PrepareForRun() {
  gemm_impl_.reset(new lite::cuda::math::Gemm<T, T>);
}

template <typename T, PrecisionType PType>
void MatchMatrixTensorCompute<T, PType>::Run() {
  auto& param = this->template Param<param_t>();
  auto& context = this->ctx_->template As<CUDAContext>();
  auto stream = context.exec_stream();

  auto* x = param.x;
  auto* w = param.w;
  auto* y = param.y;
  auto* out = param.out;

  int dim_t = param.dim_t;
  int dim_in = x->dims()[1];
  bool fuse_relu = param.fuse_relu;

  bool is_x_lod_same_len{true};
  bool is_l_same{true};

  const auto& offset_l = x->lod()[0];
  const auto& offset_r = y->lod()[0];
  std::vector<int> offset_r_int(offset_r.size());
  std::transform(offset_r.begin(),
                 offset_r.end(),
                 offset_r_int.begin(),
                 [](int64_t x) -> int { return static_cast<int>(x); });

  int batch = offset_r.size() - 1;
  int batch_l = x->dims()[0];
  int batch_r = offset_r[offset_r.size() - 1];
  int len_l = offset_l[1] - offset_l[0];
  int max_len_l = len_l;
  for (int i = 1; i < offset_l.size() - 1; i++) {
    int cur_len = offset_l[i + 1] - offset_l[i];
    if (cur_len != len_l) {
      is_x_lod_same_len = false;
      max_len_l = cur_len > max_len_l ? cur_len : max_len_l;
    }
  }
  int max_len_r = 0;
  for (int i = 0; i < offset_r.size() - 1; ++i) {
    int cur_len = offset_r[i + 1] - offset_r[i];
    max_len_r = cur_len > max_len_r ? cur_len : max_len_r;
  }

  const T* input_l = x->template data<T>();
  const T* r_data = y->template data<T>();
  const T* weight_data = w->template data<T>();

  // compare the mean value of each sequence to determine whether the input is
  // same.
  if (is_x_lod_same_len) {
    thrust::device_ptr<T> dev_ptr(const_cast<T*>(input_l));
    T seq_sum = thrust::reduce(dev_ptr,
                               dev_ptr + len_l * dim_in,
                               static_cast<T>(0),
                               thrust::plus<T>());
    T all_sum = thrust::reduce(dev_ptr,
                               dev_ptr + x->dims()[0] * dim_in,
                               static_cast<T>(0),
                               thrust::plus<T>());
    // TODO(wilber) consider for half.
    if (abs(all_sum / batch - seq_sum) > 1e-5) {
      is_l_same = false;
    }
  } else {
    is_l_same = false;
  }

  if (is_l_same) {
    input_l_transform_.Resize({1, dim_t, dim_in, len_l});
    input_l_transform_reorganize_.Resize({1, dim_t, len_l, dim_in});
    param.tmp->Resize({batch, max_len_r, dim_t, len_l});
    out->Resize({batch, dim_t, len_l, max_len_r});

    offset_r_.Resize({static_cast<int64_t>(offset_r.size())});
    TargetWrapperCuda::MemcpyAsync(offset_r_.mutable_data<int>(TARGET(kCUDA)),
                                   &offset_r_int[0],
                                   sizeof(int) * offset_r.size(),
                                   IoDirection::HtoD,
                                   stream);

    T* input_l_transform = input_l_transform_.mutable_data<T>(TARGET(kCUDA));
    T* input_l_transform_reorganize =
        input_l_transform_reorganize_.mutable_data<T>(TARGET(kCUDA));
    T* output_tmp = param.tmp->template mutable_data<T>(TARGET(kCUDA));
    T* out_data = out->template mutable_data<T>(TARGET(kCUDA));

    gemm_impl_->init(true, true, dim_t * dim_in, len_l, dim_in, &context);
    gemm_impl_->run(
        1.0f, 0.0f, weight_data, input_l, input_l_transform, &context);
    trans_.transpose(input_l_transform_reorganize,
                     input_l_transform,
                     input_l_transform_.dims().Vectorize(),
                     {0, 1, 3, 2},
                     &stream);

    gemm_impl_->init(false, true, batch_r, dim_t * len_l, dim_in, &context);
    gemm_impl_->run(
        1.0f, 0.0f, r_data, input_l_transform_reorganize, output_tmp, &context);
    int batch = offset_r.size() - 1;
    int count = batch * max_len_r * dim_t * len_l;
    const int blocks = 512;
    const int grids = (count + blocks - 1) / blocks;
    PaddingOut<T><<<grids, blocks, 0, stream>>>(param.tmp->template data<T>(),
                                                offset_r_.data<int>(),
                                                batch,
                                                max_len_r,
                                                dim_t * len_l,
                                                count,
                                                fuse_relu,
                                                out_data);
    CUDA_POST_KERNEL_CHECK;
    out->set_lod(y->lod());
  } else if (is_x_lod_same_len) {
    input_l_transform_.Resize({dim_t, dim_in, batch_l});
    input_l_transform_reorganize_.Resize({batch_l, dim_t, dim_in});
    param.tmp->Resize({batch_r, dim_t, len_l});
    out->Resize({batch, dim_t, len_l, max_len_r});

    offset_r_.Resize({static_cast<int64_t>(offset_r.size())});
    TargetWrapperCuda::MemcpyAsync(offset_r_.mutable_data<int>(TARGET(kCUDA)),
                                   &offset_r_int[0],
                                   sizeof(int) * offset_r.size(),
                                   IoDirection::HtoD,
                                   stream);

    T* input_l_transform = input_l_transform_.mutable_data<T>(TARGET(kCUDA));
    T* input_l_transform_reorganize =
        input_l_transform_reorganize_.mutable_data<T>(TARGET(kCUDA));
    T* output_tmp = param.tmp->template mutable_data<T>(TARGET(kCUDA));
    T* out_data = out->template mutable_data<T>(TARGET(kCUDA));

    gemm_impl_->init(true, true, dim_t * dim_in, batch_l, dim_in, &context);
    gemm_impl_->run(
        1.0f, 0.0f, weight_data, input_l, input_l_transform, &context);
    trans_.transpose(input_l_transform_reorganize,
                     input_l_transform,
                     input_l_transform_.dims().Vectorize(),
                     {2, 0, 1},
                     &stream);

    auto* tmp_out = output_tmp;
    for (int i = 0; i < batch; ++i) {
      int len_r = offset_r[i + 1] - offset_r[i];
      auto tmp_input_r = r_data + offset_r[i] * dim_in;
      auto tmp_input_l =
          input_l_transform_reorganize + i * len_l * dim_t * dim_in;
      gemm_impl_->init(false, true, len_r, len_l * dim_t, dim_in, &context);
      gemm_impl_->run(1.0f, 0.0f, tmp_input_r, tmp_input_l, tmp_out, &context);
      tmp_out += len_r * dim_t * len_l;
    }
    int count = batch * dim_t * len_l * max_len_r;
    if (fuse_relu) {
      PaddingOutNotSameLFuseRelu<
          T><<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(
          param.tmp->template data<T>(),
          offset_r_.data<int>(),
          batch,
          max_len_r,
          dim_t,
          len_l,
          count,
          out_data);
    } else {
      PaddingOutNotSameL<
          T><<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(
          param.tmp->template data<T>(),
          offset_r_.data<int>(),
          batch,
          max_len_r,
          dim_t,
          len_l,
          count,
          out_data);
    }
    CUDA_POST_KERNEL_CHECK;
    out->set_lod(y->lod());
  } else {
    // naive method.
    int out_len = 0;
    for (size_t i = 0; i < batch; i++) {
      int len_l = offset_l[i + 1] - offset_l[i];
      int len_r = offset_r[i + 1] - offset_r[i];
      out_len += len_l * len_r * dim_t;
    }
    out->Resize({out_len});
    T* out_data = out->template mutable_data<T>(TARGET(kCUDA));
    param.tmp->Resize({max_len_r, max_len_l, dim_t});
    T* out_tmp_data = param.tmp->template mutable_data<T>(TARGET(kCUDA));

    input_l_transform_.Resize({dim_t, dim_in, batch_l});
    input_l_transform_reorganize_.Resize({batch_l, dim_t, dim_in});
    T* input_l_transform = input_l_transform_.mutable_data<T>(TARGET(kCUDA));
    T* input_l_transform_reorganize =
        input_l_transform_reorganize_.mutable_data<T>(TARGET(kCUDA));

    std::vector<int> offset_l_int(offset_l.size());
    std::transform(offset_l.begin(),
                   offset_l.end(),
                   offset_l_int.begin(),
                   [](int64_t x) -> int { return static_cast<int>(x); });
    offset_l_.Resize({static_cast<int64_t>(offset_l.size())});
    TargetWrapperCuda::MemcpyAsync(offset_l_.mutable_data<int>(TARGET(kCUDA)),
                                   &offset_l_int[0],
                                   sizeof(int) * offset_l.size(),
                                   IoDirection::HtoD,
                                   stream);

    gemm_impl_->init(true, true, dim_t * dim_in, batch_l, dim_in, &context);
    gemm_impl_->run(
        1.0f, 0.0f, weight_data, input_l, input_l_transform, &context);
    trans_.transpose(input_l_transform_reorganize,
                     input_l_transform,
                     input_l_transform_.dims().Vectorize(),
                     {2, 0, 1},
                     &stream);

    auto* l_data = input_l_transform_reorganize;

    for (size_t i = 0; i < batch; ++i) {
      int len_l = offset_l[i + 1] - offset_l[i];
      int len_r = offset_r[i + 1] - offset_r[i];
      int out_seq_count = len_r * len_l * dim_t;
      gemm_impl_->init(false, true, len_r, len_l * dim_t, dim_in, &context);
      gemm_impl_->run(1.0f,
                      0.0f,
                      r_data,
                      l_data,
                      param.tmp->template mutable_data<T>(TARGET(kCUDA)),
                      &context);
      if (fuse_relu) {
        ReorganizeOutputFuseRelu<
            T><<<CUDA_GET_BLOCKS(out_seq_count), CUDA_NUM_THREADS, 0, stream>>>(
            param.tmp->template data<T>(),
            out_data,
            out_seq_count,
            len_l,
            len_r,
            dim_t);
      } else {
        ReorganizeOutput<
            T><<<CUDA_GET_BLOCKS(out_seq_count), CUDA_NUM_THREADS, 0, stream>>>(
            param.tmp->template data<T>(),
            out_data,
            out_seq_count,
            len_l,
            len_r,
            dim_t);
      }
      r_data += len_r * dim_in;
      l_data += dim_t * len_l * dim_in;
      out_data += out_seq_count;
    }
    CUDA_POST_KERNEL_CHECK;

    std::vector<std::vector<uint64_t>> lod;
    lod.push_back(x->lod()[0]);
    lod.push_back(y->lod()[0]);
    lod.push_back({dim_t});
    out->set_lod(lod);
  }
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using MMTFp32 =
    paddle::lite::kernels::cuda::MatchMatrixTensorCompute<float,
                                                          PRECISION(kFloat)>;

REGISTER_LITE_KERNEL(match_matrix_tensor, kCUDA, kFloat, kNCHW, MMTFp32, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindInput("W",
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
    .BindOutput("Tmp",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .Finalize();
