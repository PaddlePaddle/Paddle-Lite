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
#include "lite/kernels/cuda/match_matrix_tensor_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {
using Tensor = lite::Tensor;

template <typename dtype>
void gpu_transpose(
    cublasHandle_t handle, const dtype* src, int M, int N, dtype* dst);

template <>
void gpu_transpose<float>(
    cublasHandle_t handle, const float* src, int M, int N, float* dst) {
  float alpha = 1.0;
  float beta = 0.0;
  CUBLAS_CHECK(cublasSgeam(handle,
                           CUBLAS_OP_T,
                           CUBLAS_OP_N,
                           M,
                           N,
                           &alpha,
                           src,
                           N,
                           &beta,
                           dst,
                           M,
                           dst,
                           M));
}

template <typename dtype>
__global__ void padding_out(const dtype* src,
                            const int* offset,
                            const int seq_num_r,
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

void MatchMatrixTensorCompute::PrepareForRun() {
  gemm_impl_.reset(new lite::cuda::math::Gemm<float, float>);
}

void MatchMatrixTensorCompute::Run() {
  CHECK(ctx_) << "running context should be set first";
  auto& param = this->Param<param_t>();
  auto& context = this->ctx_->template As<CUDAContext>();
  auto stream = context.exec_stream();

  auto* x = param.x;
  auto* w = param.w;
  auto* y = param.y;
  auto* out = param.out;
  auto* tmp = param.tmp;
  int dim_t = param.dim_t;
  int dim_in = x->dims()[1];
  bool fuse_relu = param.fuse_relu;

  const auto& offset_l = x->lod()[0];
  const auto& offset_r = y->lod()[0];
  std::vector<int> offset_r_int(offset_r.size());
  std::transform(offset_r.begin(),
                 offset_r.end(),
                 offset_r_int.begin(),
                 [](int64_t x) -> int { return static_cast<int>(x); });

  int batch = offset_r.size() - 1;
  int len_l = offset_l[1] - offset_l[0];
  for (int i = 1; i < offset_l.size() - 1; i++) {
    int cur_len = offset_l[i + 1] - offset_l[i];
    CHECK_EQ(cur_len, len_l)
        << "each sequence of left matrix is the same length";
  }
  int max_len_r = 0;
  for (int i = 0; i < offset_r.size() - 1; ++i) {
    int cur_len = offset_r[i + 1] - offset_r[i];
    max_len_r = cur_len > max_len_r ? cur_len : max_len_r;
  }

  _input_l_transform.Resize({batch, dim_t, dim_in, len_l});
  _input_l_transform_reorganize.Resize({batch, dim_t, len_l, dim_in});
  _output_tmp.Resize({batch, max_len_r, dim_t, len_l});
  out->Resize({batch, dim_t, len_l, max_len_r});

  _offset_r.Resize({static_cast<int64_t>(offset_r.size())});
  TargetWrapperCuda::MemcpyAsync(_offset_r.mutable_data<int>(TARGET(kCUDA)),
                                 &offset_r_int[0],
                                 sizeof(int) * offset_r.size(),
                                 IoDirection::HtoD,
                                 stream);

  int len_r = offset_r[offset_r.size() - 1];
  const float* input_l = x->data<float>();
  const float* input_r = y->data<float>();
  const float* weight_data = w->data<float>();
  float* input_l_transform =
      _input_l_transform.mutable_data<float>(TARGET(kCUDA));
  float* input_l_transform_reorganize =
      _input_l_transform_reorganize.mutable_data<float>(TARGET(kCUDA));
  float* output_tmp = _output_tmp.mutable_data<float>(TARGET(kCUDA));
  float* out_data = out->mutable_data<float>(TARGET(kCUDA));

  gemm_impl_->init(true, true, dim_t * dim_in, len_l, dim_in, &context);
  gemm_impl_->run(
      1.0f, 0.0f, weight_data, input_l, input_l_transform, &context);
  for (int i = 0; i < dim_t; ++i) {
    int offset = i * dim_in * len_l;
    gpu_transpose(gemm_impl_->get_handle(),
                  input_l_transform + offset,
                  dim_in,
                  len_l,
                  input_l_transform_reorganize + offset);
  }
  gemm_impl_->init(false, true, len_r, dim_t * len_l, dim_in, &context);
  gemm_impl_->run(
      1.0f, 0.0f, input_r, input_l_transform_reorganize, output_tmp, &context);
  int seq_num = offset_r.size() - 1;
  int count = seq_num * max_len_r * dim_t * len_l;
  const int blocks = 512;
  const int grids = (count + blocks - 1) / blocks;
  padding_out<float><<<grids, blocks, 0, stream>>>(_output_tmp.data<float>(),
                                                   _offset_r.data<int>(),
                                                   seq_num,
                                                   max_len_r,
                                                   dim_t * len_l,
                                                   count,
                                                   fuse_relu,
                                                   out_data);
  out->set_lod(y->lod());
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(match_matrix_tensor,
                     kCUDA,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::cuda::MatchMatrixTensorCompute,
                     def)
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
