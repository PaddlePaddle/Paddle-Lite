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

#include "lite/core/op_registry.h"
#include "lite/kernels/cuda/search_seq_fc_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

template <typename dtype>
__global__ void add_bias(int n,
                         int output_size,
                         const dtype* bias,
                         dtype* dout) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int bias_index = index % output_size;
  if (index < n) {
    dout[index] = dout[index] + bias[bias_index];
  }
}

void SearchSeqFcCompute::PrepareForRun() {
  gemm_impl_.reset(new lite::cuda::math::Gemm<float, float>);
}

void SearchSeqFcCompute::Run() {
  auto& param = this->Param<param_t>();
  CHECK(ctx_) << "running context should be set first";
  auto& cuda_ctx = ctx_->template As<CUDAContext>();
  auto cuda_stream = cuda_ctx.exec_stream();

  auto x = param.x;
  auto w = param.w;
  auto b = param.b;
  auto out = param.out;
  auto out_size = param.out_size;
  const auto x_dims = x->dims();
  const auto w_dims = w->dims();
  const auto out_dims = out->dims();
  CHECK_EQ(x_dims.size(), 2) << "The Input(X) should be 2-D tensor.";
  CHECK_EQ(w_dims.size(), 2) << "W should be 2-D tensor.";
  CHECK_EQ(out_dims.size(), 2) << "The Output(Out) should be 2-D tensor.";
  CHECK_EQ(x_dims[1], w_dims[1]) << "Wrong shape: x_dims[1] != w_dims[1]";
  CHECK_EQ(w_dims[0], out_size) << "Wrong shape: w_dims[0] != out_size";
  CHECK_EQ(out_dims[0], x_dims[0]) << "Wrong shape: out_dims[0] != x_dims[0]";
  CHECK_EQ(out_dims[1], out_size) << "Wrong shape: out_dims[1] != out_size";
  int M = x_dims[0];
  int K = x_dims[1];
  int N = w_dims[0];
  auto x_data = x->data<float>();
  auto w_data = w->data<float>();
  auto out_data = out->mutable_data<float>(TARGET(kCUDA));

  CHECK(gemm_impl_->init(false, true, M, N, K, &cuda_ctx));
  gemm_impl_->run(1.0f, 0.0f, x_data, w_data, out_data, &cuda_ctx);

  if (b != nullptr) {
    auto b_dims = b->dims();
    CHECK_EQ(b_dims.size(), 1) << "b should be 1-D tensor.";
    CHECK_EQ(b_dims[0], w_dims[0]) << "Wrong shape: b_dims[0] != w_dims[0]";
    auto b_data = b->mutable_data<float>();
    int total_size = M * N;
    add_bias<float><<<CUDA_GET_BLOCKS(total_size),
                      CUDA_NUM_THREADS,
                      0,
                      cuda_stream>>>(total_size, N, b_data, out_data);
  }
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(search_seq_fc,
                     kCUDA,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::cuda::SearchSeqFcCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindInput("b", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .Finalize();
