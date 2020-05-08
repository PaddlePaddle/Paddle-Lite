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

#include "lite/kernels/cuda/search_fc_compute.h"
#include <gtest/gtest.h>
#include <memory>
#include <utility>
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

void fc_cpu_base(const lite::Tensor* X,
                 const lite::Tensor* W,
                 const lite::Tensor* b,
                 int out_size,
                 lite::Tensor* Out) {
  const float* data_in = X->data<float>();
  const float* bias = b->data<float>();
  const float* weights = W->data<float>();
  float* data_out = Out->mutable_data<float>();
  int out_rows = X->dims()[0];
  int in_cols = X->numel() / out_rows;
  int out_cols = W->numel() / in_cols;
  int index_out;

  for (int i = 0; i < out_rows; i++) {
    for (int j = 0; j < out_cols; j++) {
      index_out = i * out_cols + j;
      data_out[index_out] = bias ? bias[j] : 0;

      for (int k = 0; k < in_cols; k++) {
        data_out[index_out] +=
            data_in[i * in_cols + k] * weights[j * in_cols + k];
      }
    }
  }
}

TEST(search_fc, normal) {
  SearchFcCompute<float> search_fc_kernel;
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  auto& context = ctx->As<CUDAContext>();
  operators::SearchFcParam param;
  lite::Tensor X, X_gpu, W, W_gpu, b, b_gpu;
  lite::Tensor Out, Out_cpu, out_ref;
  std::vector<int64_t> x_shape{1, 4};
  X.Resize(lite::DDim(x_shape));
  std::vector<int64_t> w_shape{3, 4};
  W.Resize(lite::DDim(w_shape));
  std::vector<int64_t> b_shape{3};
  b.Resize(lite::DDim(b_shape));
  std::vector<int64_t> out_shape{1, 4};
  Out.Resize(lite::DDim(out_shape));
  out_ref.Resize(lite::DDim(out_shape));
  auto x_data = X.mutable_data<float>();
  auto w_data = W.mutable_data<float>();
  auto b_data = b.mutable_data<float>();
  auto out_data_ref = out_ref.mutable_data<float>();
  for (int64_t i = 0; i < X.dims().production(); i++) {
    x_data[i] = static_cast<float>(i);
  }
  for (int64_t i = 0; i < W.dims().production(); i++) {
    w_data[i] = static_cast<float>(i);
  }
  for (int64_t i = 0; i < b.dims().production(); i++) {
    b_data[i] = static_cast<float>(i);
  }
  X_gpu.Assign<float, lite::DDim, TARGET(kCUDA)>(x_data, X.dims());
  W_gpu.Assign<float, lite::DDim, TARGET(kCUDA)>(w_data, W.dims());
  b_gpu.Assign<float, lite::DDim, TARGET(kCUDA)>(b_data, b.dims());
  param.X = &X_gpu;
  param.W = &W_gpu;
  param.b = &b_gpu;
  param.out_size = 4;
  param.Out = &Out;
  search_fc_kernel.SetParam(param);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  context.SetExecStream(stream);
  search_fc_kernel.SetContext(std::move(ctx));
  search_fc_kernel.Run();
  fc_cpu_base(&X, &W, &b, 4, &out_ref);
  cudaDeviceSynchronize();
  const float* out_data = Out.data<float>();
  float* out_cpu_data = Out_cpu.mutable_data<float>();
  CopySync<TARGET(kCUDA)>(
      out_cpu_data, out_data, sizeof(float) * Out.numel(), IoDirection::DtoH);
  for (int i = 0; i < Out.numel(); ++i) {
    EXPECT_NEAR(out_cpu_data[i], out_data_ref[i], 1e-5);
  }
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
