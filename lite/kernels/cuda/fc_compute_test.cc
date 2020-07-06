// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/cuda/fc_compute.h"

#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "lite/api/test_helper.h"
#include "lite/utils/float16.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

class FcTest : public ::testing::Test {
 protected:
  FcTest()
      : m(128),
        k(512),
        n(64),
        in_num_col_dims(1),
        act_type("relu"),
        x_shape({m, k}),
        w_shape({k, n}),
        b_shape({n}),
        out_shape({m, n}) {
    X_gpu.Resize(lite::DDim(x_shape));
    X_ref.Resize(lite::DDim(x_shape));

    W_gpu.Resize(lite::DDim(w_shape));
    W_ref.Resize(lite::DDim(w_shape));

    b_gpu.Resize(lite::DDim(b_shape));
    b_ref.Resize(lite::DDim(b_shape));

    auto x_ref_data = X_ref.mutable_data<float>();
    auto w_ref_data = W_ref.mutable_data<float>();
    auto b_ref_data = b_ref.mutable_data<float>();

    // prepare input
    for (int64_t i = 0; i < X_ref.numel(); i++) {
      x_ref_data[i] = static_cast<float>(i % 10 * 0.2);
    }
    for (int64_t i = 0; i < W_ref.numel(); i++) {
      w_ref_data[i] = static_cast<float>(i % 10 * 0.2);
    }
    for (int64_t i = 0; i < b_ref.numel(); i++) {
      b_ref_data[i] = static_cast<float>(i % 10 * 0.2);
    }

    Out_ref.Resize(lite::DDim(out_shape));
    Out_cpu.Resize(Out_ref.dims());
    Out_gpu.Resize(Out_ref.dims());
    fc_cpu_base(&X_ref, &W_ref, &b_ref, &Out_ref);

    device_init();
  }

  void device_init() {
    ctx.reset(new KernelContext);
    cudaStreamCreate(&stream);
    auto& context = ctx->As<CUDAContext>();
    context.SetExecStream(stream);
    param.input = &X_gpu;
    param.w = &W_gpu;
    param.bias = &b_gpu;
    param.in_num_col_dims = in_num_col_dims;
    param.activation_type = act_type;
    param.output = &Out_gpu;
  }

  void float_data_init() {
    X_gpu.Assign<float, lite::DDim, TARGET(kCUDA)>(X_ref.data<float>(),
                                                   X_gpu.dims());
    W_gpu.Assign<float, lite::DDim, TARGET(kCUDA)>(W_ref.data<float>(),
                                                   W_gpu.dims());
    b_gpu.Assign<float, lite::DDim, TARGET(kCUDA)>(b_ref.data<float>(),
                                                   b_gpu.dims());
  }

  void half_data_init() {
    X_half.Resize(lite::DDim(x_shape));
    auto x_half_data = X_half.mutable_data<half>();
    for (int64_t i = 0; i < X_half.numel(); i++) {
      x_half_data[i] = half(lite::float16(X_ref.data<float>()[i]));
    }
    X_gpu.Assign<half, lite::DDim, TARGET(kCUDA)>(x_half_data, X_gpu.dims());
    W_half.Resize(W_ref.dims());
    auto w_half_data = W_half.mutable_data<half>();
    for (int64_t i = 0; i < W_half.numel(); i++) {
      w_half_data[i] = half(lite::float16(W_ref.data<float>()[i]));
    }
    W_gpu.Assign<half, lite::DDim, TARGET(kCUDA)>(w_half_data, W_gpu.dims());
    b_half.Resize(b_ref.dims());
    auto b_half_data = b_half.mutable_data<half>();
    for (int64_t i = 0; i < b_half.numel(); i++) {
      b_half_data[i] = half(lite::float16(b_ref.data<float>()[i]));
    }
    b_gpu.Assign<half, lite::DDim, TARGET(kCUDA)>(b_half_data, b_gpu.dims());
  }

  void fc_cpu_base(const lite::Tensor* X,
                   const lite::Tensor* W,
                   const lite::Tensor* b,
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
              data_in[i * in_cols + k] * weights[k * out_cols + j];
        }
        if (act_type == "relu") {
          data_out[index_out] *= static_cast<int>(data_out[index_out] > 0);
        }
      }
    }
  }

  int m, k, n, in_num_col_dims;
  std::string act_type;
  std::vector<int64_t> x_shape, w_shape, b_shape, out_shape;
  lite::Tensor X_ref, W_ref, b_ref, Out_ref;
  lite::Tensor X_gpu, W_gpu, b_gpu;
  lite::Tensor X_half, W_half, b_half;
  lite::Tensor Out_cpu, Out_gpu;

  operators::FcParam param;
  std::unique_ptr<KernelContext> ctx;
  cudaStream_t stream;
};

TEST_F(FcTest, TestFP32) {
  float_data_init();
  FcCompute<float, PRECISION(kFloat)> kernel;
  kernel.SetParam(param);
  kernel.SetContext(std::move(ctx));

  for (int i = 0; i < FLAGS_warmup; ++i) {
    kernel.Launch();
    cudaDeviceSynchronize();
  }

  auto start = GetCurrentUS();
  kernel.PrepareForRun();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    kernel.Run();
  }
  cudaDeviceSynchronize();
  auto duration = (GetCurrentUS() - start) / 1000.0;
  LOG(INFO) << "fp32, warmup: " << FLAGS_warmup
            << ", repeats: " << FLAGS_repeats << ", spend "
            << duration / FLAGS_repeats << " ms in average.";

  CopySync<TARGET(kCUDA)>(Out_cpu.mutable_data<float>(),
                          Out_gpu.data<float>(),
                          sizeof(float) * Out_gpu.numel(),
                          IoDirection::DtoH);

  for (int i = 0; i < Out_gpu.numel(); ++i) {
    float res = Out_cpu.data<float>()[i];
    float ref = Out_ref.data<float>()[i];
    EXPECT_NEAR(fabs(res - ref) / ref, 0.f, 1e-5);
  }
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
