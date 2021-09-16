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

#include "lite/kernels/cuda/var_conv_2d_compute.h"
#include <gtest/gtest.h>
#include <memory>
#include <utility>
#include <vector>
#include "lite/api/test/test_helper.h"
#include "lite/utils/float16.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

static void im2col_ref(const lite::Tensor& input,
                       const int batch,
                       const int height,
                       const int width,
                       const int kernel_h,
                       const int kernel_w,
                       const int stride_h,
                       const int stride_w,
                       const int input_channel,
                       lite::Tensor* col) {
  int top_im_x = (width - 1) / stride_w + 1;
  int top_im_y = (height - 1) / stride_h + 1;
  int top_x = top_im_x * top_im_y;
  int top_y = input_channel * kernel_h * kernel_w;
  int top_size = top_x * top_y;
  std::vector<int64_t> col_dims_vec{batch, top_size};
  col->Resize(col_dims_vec);
  auto* top_data = col->mutable_data<float>();
  const auto* bottom_data = input.data<float>();
  int kernel_win_size = kernel_h * kernel_w;
  int half_kernel_h = kernel_h / 2;
  int half_kernel_w = kernel_w / 2;
  for (int b = 0; b < batch; ++b) {
    int top_im_x = (width - 1) / stride_w + 1;
    int top_im_y = (height - 1) / stride_h + 1;
    int top_x = top_im_y * top_im_x;
    for (int z = 0; z < input_channel; ++z) {
      int row_offset = kernel_win_size * z;
      int im_offset = z * width * height;
      for (int y = 0; y < height; y += stride_h) {
        for (int x = 0; x < width; x += stride_w) {
          int col_offset = x / stride_w + y / stride_h * top_im_x;
          for (int ky = 0; ky < kernel_h; ++ky) {
            for (int kx = 0; kx < kernel_w; ++kx) {
              int im_y = y + ky - half_kernel_h;
              int im_x = x + kx - half_kernel_w;
              if (im_x >= 0 && im_x < width && im_y >= 0 && im_y < height) {
                top_data[b * top_size +
                         (row_offset + ky * kernel_w + kx) * top_x +
                         col_offset] =
                    bottom_data[b * input_channel * height * width + im_offset +
                                im_y * width + im_x];
              } else {
                top_data[b * top_size +
                         (row_offset + ky * kernel_w + kx) * top_x +
                         col_offset] = 0;
              }
            }
          }
        }
      }
    }
  }
}

static void naive_sgemm(const bool transpose_A,
                        const bool transpose_B,
                        const int M,
                        const int N,
                        const int K,
                        const float alpha,
                        const float* A,  // m x k (after transpose if TransA)
                        const int lda,   // leading dimension of a
                        const float* B,  // k x n (after transpose if TransB)
                        const int ldb,   // leading dimension of b
                        const float beta,
                        float* C,  // m x n
                        const int ldc) {
  for (int m = 0; m < M; ++m) {
    for (int k = 0; k < K; ++k) {
      for (int n = 0; n < N; ++n) {
        C[m * N + n] += beta * C[m * N + n];
        size_t A_idx = 0, B_idx = 0;
        if (transpose_A) {
          A_idx = k * M + m;  // A is k x m
        } else {
          A_idx = m * K + k;  // A is m x k
        }

        if (transpose_B) {
          B_idx = n * K + k;  // B is n x k
        } else {
          B_idx = k * N + n;  // B is k x n
        }

        C[m * N + n] += alpha * A[A_idx] * B[B_idx];
      }
    }
  }
}

static void var_conv_2d_ref(const lite::Tensor* bottom,
                            const lite::Tensor* w,
                            const int batch,
                            const int height,
                            const int width,
                            const int kernel_h,
                            const int kernel_w,
                            const int stride_h,
                            const int stride_w,
                            const int input_channel,
                            const int output_channel,
                            lite::Tensor* top,
                            lite::Tensor* col) {
  im2col_ref(*bottom,
             batch,
             height,
             width,
             kernel_h,
             kernel_w,
             stride_h,
             stride_w,
             input_channel,
             col);
  int top_im_x = (width - 1) / stride_w + 1;
  int top_im_y = (height - 1) / stride_h + 1;
  int top_im_size = top_im_y * top_im_x;
  auto* top_data = top->mutable_data<float>();
  const auto* w_data = w->data<float>();
  const auto* col_data = col->data<float>();

  for (int b = 0; b < batch; ++b) {
    naive_sgemm(
        false,
        false,
        output_channel,
        top_im_size,
        input_channel * kernel_h * kernel_w,
        1.0,
        w_data,
        input_channel * kernel_h * kernel_w,
        col_data + b * input_channel * kernel_h * kernel_w * top_im_size,
        top_im_size,
        0.0,
        top_data + b * output_channel * top_im_size,
        top_im_size);
  }
}

class VarConvTest : public ::testing::Test {
 protected:
  VarConvTest()
      : batch(2),
        in_channels(4),
        out_channels(32),
        height(128),
        width(128),
        kernel_h(5),
        kernel_w(5),
        stride_h(1),
        stride_w(1),
        x_lod({{0, 128, 256}}),
        x_shape({batch, in_channels, height, width}),
        w_shape({out_channels, in_channels, kernel_h, kernel_w}),
        out_shape({batch,
                   out_channels,
                   (height - 1) / stride_h + 1,
                   (width - 1) / stride_w + 1}) {
    X_gpu.Resize(lite::DDim(x_shape));
    X_ref.Resize(lite::DDim(x_shape));
    X_ref.set_lod(x_lod);

    W_gpu.Resize(lite::DDim(w_shape));
    W_ref.Resize(lite::DDim(w_shape));

    auto x_ref_data = X_ref.mutable_data<float>();
    auto w_ref_data = W_ref.mutable_data<float>();

    // prepare input
    for (int64_t i = 0; i < X_ref.numel(); i++) {
      x_ref_data[i] = static_cast<float>(i % 10 * 0.2);
    }
    for (int64_t i = 0; i < W_ref.numel(); i++) {
      w_ref_data[i] = static_cast<float>(i % 10 * 0.2);
    }

    Out_ref.Resize(lite::DDim(out_shape));
    Out_cpu.Resize(lite::DDim(out_shape));
    conv_cpu_base(&X_ref, &W_ref, &Out_ref, &Col_ref);

    device_init();
  }

  void device_init() {
    ctx.reset(new KernelContext);
    cudaStreamCreate(&stream);
    auto& context = ctx->As<CUDAContext>();
    context.SetExecStream(stream);
    param.X = &X_gpu;
    param.W = &W_gpu;
    param.Out = &Out_gpu;
    param.stride_h = stride_h;
    param.stride_w = stride_w;
    param.kernel_h = kernel_h;
    param.kernel_w = kernel_w;
    param.input_channel = in_channels;
    param.output_channel = out_channels;
  }

  void float_data_init() {
    X_gpu.Assign<float, lite::DDim, TARGET(kCUDA)>(X_ref.data<float>(),
                                                   X_gpu.dims());
    X_gpu.set_lod(X_ref.lod());
    W_gpu.Assign<float, lite::DDim, TARGET(kCUDA)>(W_ref.data<float>(),
                                                   W_gpu.dims());
  }

  void half_data_init() {
    X_half.Resize(lite::DDim(x_shape));
    auto x_half_data = X_half.mutable_data<__half>();
    for (int64_t i = 0; i < X_half.numel(); i++) {
      x_half_data[i] = half(lite::float16(X_ref.data<float>()[i]));
    }
    X_gpu.Assign<__half, lite::DDim, TARGET(kCUDA)>(x_half_data, X_gpu.dims());
    X_gpu.set_lod(X_ref.lod());

    W_half.Resize(W_ref.dims());
    auto w_half_data = W_half.mutable_data<half>();
    for (int64_t i = 0; i < W_half.numel(); i++) {
      w_half_data[i] = half(lite::float16(W_ref.data<float>()[i]));
    }
    W_gpu.Assign<half, lite::DDim, TARGET(kCUDA)>(w_half_data, W_gpu.dims());
  }

  void conv_cpu_base(const lite::Tensor* X,
                     const lite::Tensor* W,
                     lite::Tensor* Out,
                     lite::Tensor* Col) {
    var_conv_2d_ref(X,
                    W,
                    batch,
                    height,
                    width,
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                    in_channels,
                    out_channels,
                    Out,
                    Col);
  }

  int batch, in_channels, out_channels, height, width;
  int kernel_h, kernel_w;
  int stride_h, stride_w;
  LoD x_lod;
  std::vector<int64_t> x_shape, w_shape, out_shape;
  lite::Tensor X_ref, W_ref, Out_ref, Col_ref;
  lite::Tensor X_gpu, W_gpu;
  lite::Tensor X_half, W_half;
  lite::Tensor Out_cpu, Out_gpu;

  operators::VarConv2DParam param;
  std::unique_ptr<KernelContext> ctx;
  cudaStream_t stream;
};

TEST_F(VarConvTest, TestFP32) {
  float_data_init();
  VarConv2DCompute<float, PRECISION(kFloat)> var_conv_2d_kernel;
  var_conv_2d_kernel.SetParam(param);
  var_conv_2d_kernel.SetContext(std::move(ctx));

  for (int i = 0; i < FLAGS_warmup; ++i) {
    var_conv_2d_kernel.Launch();
    cudaDeviceSynchronize();
  }

  auto start = GetCurrentUS();
  var_conv_2d_kernel.PrepareForRun();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    var_conv_2d_kernel.Run();
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
    EXPECT_NEAR(Out_cpu.data<float>()[i], Out_ref.data<float>()[i], 5e-4);
  }
}

TEST_F(VarConvTest, TestFP16) {
  half_data_init();
  VarConv2DCompute<half, PRECISION(kFP16)> var_conv_2d_kernel;
  var_conv_2d_kernel.SetParam(param);
  var_conv_2d_kernel.SetContext(std::move(ctx));

  for (int i = 0; i < FLAGS_warmup; ++i) {
    var_conv_2d_kernel.Launch();
    cudaDeviceSynchronize();
  }

  auto start = GetCurrentUS();
  var_conv_2d_kernel.PrepareForRun();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    var_conv_2d_kernel.Run();
  }
  cudaDeviceSynchronize();
  auto duration = (GetCurrentUS() - start) / 1000.0;
  LOG(INFO) << "fp16, warmup: " << FLAGS_warmup
            << ", repeats: " << FLAGS_repeats << ", spend "
            << duration / FLAGS_repeats << " ms in average.";

  const __half* out_gpu_data = Out_gpu.data<__half>();
  __half* out_cpu_data = Out_cpu.mutable_data<__half>();
  CopySync<TARGET(kCUDA)>(out_cpu_data,
                          out_gpu_data,
                          sizeof(__half) * Out_gpu.numel(),
                          IoDirection::DtoH);

  for (int i = 0; i < Out_cpu.numel(); ++i) {
    float res = static_cast<float>(lite::float16(out_cpu_data[i]));
    float ref = Out_ref.data<float>()[i];
    EXPECT_NEAR(fabs(res - ref) / (ref + 1e-5), 0., 1e-2);
  }
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
