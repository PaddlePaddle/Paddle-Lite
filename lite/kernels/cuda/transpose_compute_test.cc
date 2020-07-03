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

#include "lite/kernels/cuda/transpose_compute.h"

#include <gtest/gtest.h>
#include <memory>
#include <utility>
#include <vector>

#include "lite/api/test_helper.h"
#include "lite/backends/cuda/cuda_utils.h"
#include "lite/utils/float16.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

namespace {

#define IN(n, c, h, w)                                 \
  input_data[w + h * input_w + c * input_h * input_w + \
             n * input_c * input_h * input_w]
#define OUT(n, c, h, w)                                    \
  output_data[w + h * output_w + c * output_h * output_w + \
              n * output_c * output_h * output_w]
void nchw2nhwc_ref(lite::Tensor* input,
                   lite::Tensor* output,
                   const std::vector<int> axies) {
  auto* input_data = input->data<float>();
  auto* output_data = output->mutable_data<float>();

  int input_n = input->dims()[0];
  int input_c = input->dims()[1];
  int input_h = input->dims()[2];
  int input_w = input->dims()[3];
  int output_c = output->dims()[1];
  int output_h = output->dims()[2];
  int output_w = output->dims()[3];

  for (int n = 0; n < input_n; ++n) {
    for (int c = 0; c < input_c; ++c) {
      for (int h = 0; h < input_h; ++h) {
        for (int w = 0; w < input_w; ++w) {
          OUT(n, h, w, c) = IN(n, c, h, w);
        }
      }
    }
  }
}
#undef IN
#undef OUT

#define IN(n, h, w, c)                                 \
  input_data[c + w * input_c + h * input_w * input_c + \
             n * input_h * input_w * input_c]
#define OUT(n, h, w, c)                                    \
  output_data[c + w * output_c + h * output_w * output_c + \
              n * output_h * output_w * output_c]
void nhwc2nchw_ref(lite::Tensor* input,
                   lite::Tensor* output,
                   const std::vector<int> axies) {
  auto* input_data = input->data<float>();
  auto* output_data = output->mutable_data<float>();

  int input_n = input->dims()[0];
  int input_h = input->dims()[1];
  int input_w = input->dims()[2];
  int input_c = input->dims()[3];
  int output_h = output->dims()[1];
  int output_w = output->dims()[2];
  int output_c = output->dims()[3];

  for (int n = 0; n < input_n; ++n) {
    for (int c = 0; c < input_c; ++c) {
      for (int h = 0; h < input_h; ++h) {
        for (int w = 0; w < input_w; ++w) {
          OUT(n, c, h, w) = IN(n, h, w, c);
        }
      }
    }
  }
}

void transpose_ref(const lite::Tensor* input,
                   lite::Tensor* output,
                   const std::vector<int> axes) {
  auto* input_data = input->data<float>();
  auto* output_data = output->mutable_data<float>();

  int ndim = input->dims().size();
  auto dims = input->dims();
  std::vector<int> strides(ndim, 0);
  std::vector<int> buf(ndim, 0);
  int cur_stride = 1;
  for (int i = ndim - 1; i >= 0; --i) {
    buf[i] = cur_stride;
    cur_stride *= dims[i];
  }
  for (int i = 0; i < ndim; ++i) {
    strides[i] = buf[axes[i]];
  }

  auto y_dims = output->dims();
  int size = input->dims().production();
  for (int i = 0; i < size; ++i) {
    int idx = 0;
    int v = i;
    for (int j = ndim - 1; j >= 0; --j) {
      idx += v % y_dims[j] * strides[j];
      v /= y_dims[j];
    }
    output_data[i] = input_data[idx];
  }
}
}  // namespace

TEST(transpose_nchw, normal) {
  TransposeCompute<float, PRECISION(kFloat)> transpose_kernel;
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  auto& context = ctx->As<CUDAContext>();

  operators::TransposeParam param;

  lite::Tensor x, x_cpu, x_ref;
  lite::Tensor out, out_cpu, out_ref;

  int N = 5, C = 6, H = 7, W = 8;
  std::vector<int> axes({0, 2, 3, 1});
  x.Resize({N, C, H, W});
  out.Resize({N, H, W, C});

  x_cpu.Resize({N, C, H, W});
  out_cpu.Resize({N, H, W, C});

  x_ref.Resize({N, C, H, W});
  out_ref.Resize({N, H, W, C});

  auto* x_cpu_data = x_cpu.mutable_data<float>();
  auto* out_cpu_data = out_cpu.mutable_data<float>();
  auto* x_ref_data = x_ref.mutable_data<float>();

  for (int i = 0; i < x_cpu.numel(); ++i) {
    x_cpu_data[i] = i + 1;
    x_ref_data[i] = i + 1;
  }

  x.Assign<float, lite::DDim, TARGET(kCUDA)>(x_cpu_data, x_cpu.dims());

  param.x = &x;
  param.output = &out;
  param.axis = axes;
  transpose_kernel.SetParam(param);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  context.SetExecStream(stream);
  transpose_kernel.SetContext(std::move(ctx));
  transpose_kernel.Launch();
  cudaDeviceSynchronize();
  auto* out_data = out.mutable_data<float>(TARGET(kCUDA));
  CopySync<TARGET(kCUDA)>(
      out_cpu_data, out_data, sizeof(float) * out.numel(), IoDirection::DtoH);
  nchw2nhwc_ref(&x_ref, &out_ref, axes);
  auto* out_ref_data = out_ref.mutable_data<float>();
  // transpose_ref(&x_ref, &out_ref, axes);
  for (int i = 0; i < out.numel(); i++) {
    EXPECT_NEAR(out_cpu_data[i], out_ref_data[i], 1e-5);
  }
}

TEST(transpose_nhwc, normal) {
  TransposeCompute<float, PRECISION(kFloat)> transpose_kernel;
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  auto& context = ctx->As<CUDAContext>();

  operators::TransposeParam param;

  lite::Tensor x, x_cpu, x_ref;
  lite::Tensor out, out_cpu, out_ref;

  int N = 5, C = 6, H = 7, W = 8;
  std::vector<int> axes({0, 3, 1, 2});
  x.Resize({N, H, W, C});
  out.Resize({N, C, H, W});

  x_cpu.Resize({N, H, W, C});
  out_cpu.Resize({N, C, H, W});

  x_ref.Resize({N, H, W, C});
  out_ref.Resize({N, C, H, W});

  auto* x_cpu_data = x_cpu.mutable_data<float>();
  auto* out_cpu_data = out_cpu.mutable_data<float>();
  auto* x_ref_data = x_ref.mutable_data<float>();

  for (int i = 0; i < x_cpu.numel(); ++i) {
    x_cpu_data[i] = i + 1;
    x_ref_data[i] = i + 1;
  }

  x.Assign<float, lite::DDim, TARGET(kCUDA)>(x_cpu_data, x_cpu.dims());
  param.x = &x;
  param.output = &out;
  param.axis = axes;
  transpose_kernel.SetParam(param);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  context.SetExecStream(stream);
  transpose_kernel.SetContext(std::move(ctx));
  transpose_kernel.Launch();
  cudaDeviceSynchronize();
  auto* out_data = out.mutable_data<float>(TARGET(kCUDA));
  CopySync<TARGET(kCUDA)>(
      out_cpu_data, out_data, sizeof(float) * out.numel(), IoDirection::DtoH);
  nhwc2nchw_ref(&x_ref, &out_ref, axes);
  // transpose_ref(&x_ref, &out_ref, axes);
  auto* out_ref_data = out_ref.mutable_data<float>();
  for (int i = 0; i < out.numel(); i++) {
    EXPECT_NEAR(out_cpu_data[i], out_ref_data[i], 1e-5);
  }
}

class TransposeTest : public ::testing::Test {
 protected:
  TransposeTest()
      : C(3),
        H(128),
        W(64),
        axes({1, 2, 0}),
        x_shape({C, H, W}),
        out_shape({H, W, C}) {
    X_ref.Resize(lite::DDim(x_shape));
    X_gpu.Resize(X_ref.dims());

    auto x_ref_data = X_ref.mutable_data<float>();

    // prepare input
    for (int64_t i = 0; i < X_ref.numel(); i++) {
      x_ref_data[i] = static_cast<float>(i);
    }

    Out_ref.Resize(lite::DDim(out_shape));
    Out_gpu.Resize(Out_ref.dims());
    Out_cpu.Resize(Out_ref.dims());
    cpu_base(&X_ref, &Out_ref);

    device_init();
  }

  void device_init() {
    ctx.reset(new KernelContext);
    cudaStreamCreate(&stream);
    auto& context = ctx->As<CUDAContext>();
    context.SetExecStream(stream);
    param.x = &X_gpu;
    param.output = &Out_gpu;
    param.axis = axes;
  }

  void float_data_init() {
    X_gpu.Assign<float, lite::DDim, TARGET(kCUDA)>(X_ref.data<float>(),
                                                   X_gpu.dims());
  }

  void half_data_init() {
    X_half.Resize(lite::DDim(X_ref.dims()));
    auto x_half_data = X_half.mutable_data<half>();
    for (int64_t i = 0; i < X_half.numel(); i++) {
      x_half_data[i] = half(lite::float16(X_ref.data<float>()[i]));
    }
    X_gpu.Assign<half, lite::DDim, TARGET(kCUDA)>(x_half_data, X_gpu.dims());
  }

  void cpu_base(const lite::Tensor* X, lite::Tensor* Out) {
    transpose_ref(X, Out, axes);
  }

  int C, H, W;
  std::vector<int> axes;
  std::vector<int64_t> x_shape, out_shape;

  lite::Tensor X_ref, Out_ref;
  lite::Tensor X_gpu, Out_gpu;
  lite::Tensor X_half;
  lite::Tensor Out_cpu;

  operators::TransposeParam param;
  std::unique_ptr<KernelContext> ctx;
  cudaStream_t stream;
};

TEST_F(TransposeTest, fp32) {
  float_data_init();
  TransposeCompute<float, PRECISION(kFloat)> kernel;
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
    EXPECT_NEAR(Out_cpu.data<float>()[i], Out_ref.data<float>()[i], 1e-5);
  }
}

TEST_F(TransposeTest, TestFP16) {
  half_data_init();
  TransposeCompute<half, PRECISION(kFP16)> kernel;
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
  LOG(INFO) << "fp16, warmup: " << FLAGS_warmup
            << ", repeats: " << FLAGS_repeats << ", spend "
            << duration / FLAGS_repeats << " ms in average.";

  const half* out_gpu_data = Out_gpu.data<half>();
  half* out_cpu_data = Out_cpu.mutable_data<half>();
  CopySync<TARGET(kCUDA)>(out_cpu_data,
                          out_gpu_data,
                          sizeof(half) * Out_gpu.numel(),
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
