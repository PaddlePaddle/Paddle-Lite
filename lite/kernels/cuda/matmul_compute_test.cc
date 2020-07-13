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

#include "lite/kernels/cuda/matmul_compute.h"

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

class MatMulTest : public ::testing::Test {
 protected:
  MatMulTest()
      : x_trans_(false),
        y_trans_(true),
        alpha_(1.0f),
        x_shape_({4, 1, 2}),
        y_shape_({4, 1, 2}),
        out_shape_({4, 1, 1}) {
    x_ref_.Resize(lite::DDim(x_shape_));
    x_gpu_.Resize(x_ref_.dims());

    y_ref_.Resize(lite::DDim(y_shape_));
    y_gpu_.Resize(y_ref_.dims());

    auto x_ref_data = x_ref_.mutable_data<float>();
    auto y_ref_data = y_ref_.mutable_data<float>();

    // prepare input
    for (int64_t i = 0; i < x_ref_.numel(); i++) {
      x_ref_data[i] = static_cast<float>(1);
    }
    for (int64_t i = 0; i < y_ref_.numel(); i++) {
      y_ref_data[i] = static_cast<float>(1);
    }

    out_ref_.Resize(lite::DDim(out_shape_));
    out_cpu_.Resize(out_ref_.dims());
    out_gpu_.Resize(out_ref_.dims());
    RunBaseLine();

    InitParamAndContext();
  }

  void InitParamAndContext() {
    ctx_.reset(new KernelContext);
    cudaStreamCreate(&stream_);
    auto& context = ctx_->As<CUDAContext>();
    context.SetExecStream(stream_);
    param_.X = &x_gpu_;
    param_.Y = &y_gpu_;
    param_.transpose_X = x_trans_;
    param_.transpose_Y = y_trans_;
    param_.alpha = alpha_;
    param_.Out = &out_gpu_;
  }

  void InitFloatInput() {
    x_gpu_.Assign<float, lite::DDim, TARGET(kCUDA)>(x_ref_.data<float>(),
                                                    x_gpu_.dims());
    y_gpu_.Assign<float, lite::DDim, TARGET(kCUDA)>(y_ref_.data<float>(),
                                                    y_gpu_.dims());
  }

  void InitHalfInput() {
    x_half_.Resize(x_ref_.dims());
    auto x_half_data = x_half_.mutable_data<half>();
    for (int64_t i = 0; i < x_half_.numel(); ++i) {
      x_half_data[i] = half(lite::float16(x_ref_.data<float>()[i]));
    }
    x_gpu_.Assign<half, lite::DDim, TARGET(kCUDA)>(x_half_data, x_gpu_.dims());
    y_half_.Resize(y_ref_.dims());
    auto y_half_data = y_half_.mutable_data<half>();
    for (int64_t i = 0; i < y_half_.numel(); i++) {
      y_half_data[i] = half(lite::float16(y_ref_.data<float>()[i]));
    }
    y_gpu_.Assign<half, lite::DDim, TARGET(kCUDA)>(y_half_data, y_gpu_.dims());
  }

  void RunBaseLine() {
    auto* out_data = out_ref_.mutable_data<float>();
    for (int64_t i = 0; i < out_ref_.numel(); ++i) {
      out_data[i] = 2;
    }
  }

  bool x_trans_, y_trans_;
  float alpha_;
  std::vector<int64_t> x_shape_, y_shape_, out_shape_;
  lite::Tensor x_ref_, y_ref_, out_ref_;
  lite::Tensor x_gpu_, y_gpu_;
  lite::Tensor x_half_, y_half_;
  lite::Tensor out_cpu_, out_gpu_;

  operators::MatMulParam param_;
  std::unique_ptr<KernelContext> ctx_;
  cudaStream_t stream_;
};

TEST_F(MatMulTest, TestFP32) {
  InitFloatInput();
  MatMulCompute<float, PRECISION(kFloat)> kernel;
  kernel.SetParam(param_);
  kernel.SetContext(std::move(ctx_));

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

  CopySync<TARGET(kCUDA)>(out_cpu_.mutable_data<float>(),
                          out_gpu_.data<float>(),
                          sizeof(float) * out_gpu_.numel(),
                          IoDirection::DtoH);

  for (int i = 0; i < out_gpu_.numel(); ++i) {
    float res = out_cpu_.data<float>()[i];
    float ref = out_ref_.data<float>()[i];
    EXPECT_NEAR(fabs(res - ref) / ref, 0.f, 1e-5);
  }
}

TEST_F(MatMulTest, TestFP16) {
  InitHalfInput();
  MatMulCompute<half, PRECISION(kFP16)> kernel;
  kernel.SetParam(param_);
  kernel.SetContext(std::move(ctx_));

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

  const half* out_gpu_data = out_gpu_.data<half>();
  half* out_cpu_data = out_cpu_.mutable_data<half>();
  CopySync<TARGET(kCUDA)>(out_cpu_data,
                          out_gpu_data,
                          sizeof(half) * out_gpu_.numel(),
                          IoDirection::DtoH);

  for (int i = 0; i < out_gpu_.numel(); ++i) {
    float res = static_cast<float>(lite::float16(out_cpu_data[i]));
    float ref = out_ref_.data<float>()[i];
    EXPECT_NEAR(fabs(res - ref) / (ref + 1e-5), 0., 1e-2);
  }
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
