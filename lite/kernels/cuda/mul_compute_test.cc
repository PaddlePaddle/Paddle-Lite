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

#include "lite/kernels/cuda/mul_compute.h"
#include <gtest/gtest.h>
#include <memory>
#include <utility>
#include <vector>
#include "lite/api/test_helper.h"
#include "lite/utils/float16.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

class MulTest : public ::testing::Test {
 protected:
  MulTest()
      : m_(2),
        k_(3),
        n_(4),
        x_shape_({m_, k_}),
        y_shape_({k_, n_}),
        out_shape_({m_, n_}) {
    x_gpu_.Resize(lite::DDim(x_shape_));
    x_ref_.Resize(lite::DDim(x_shape_));

    y_gpu_.Resize(lite::DDim(y_shape_));
    y_ref_.Resize(lite::DDim(y_shape_));

    auto x_ref_data = x_ref_.mutable_data<float>();
    auto y_ref_data = y_ref_.mutable_data<float>();

    // prepare input
    for (int64_t i = 0; i < x_ref_.numel(); i++) {
      x_ref_data[i] = static_cast<float>(i % 10 * 0.2);
    }
    for (int64_t i = 0; i < y_ref_.numel(); i++) {
      y_ref_data[i] = static_cast<float>(i % 10 * 0.2);
    }

    out_ref_.Resize(lite::DDim(out_shape_));
    out_cpu_.Resize(lite::DDim(out_shape_));
    out_gpu_.Resize(lite::DDim(out_shape_));
    RunBaseLine(&x_ref_, &y_ref_, &out_ref_);

    InitParamAndContext();
  }

  void InitParamAndContext() {
    ctx_.reset(new KernelContext);
    cudaStreamCreate(&stream_);
    auto& context = ctx_->As<CUDAContext>();
    context.SetExecStream(stream_);
    param_.x = &x_gpu_;
    param_.y = &y_gpu_;
    param_.output = &out_gpu_;
  }

  void InitFloatInput() {
    x_gpu_.Assign<float, lite::DDim, TARGET(kCUDA)>(x_ref_.data<float>(),
                                                    x_gpu_.dims());
    y_gpu_.Assign<float, lite::DDim, TARGET(kCUDA)>(y_ref_.data<float>(),
                                                    y_gpu_.dims());
  }

  void InitHalfInput() {
    x_half_.Resize(lite::DDim(x_ref_.dims()));
    auto x_half_data = x_half_.mutable_data<half>();
    for (int64_t i = 0; i < x_half_.numel(); i++) {
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

  void RunBaseLine(const lite::Tensor* x,
                   const lite::Tensor* w,
                   lite::Tensor* out) {
    const float* data_in = x->data<float>();
    const float* weights = w->data<float>();
    float* data_out = out->mutable_data<float>();
    int out_rows = x->dims()[0];
    int in_cols = x->numel() / out_rows;
    int out_cols = w->numel() / in_cols;
    int index_out;
    for (int i = 0; i < out_rows; i++) {
      for (int j = 0; j < out_cols; j++) {
        index_out = i * out_cols + j;
        data_out[index_out] = 0;
        for (int k = 0; k < in_cols; k++) {
          data_out[index_out] +=
              data_in[i * in_cols + k] * weights[k * out_cols + j];
        }
      }
    }
  }

  int m_, k_, n_;
  std::vector<int64_t> x_shape_, y_shape_, out_shape_;
  lite::Tensor x_ref_, y_ref_, out_ref_;
  lite::Tensor x_gpu_, y_gpu_;
  lite::Tensor x_half_, y_half_;
  lite::Tensor out_cpu_, out_gpu_;

  operators::MulParam param_;
  std::unique_ptr<KernelContext> ctx_;
  cudaStream_t stream_;
};

TEST_F(MulTest, TestFP32) {
  InitFloatInput();
  MulCompute<float, PRECISION(kFloat)> mul_kernel;
  mul_kernel.SetParam(param_);
  mul_kernel.SetContext(std::move(ctx_));

  for (int i = 0; i < FLAGS_warmup; ++i) {
    mul_kernel.Launch();
    cudaDeviceSynchronize();
  }

  auto start = GetCurrentUS();
  mul_kernel.PrepareForRun();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    mul_kernel.Run();
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
    EXPECT_NEAR(fabs(res - ref) / (ref + 1e-5), 0., 1e-4);
  }
}

TEST_F(MulTest, TestFP16) {
  InitHalfInput();
  MulCompute<half, PRECISION(kFP16)> mul_kernel;
  mul_kernel.SetParam(param_);
  mul_kernel.SetContext(std::move(ctx_));

  for (int i = 0; i < FLAGS_warmup; ++i) {
    mul_kernel.Launch();
    cudaDeviceSynchronize();
  }

  auto start = GetCurrentUS();
  mul_kernel.PrepareForRun();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    mul_kernel.Run();
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

  for (int i = 0; i < out_cpu_.numel(); ++i) {
    float res = static_cast<float>(lite::float16(out_cpu_data[i]));
    float ref = out_ref_.data<float>()[i];
    EXPECT_NEAR(fabs(res - ref) / (ref + 1e-5), 0., 1e-2);
  }
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
