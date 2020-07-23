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
      : m_(8),
        k_(16),
        n_(64),
        in_num_col_dims_(1),
        act_type_("relu"),
        x_shape_({m_, k_}),
        w_shape_({k_, n_}),
        b_shape_({n_}),
        out_shape_({m_, n_}) {
    x_ref_.Resize(lite::DDim(x_shape_));
    x_gpu_.Resize(lite::DDim(x_shape_));

    w_ref_.Resize(lite::DDim(w_shape_));
    w_gpu_.Resize(lite::DDim(w_shape_));

    b_ref_.Resize(lite::DDim(b_shape_));
    b_gpu_.Resize(lite::DDim(b_shape_));

    auto x_ref_data = x_ref_.mutable_data<float>();
    auto w_ref_data = w_ref_.mutable_data<float>();
    auto b_ref_data = b_ref_.mutable_data<float>();

    // prepare input
    for (int64_t i = 0; i < x_ref_.numel(); i++) {
      x_ref_data[i] = static_cast<float>(i % 10 * 0.2);
    }
    for (int64_t i = 0; i < w_ref_.numel(); i++) {
      w_ref_data[i] = static_cast<float>(i % 10 * 0.2);
    }
    for (int64_t i = 0; i < b_ref_.numel(); i++) {
      b_ref_data[i] = static_cast<float>(i % 10 * 0.2);
    }

    out_ref_.Resize(lite::DDim(out_shape_));
    out_cpu_.Resize(out_ref_.dims());
    out_gpu_.Resize(out_ref_.dims());
    RunBaseLine(&x_ref_, &w_ref_, &b_ref_, &out_ref_);

    InitParamAndContext();
  }

  void InitParamAndContext() {
    ctx_.reset(new KernelContext);
    cudaStreamCreate(&stream_);
    auto& context = ctx_->As<CUDAContext>();
    context.SetExecStream(stream_);
    param_.input = &x_gpu_;
    param_.w = &w_gpu_;
    param_.bias = &b_gpu_;
    param_.in_num_col_dims = in_num_col_dims_;
    param_.activation_type = act_type_;
    param_.output = &out_gpu_;
  }

  void InitFloatInput() {
    x_gpu_.Assign<float, lite::DDim, TARGET(kCUDA)>(x_ref_.data<float>(),
                                                    x_gpu_.dims());
    w_gpu_.Assign<float, lite::DDim, TARGET(kCUDA)>(w_ref_.data<float>(),
                                                    w_gpu_.dims());
    b_gpu_.Assign<float, lite::DDim, TARGET(kCUDA)>(b_ref_.data<float>(),
                                                    b_gpu_.dims());
  }

  void InitHalfInput() {
    x_half_.Resize(lite::DDim(x_shape_));
    auto x_half_data = x_half_.mutable_data<half>();
    for (int64_t i = 0; i < x_half_.numel(); i++) {
      x_half_data[i] = half(lite::float16(x_ref_.data<float>()[i]));
    }
    x_gpu_.Assign<half, lite::DDim, TARGET(kCUDA)>(x_half_data, x_gpu_.dims());
    w_half_.Resize(w_ref_.dims());
    auto w_half_data = w_half_.mutable_data<half>();
    for (int64_t i = 0; i < w_half_.numel(); i++) {
      w_half_data[i] = half(lite::float16(w_ref_.data<float>()[i]));
    }
    w_gpu_.Assign<half, lite::DDim, TARGET(kCUDA)>(w_half_data, w_gpu_.dims());
    b_half_.Resize(b_ref_.dims());
    auto b_half_data = b_half_.mutable_data<half>();
    for (int64_t i = 0; i < b_half_.numel(); i++) {
      b_half_data[i] = half(lite::float16(b_ref_.data<float>()[i]));
    }
    b_gpu_.Assign<half, lite::DDim, TARGET(kCUDA)>(b_half_data, b_gpu_.dims());
  }

  void RunBaseLine(const lite::Tensor* x,
                   const lite::Tensor* w,
                   const lite::Tensor* b,
                   lite::Tensor* out) {
    const float* data_in = x->data<float>();
    const float* bias = b->data<float>();
    const float* weights = w->data<float>();
    float* data_out = out->mutable_data<float>();
    int out_rows = x->dims()[0];
    int in_cols = x->numel() / out_rows;
    int out_cols = w->numel() / in_cols;
    int index_out;
    for (int i = 0; i < out_rows; i++) {
      for (int j = 0; j < out_cols; j++) {
        index_out = i * out_cols + j;
        data_out[index_out] = bias ? bias[j] : 0;
        for (int k = 0; k < in_cols; k++) {
          data_out[index_out] +=
              data_in[i * in_cols + k] * weights[k * out_cols + j];
        }
        if (act_type_ == "relu") {
          data_out[index_out] *= static_cast<int>(data_out[index_out] > 0);
        }
      }
    }
  }

  int m_, k_, n_, in_num_col_dims_;
  std::string act_type_;
  std::vector<int64_t> x_shape_, w_shape_, b_shape_, out_shape_;
  lite::Tensor x_ref_, w_ref_, b_ref_, out_ref_;
  lite::Tensor x_gpu_, w_gpu_, b_gpu_;
  lite::Tensor x_half_, w_half_, b_half_;
  lite::Tensor out_cpu_, out_gpu_;

  operators::FcParam param_;
  std::unique_ptr<KernelContext> ctx_;
  cudaStream_t stream_;
};

TEST_F(FcTest, TestFP32) {
  InitFloatInput();
  FcCompute<float, PRECISION(kFloat)> kernel;
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

TEST_F(FcTest, TestFP16) {
  InitHalfInput();
  FcCompute<half, PRECISION(kFP16)> kernel;
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
    EXPECT_NEAR(fabs(res - ref) / (ref + 1e-5), 0., 2e-2);
  }
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
