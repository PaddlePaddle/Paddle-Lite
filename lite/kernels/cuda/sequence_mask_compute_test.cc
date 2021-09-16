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

#include "lite/kernels/cuda/sequence_mask_compute.h"

#include <gtest/gtest.h>

#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "lite/api/test/test_helper.h"
#include "lite/backends/cuda/cuda_utils.h"
#include "lite/utils/float16.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

class SequenceMaskTest : public ::testing::Test {
 protected:
  SequenceMaskTest()
      : maxlen_(4),
        out_dtype_(5),
        x_data_({3, 2, 1, 0}),
        out_shape_({static_cast<int64_t>(x_data_.size()), maxlen_}) {
    x_ref_.Resize(lite::DDim({static_cast<int64_t>(x_data_.size())}));
    x_gpu_.Resize(x_ref_.dims());

    auto* x_ref_data = x_ref_.mutable_data<int64_t>();

    // prepare input
    for (size_t i = 0; i < x_data_.size(); i++) {
      x_ref_data[i] = x_data_[i];
    }

    out_ref_.Resize(lite::DDim(out_shape_));
    out_gpu_.Resize(out_ref_.dims());
    out_cpu_.Resize(out_ref_.dims());
    RunBaseLine(&x_ref_, &out_ref_);

    InitParamAndContext();
  }

  void InitParamAndContext() {
    ctx_.reset(new KernelContext);
    cudaStreamCreate(&stream_);
    auto& context = ctx_->As<CUDAContext>();
    context.SetExecStream(stream_);
    param_.X = &x_gpu_;
    param_.Y = &out_gpu_;
    param_.maxlen = maxlen_;
    param_.out_dtype = out_dtype_;
  }

  void InitFloatInput() {
    x_gpu_.Assign<int64_t, lite::DDim, TARGET(kCUDA)>(x_ref_.data<int64_t>(),
                                                      x_gpu_.dims());
  }

  void InitHalfInput() {
    x_gpu_.Assign<int64_t, lite::DDim, TARGET(kCUDA)>(x_ref_.data<int64_t>(),
                                                      x_gpu_.dims());
  }

  void RunBaseLine(const lite::Tensor* x, lite::Tensor* out) {
    auto* out_data = out->mutable_data<float>();

    for (size_t i = 0; i < x_data_.size(); ++i) {
      for (int j = 0; j < maxlen_; ++j) {
        out_data[i * maxlen_ + j] = j < x_data_[i] ? 1 : 0;
      }
    }
  }

  int maxlen_, out_dtype_;
  std::vector<int64_t> x_data_, out_shape_;

  lite::Tensor x_ref_, out_ref_;
  lite::Tensor x_gpu_, out_gpu_;
  lite::Tensor out_cpu_;

  operators::SequenceMaskParam param_;
  std::unique_ptr<KernelContext> ctx_;
  cudaStream_t stream_;
};

TEST_F(SequenceMaskTest, fp32) {
  InitFloatInput();
  SequenceMaskCompute<float, PRECISION(kFloat)> kernel;
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
    EXPECT_NEAR(out_cpu_.data<float>()[i], out_ref_.data<float>()[i], 1e-5);
  }
}

TEST_F(SequenceMaskTest, TestFP16) {
  InitHalfInput();
  SequenceMaskCompute<half, PRECISION(kFP16)> kernel;
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
