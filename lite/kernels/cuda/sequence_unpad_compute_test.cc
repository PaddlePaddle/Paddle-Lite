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

#include "lite/kernels/cuda/sequence_unpad_compute.h"

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

class SequenceUnpadTest : public ::testing::Test {
 protected:
  SequenceUnpadTest()
      : batch_(5),
        features_(2),
        padded_length_(3),
        out_lod_({{0, 2, 5}}),
        x_shape_({static_cast<int64_t>(out_lod_[0].size() - 1),
                  padded_length_,
                  features_}),
        out_shape_({batch_, features_}) {
    x_ref_.Resize(lite::DDim(x_shape_));
    x_gpu_.Resize(x_ref_.dims());

    length_ref_.Resize(
        lite::DDim({static_cast<int64_t>(out_lod_[0].size() - 1)}));
    length_gpu_.Resize(length_ref_.dims());

    auto* x_ref_data = x_ref_.mutable_data<float>();
    auto* length_ref_data = length_ref_.mutable_data<int64_t>();

    // prepare input
    for (int64_t i = 0; i < x_ref_.numel(); i++) {
      x_ref_data[i] = static_cast<float>(i);
    }
    for (size_t i = 0; i < out_lod_[0].size() - 1; ++i) {
      length_ref_data[i] = out_lod_[0][i + 1] - out_lod_[0][i];
    }

    out_ref_.Resize(lite::DDim(out_shape_));
    out_ref_.set_lod(out_lod_);
    out_gpu_.Resize(out_ref_.dims());
    out_gpu_.set_lod(out_ref_.lod());
    out_cpu_.Resize(out_ref_.dims());
    out_cpu_.set_lod(out_ref_.lod());

    RunBaseLine(&x_ref_, &length_ref_, &out_ref_);

    InitParamAndContext();
  }

  void InitParamAndContext() {
    ctx_.reset(new KernelContext);
    cudaStreamCreate(&stream_);
    auto& context = ctx_->As<CUDAContext>();
    context.SetExecStream(stream_);
    param_.X = &x_gpu_;
    param_.Length = &length_gpu_;
    param_.Out = &out_gpu_;
  }

  void InitFloatInput() {
    x_gpu_.Assign<float, lite::DDim, TARGET(kCUDA)>(x_ref_.data<float>(),
                                                    x_gpu_.dims());
    length_gpu_.Assign<int64_t, lite::DDim, TARGET(kCUDA)>(
        length_ref_.data<int64_t>(), length_gpu_.dims());
  }

  void InitHalfInput() {
    x_half_.Resize(lite::DDim(x_shape_));
    auto x_half_data = x_half_.mutable_data<half>();
    for (int64_t i = 0; i < x_half_.numel(); i++) {
      x_half_data[i] = half(lite::float16(x_ref_.data<float>()[i]));
    }
    x_gpu_.Assign<half, lite::DDim, TARGET(kCUDA)>(x_half_data, x_gpu_.dims());
    length_gpu_.Assign<int64_t, lite::DDim, TARGET(kCUDA)>(
        length_ref_.data<int64_t>(), length_gpu_.dims());
  }

  void RunBaseLine(const lite::Tensor* X,
                   const lite::Tensor* Length,
                   lite::Tensor* Out) {
    auto* out_data = Out->mutable_data<float>();

    for (size_t i = 0; i < 4; ++i) {
      out_data[i] = i;
    }
    for (size_t i = 6; i < 12; ++i) {
      out_data[i - 2] = i;
    }
  }

  int batch_, features_, padded_length_;
  LoD out_lod_;
  std::vector<int64_t> x_shape_, out_shape_;

  lite::Tensor x_ref_, out_ref_, length_ref_;
  lite::Tensor x_gpu_, out_gpu_, length_gpu_;
  lite::Tensor x_half_;
  lite::Tensor out_cpu_, length_cpu_;

  operators::SequencePadParam param_;
  std::unique_ptr<KernelContext> ctx_;
  cudaStream_t stream_;
};

TEST_F(SequenceUnpadTest, fp32) {
  InitFloatInput();
  SequenceUnpadCompute<float, PRECISION(kFloat)> kernel;
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

TEST_F(SequenceUnpadTest, TestFP16) {
  InitHalfInput();
  SequenceUnpadCompute<half, PRECISION(kFP16)> kernel;
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
