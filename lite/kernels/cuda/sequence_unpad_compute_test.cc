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

#include "lite/api/test_helper.h"
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
    X_ref_.Resize(lite::DDim(x_shape_));
    X_gpu_.Resize(X_ref_.dims());

    Length_ref_.Resize(
        lite::DDim({static_cast<int64_t>(out_lod_[0].size() - 1)}));
    Length_gpu_.Resize(Length_ref_.dims());

    auto* x_ref_data = X_ref_.mutable_data<float>();
    auto* length_ref_data = Length_ref_.mutable_data<int64_t>();

    // prepare input
    for (int64_t i = 0; i < X_ref_.numel(); i++) {
      x_ref_data[i] = static_cast<float>(i);
    }
    for (size_t i = 0; i < out_lod_[0].size() - 1; ++i) {
      length_ref_data[i] = out_lod_[0][i + 1] - out_lod_[0][i];
    }

    Out_ref_.Resize(lite::DDim(out_shape_));
    Out_ref_.set_lod(out_lod_);
    Out_gpu_.Resize(Out_ref_.dims());
    Out_gpu_.set_lod(Out_ref_.lod());
    Out_cpu_.Resize(Out_ref_.dims());
    Out_cpu_.set_lod(Out_ref_.lod());

    RunBaseLine(&X_ref_, &Length_ref_, &Out_ref_);

    InitParamAndContext();
  }

  void InitParamAndContext() {
    ctx_.reset(new KernelContext);
    cudaStreamCreate(&stream_);
    auto& context = ctx_->As<CUDAContext>();
    context.SetExecStream(stream_);
    param_.X = &X_gpu_;
    param_.Length = &Length_gpu_;
    param_.Out = &Out_gpu_;
  }

  void InitFloatInput() {
    X_gpu_.Assign<float, lite::DDim, TARGET(kCUDA)>(X_ref_.data<float>(),
                                                    X_gpu_.dims());
    Length_gpu_.Assign<int64_t, lite::DDim, TARGET(kCUDA)>(
        Length_ref_.data<int64_t>(), Length_gpu_.dims());
  }

  void InitHalfInput() {}

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

  lite::Tensor X_ref_, Out_ref_, Length_ref_;
  lite::Tensor X_gpu_, Out_gpu_, Length_gpu_;
  lite::Tensor Out_cpu_, Length_cpu_;

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

  CopySync<TARGET(kCUDA)>(Out_cpu_.mutable_data<float>(),
                          Out_gpu_.data<float>(),
                          sizeof(float) * Out_gpu_.numel(),
                          IoDirection::DtoH);
  for (int i = 0; i < Out_gpu_.numel(); ++i) {
    EXPECT_NEAR(Out_cpu_.data<float>()[i], Out_ref_.data<float>()[i], 1e-5);
  }
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
