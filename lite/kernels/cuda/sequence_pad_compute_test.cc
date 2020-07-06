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

#include "lite/kernels/cuda/sequence_pad_compute.h"

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

class SequencePadTest : public ::testing::Test {
 protected:
  SequencePadTest()
      : batch_(5),
        features_(2),
        padded_length_(3),
        x_lod_({{0, 2, 5}}),
        x_shape_({batch_, features_}),
        pad_value_shape_({features_}),
        out_shape_({static_cast<int64_t>(x_lod_[0].size() - 1),
                    padded_length_,
                    features_}) {
    X_ref_.Resize(lite::DDim(x_shape_));
    X_ref_.set_lod(x_lod_);
    X_gpu_.Resize(X_ref_.dims());

    PadValue_ref_.Resize(lite::DDim(pad_value_shape_));
    PadValue_gpu_.Resize(PadValue_ref_.dims());

    Length_ref_.Resize(
        lite::DDim({static_cast<int64_t>(x_lod_[0].size() - 1)}));
    Length_gpu_.Resize(Length_ref_.dims());

    auto x_ref_data = X_ref_.mutable_data<float>();
    auto pad_value_ref_data = PadValue_ref_.mutable_data<float>();

    // prepare input
    for (int64_t i = 0; i < X_ref_.numel(); i++) {
      x_ref_data[i] = static_cast<float>(i);
    }
    for (int64_t i = 0; i < PadValue_ref_.numel(); i++) {
      pad_value_ref_data[i] = static_cast<float>(i);
    }

    Out_ref_.Resize(lite::DDim(out_shape_));
    Out_gpu_.Resize(Out_ref_.dims());
    Out_cpu_.Resize(Out_ref_.dims());
    RunBaseLine(&X_ref_, &PadValue_ref_, &Out_ref_, &Length_ref_);

    InitParamAndContext();
  }

  void InitParamAndContext() {
    ctx_.reset(new KernelContext);
    cudaStreamCreate(&stream_);
    auto& context = ctx_->As<CUDAContext>();
    context.SetExecStream(stream_);
    param_.X = &X_gpu_;
    param_.PadValue = &PadValue_gpu_;
    param_.Length = &Length_gpu_;
    param_.Out = &Out_gpu_;
    param_.padded_length_ = padded_length_;
  }

  void InitFloatInput() {
    X_gpu_.Assign<float, lite::DDim, TARGET(kCUDA)>(X_ref_.data<float>(),
                                                    X_gpu_.dims());
    X_gpu_.set_lod(X_ref_.lod());
    PadValue_gpu_.Assign<float, lite::DDim, TARGET(kCUDA)>(
        PadValue_ref_.data<float>(), PadValue_gpu_.dims());
  }

  void InitHalfInput() {}

  void RunBaseLine(const lite::Tensor* X,
                   const lite::Tensor* PadValue,
                   lite::Tensor* Out,
                   lite::Tensor* Length) {
    auto* length_data = Length->mutable_data<int64_t>();
    auto* out_data = Out->mutable_data<float>();
    length_data[0] = 2;
    length_data[1] = 3;

    for (size_t i = 0; i < 4; ++i) {
      out_data[i] = i;
    }
    out_data[4] = 0;
    out_data[5] = 1;
    for (size_t i = 4; i < 10; ++i) {
      out_data[2 + i] = i;
    }
  }

  int batch_, features_, padded_length_;
  LoD x_lod_;
  std::vector<int64_t> x_shape_, pad_value_shape_, out_shape_;

  lite::Tensor X_ref_, PadValue_ref_, Out_ref_, Length_ref_;
  lite::Tensor X_gpu_, PadValue_gpu_, Out_gpu_, Length_gpu_;
  lite::Tensor Out_cpu_, Length_cpu_;

  operators::SequencePadParam param_;
  std::unique_ptr<KernelContext> ctx_;
  cudaStream_t stream_;
};

TEST_F(SequencePadTest, fp32) {
  InitFloatInput();
  SequencePadCompute<float, PRECISION(kFloat)> kernel;
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
  CopySync<TARGET(kCUDA)>(Length_cpu_.mutable_data<int64_t>(),
                          Length_gpu_.data<int64_t>(),
                          sizeof(int64_t) * Length_gpu_.numel(),
                          IoDirection::DtoH);
  for (int i = 0; i < Out_gpu_.numel(); ++i) {
    EXPECT_NEAR(Out_cpu_.data<float>()[i], Out_ref_.data<float>()[i], 1e-5);
  }
  for (int i = 0; i < Length_gpu_.numel(); ++i) {
    EXPECT_NEAR(
        Length_cpu_.data<int64_t>()[i], Length_ref_.data<int64_t>()[i], 1e-5);
  }
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
