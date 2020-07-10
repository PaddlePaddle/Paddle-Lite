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
    x_ref_.Resize(lite::DDim(x_shape_));
    x_ref_.set_lod(x_lod_);
    x_gpu_.Resize(x_ref_.dims());

    pad_value_ref_.Resize(lite::DDim(pad_value_shape_));
    pad_value_gpu_.Resize(pad_value_ref_.dims());

    length_ref_.Resize(
        lite::DDim({static_cast<int64_t>(x_lod_[0].size() - 1)}));
    length_gpu_.Resize(length_ref_.dims());
    length_cpu_.Resize(length_ref_.dims());

    auto x_ref_data = x_ref_.mutable_data<float>();
    auto pad_value_ref_data = pad_value_ref_.mutable_data<float>();

    for (int64_t i = 0; i < x_ref_.numel(); i++) {
      x_ref_data[i] = static_cast<float>(i);
    }
    for (int64_t i = 0; i < pad_value_ref_.numel(); i++) {
      pad_value_ref_data[i] = static_cast<float>(i);
    }

    out_ref_.Resize(lite::DDim(out_shape_));
    out_gpu_.Resize(out_ref_.dims());
    out_cpu_.Resize(out_ref_.dims());
    RunBaseLine(&x_ref_, &pad_value_ref_, &out_ref_, &length_ref_);

    InitParamAndContext();
  }

  void InitParamAndContext() {
    ctx_.reset(new KernelContext);
    cudaStreamCreate(&stream_);
    auto& context = ctx_->As<CUDAContext>();
    context.SetExecStream(stream_);
    param_.X = &x_gpu_;
    param_.PadValue = &pad_value_gpu_;
    param_.Length = &length_gpu_;
    param_.Out = &out_gpu_;
    param_.padded_length = padded_length_;
  }

  void InitFloatInput() {
    x_gpu_.Assign<float, lite::DDim, TARGET(kCUDA)>(x_ref_.data<float>(),
                                                    x_gpu_.dims());
    x_gpu_.set_lod(x_ref_.lod());
    pad_value_gpu_.Assign<float, lite::DDim, TARGET(kCUDA)>(
        pad_value_ref_.data<float>(), pad_value_gpu_.dims());
  }

  void InitHalfInput() {
    x_half_.Resize(lite::DDim(x_shape_));
    auto x_half_data = x_half_.mutable_data<half>();
    for (int64_t i = 0; i < x_half_.numel(); i++) {
      x_half_data[i] = half(lite::float16(x_ref_.data<float>()[i]));
    }
    x_gpu_.Assign<half, lite::DDim, TARGET(kCUDA)>(x_half_data, x_gpu_.dims());
    x_gpu_.set_lod(x_ref_.lod());
    pad_value_half_.Resize(pad_value_ref_.dims());
    auto pad_value_half_data = pad_value_half_.mutable_data<half>();
    for (int64_t i = 0; i < pad_value_half_.numel(); i++) {
      pad_value_half_data[i] =
          half(lite::float16(pad_value_ref_.data<float>()[i]));
    }
    pad_value_gpu_.Assign<half, lite::DDim, TARGET(kCUDA)>(
        pad_value_half_data, pad_value_gpu_.dims());
  }

  void RunBaseLine(const lite::Tensor* x,
                   const lite::Tensor* pad_value,
                   lite::Tensor* out,
                   lite::Tensor* length) {
    auto* length_data = length->mutable_data<int64_t>();
    auto* out_data = out->mutable_data<float>();
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

  lite::Tensor x_ref_, pad_value_ref_, out_ref_, length_ref_;
  lite::Tensor x_gpu_, pad_value_gpu_, out_gpu_, length_gpu_;
  lite::Tensor x_half_, pad_value_half_;
  lite::Tensor out_cpu_, length_cpu_;

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

  CopySync<TARGET(kCUDA)>(out_cpu_.mutable_data<float>(),
                          out_gpu_.data<float>(),
                          sizeof(float) * out_gpu_.numel(),
                          IoDirection::DtoH);
  CopySync<TARGET(kCUDA)>(length_cpu_.mutable_data<int64_t>(),
                          length_gpu_.data<int64_t>(),
                          sizeof(int64_t) * length_gpu_.numel(),
                          IoDirection::DtoH);
  for (int i = 0; i < out_gpu_.numel(); ++i) {
    EXPECT_NEAR(out_cpu_.data<float>()[i], out_ref_.data<float>()[i], 1e-5);
  }
  for (int i = 0; i < length_gpu_.numel(); ++i) {
    EXPECT_NEAR(
        length_cpu_.data<int64_t>()[i], length_ref_.data<int64_t>()[i], 1e-5);
  }
}

TEST_F(SequencePadTest, TestFP16) {
  InitHalfInput();
  SequencePadCompute<half, PRECISION(kFP16)> kernel;
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
  const int64_t* length_gpu_data = length_gpu_.data<int64_t>();
  int64_t* length_cpu_data = length_cpu_.mutable_data<int64_t>();
  CopySync<TARGET(kCUDA)>(out_cpu_data,
                          out_gpu_data,
                          sizeof(half) * out_gpu_.numel(),
                          IoDirection::DtoH);
  CopySync<TARGET(kCUDA)>(length_cpu_data,
                          length_gpu_data,
                          sizeof(int64_t) * length_gpu_.numel(),
                          IoDirection::DtoH);
  for (int i = 0; i < out_gpu_.numel(); ++i) {
    float res = static_cast<float>(lite::float16(out_cpu_data[i]));
    float ref = out_ref_.data<float>()[i];
    EXPECT_NEAR(fabs(res - ref) / (ref + 1e-5), 0., 1e-2);
  }
  for (int i = 0; i < length_gpu_.numel(); ++i) {
    EXPECT_NEAR(
        length_cpu_.data<int64_t>()[i], length_ref_.data<int64_t>()[i], 1e-5);
  }
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
