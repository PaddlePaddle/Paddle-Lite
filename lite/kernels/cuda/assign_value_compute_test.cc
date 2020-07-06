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

#include "lite/kernels/cuda/assign_value_compute.h"

#include <gtest/gtest.h>

#include <functional>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "lite/api/test_helper.h"
#include "lite/backends/cuda/cuda_utils.h"
#include "lite/core/types.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

class AssignValueTest : public ::testing::Test {
 protected:
  AssignValueTest() : dtype_(5), shape_({1}) {
    int num = std::accumulate(
        shape_.begin(), shape_.end(), 1, std::multiplies<int>());
    fp32_values_.resize(num);
    int32_values_.resize(num);
    int64_values_.resize(num);
    bool_values_.resize(num);
    for (int i = 0; i < num; ++i) {
      fp32_values_[i] = i + 5;
      int32_values_[i] = i;
      int64_values_[i] = i;
      bool_values_[i] = i;
    }
    std::vector<int64_t> out_shape(shape_.size(), 0);
    for (size_t i = 0; i < shape_.size(); ++i) out_shape[i] = shape_[i];
    Out_ref_.Resize(lite::DDim(out_shape));
    Out_gpu_.Resize(Out_ref_.dims());
    Out_cpu_.Resize(Out_ref_.dims());

    RunBaseLine(&Out_ref_);

    InitParamAndContext();
  }

  void InitParamAndContext() {
    ctx_.reset(new KernelContext);
    cudaStreamCreate(&stream_);
    auto& context = ctx_->As<CUDAContext>();
    context.SetExecStream(stream_);
    param_.shape_ = shape_;
    param_.dtype_ = dtype_;
    param_.fp32_values_ = fp32_values_;
    param_.int32_values_ = int32_values_;
    param_.int64_values_ = int64_values_;
    param_.bool_values_ = bool_values_;
    param_.Out = &Out_gpu_;
  }

  void InitFloatInput() {}

  void InitHalfInput() {}

  void RunBaseLine(lite::Tensor* Out) {
    if (dtype_ == static_cast<int>(lite::core::FluidType::INT32)) {
      for (size_t i = 0; i < int32_values_.size(); ++i) {
        Out->mutable_data<int>()[i] = int32_values_[i];
      }
    } else if (dtype_ == static_cast<int>(lite::core::FluidType::FP32)) {
      for (size_t i = 0; i < fp32_values_.size(); ++i) {
        Out->mutable_data<float>()[i] = fp32_values_[i];
      }
    } else if (dtype_ == static_cast<int>(lite::core::FluidType::INT64)) {
      for (size_t i = 0; i < int64_values_.size(); ++i) {
        Out->mutable_data<int64_t>()[i] = int64_values_[i];
      }
    } else if (dtype_ == static_cast<bool>(lite::core::FluidType::BOOL)) {
      for (size_t i = 0; i < bool_values_.size(); ++i) {
        Out->mutable_data<bool>()[i] = bool_values_[i];
      }
    } else {
      LOG(FATAL) << "Unsupported dtype_ for assign_value_op:" << dtype_;
    }
  }

  int dtype_;
  std::vector<int> shape_;
  std::vector<float> fp32_values_;
  std::vector<int> int32_values_;
  std::vector<int64_t> int64_values_;
  std::vector<int> bool_values_;

  lite::Tensor Out_ref_;
  lite::Tensor Out_gpu_;
  lite::Tensor Out_cpu_;

  operators::AssignValueParam param_;
  std::unique_ptr<KernelContext> ctx_;
  cudaStream_t stream_;
};

TEST_F(AssignValueTest, fp32) {
  InitFloatInput();
  AssignValueCompute kernel;
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
