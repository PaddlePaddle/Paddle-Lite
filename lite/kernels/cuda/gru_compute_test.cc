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

#include "lite/kernels/cuda/gru_compute.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "lite/api/test/test_helper.h"
#include "lite/utils/float16.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

class GRUTest : public ::testing::Test {
 protected:
  GRUTest()
      : batch_(12),
        frame_size_(128),
        activation_("tanh"),
        gate_activation_("sigmoid"),
        is_reverse_(false),
        origin_mode_(false),
        x_shape_({batch_, frame_size_ * 3}),
        w_shape_({frame_size_, frame_size_ * 3}),
        out_shape_({batch_, frame_size_}),
        lod_({{0, 4, 9, 12}}) {
    x_ref_.Resize(lite::DDim(x_shape_));
    x_gpu_.Resize(lite::DDim(x_shape_));
    x_ref_.set_lod(lod_);

    w_ref_.Resize(lite::DDim(w_shape_));
    w_gpu_.Resize(lite::DDim(w_shape_));

    auto x_ref_data = x_ref_.mutable_data<float>();
    auto w_ref_data = w_ref_.mutable_data<float>();

    for (int64_t i = 0; i < x_ref_.numel(); i++) {
      x_ref_data[i] = static_cast<float>(i % 10 * 0.2);
    }
    for (int64_t i = 0; i < w_ref_.numel(); i++) {
      w_ref_data[i] = static_cast<float>(i % 10 * 0.2);
    }

    out_ref_.Resize(lite::DDim(out_shape_));
    out_cpu_.Resize(out_ref_.dims());
    out_gpu_.Resize(out_ref_.dims());
    batch_gate_gpu_.Resize(lite::DDim(x_shape_));
    batch_hidden_gpu_.Resize(lite::DDim(out_shape_));
    batch_reset_hidden_gpu_.Resize(lite::DDim(out_shape_));
    RunBaseLine();

    InitParamAndContext();
  }

  void InitParamAndContext() {
    ctx_.reset(new KernelContext);
    cudaStreamCreate(&stream_);
    auto& context = ctx_->As<CUDAContext>();
    context.SetExecStream(stream_);
    param_.input = &x_gpu_;
    param_.weight = &w_gpu_;
    param_.gate_activation = gate_activation_;
    param_.activation = activation_;
    param_.is_reverse = is_reverse_;
    param_.origin_mode = origin_mode_;
    param_.hidden = &out_gpu_;
    param_.batch_gate = &batch_gate_gpu_;
    param_.batch_reset_hidden_prev = &batch_reset_hidden_gpu_;
    param_.batch_hidden = &batch_hidden_gpu_;
  }

  void InitFloatInput() {
    x_gpu_.Assign<float, lite::DDim, TARGET(kCUDA)>(x_ref_.data<float>(),
                                                    x_gpu_.dims());
    x_gpu_.set_lod(x_ref_.lod());
    w_gpu_.Assign<float, lite::DDim, TARGET(kCUDA)>(w_ref_.data<float>(),
                                                    w_gpu_.dims());
  }

  void InitHalfInput() {
    x_half_.Resize(lite::DDim(x_shape_));
    auto x_half_data = x_half_.mutable_data<half>();
    for (int64_t i = 0; i < x_half_.numel(); i++) {
      x_half_data[i] = half(lite::float16(x_ref_.data<float>()[i]));
    }
    x_gpu_.Assign<half, lite::DDim, TARGET(kCUDA)>(x_half_data, x_gpu_.dims());
    x_gpu_.set_lod(x_ref_.lod());
    w_half_.Resize(w_ref_.dims());
    auto w_half_data = w_half_.mutable_data<half>();
    for (int64_t i = 0; i < w_half_.numel(); i++) {
      w_half_data[i] = half(lite::float16(w_ref_.data<float>()[i]));
    }
    w_gpu_.Assign<half, lite::DDim, TARGET(kCUDA)>(w_half_data, w_gpu_.dims());
  }

  void RunBaseLine() {}

  int batch_, frame_size_;
  std::string activation_, gate_activation_;
  bool is_reverse_, origin_mode_;
  std::vector<int64_t> x_shape_, w_shape_, out_shape_;
  LoD lod_;
  lite::Tensor x_ref_, w_ref_, out_ref_;
  lite::Tensor x_gpu_, w_gpu_;
  lite::Tensor x_half_, w_half_;
  lite::Tensor batch_gate_gpu_;
  lite::Tensor batch_hidden_gpu_;
  lite::Tensor batch_reset_hidden_gpu_;
  lite::Tensor out_cpu_, out_gpu_;

  operators::GRUParam param_;
  std::unique_ptr<KernelContext> ctx_;
  cudaStream_t stream_;
};

TEST_F(GRUTest, TestFP32) {
  InitFloatInput();
  GRUCompute<float, PRECISION(kFloat)> kernel;
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
}

TEST_F(GRUTest, TestFP16) {
  InitHalfInput();
  GRUCompute<half, PRECISION(kFP16)> kernel;
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
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
