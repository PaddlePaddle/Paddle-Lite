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

#include "lite/kernels/cuda/softmax_compute.h"
#include <gtest/gtest.h>
#include <limits>
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

class SoftmaxTest : public ::testing::Test {
 protected:
  SoftmaxTest()
      : n_(2),
        c_(2),
        h_(2),
        w_(2),
        axis_(1),
        use_cudnn_(true),
        shape_({n_, c_, h_, w_}) {
    x_ref_.Resize(lite::DDim(shape_));
    x_gpu_.Resize(lite::DDim(shape_));

    auto x_ref_data = x_ref_.mutable_data<float>();

    for (int64_t i = 0; i < x_ref_.numel(); i++) {
      x_ref_data[i] = static_cast<float>(i % 10 * 0.2);
    }

    out_ref_.Resize(lite::DDim(shape_));
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
    param_.x = &x_gpu_;
    param_.axis = axis_;
    param_.output = &out_gpu_;
    param_.use_cudnn = use_cudnn_;
  }

  void InitFloatInput() {
    x_gpu_.Assign<float, lite::DDim, TARGET(kCUDA)>(x_ref_.data<float>(),
                                                    x_gpu_.dims());
  }

  void InitHalfInput() {
    x_half_.Resize(x_ref_.dims());
    auto x_half_data = x_half_.mutable_data<half>();
    for (int64_t i = 0; i < x_half_.numel(); i++) {
      x_half_data[i] = half(lite::float16(x_ref_.data<float>()[i]));
    }
    x_gpu_.Assign<half, lite::DDim, TARGET(kCUDA)>(x_half_data, x_gpu_.dims());
  }

  void RunBaseLine() {
    const float* x_data = x_ref_.mutable_data<float>();
    float* output_data = out_ref_.mutable_data<float>();
    DDim x_dims = x_ref_.dims();
    ASSERT_EQ(x_dims.data(), out_ref_.dims().data());
    auto x_rank = x_dims.size();
    int axis = axis_;
    if (axis < 0) {
      axis += x_rank;
    }
    int axis_size = x_dims[axis];
    int outer_num = x_dims.Slice(0, axis).production();
    int inner_num = x_dims.Slice(axis + 1, x_rank).production();
    int compute_size = outer_num * inner_num;
    for (int i = 0; i < compute_size; i++) {
      int idx_inner = i % inner_num;
      int idx_outer = (i / inner_num) * axis_size;
      int start = idx_outer * inner_num + idx_inner;
      int offset;

      offset = start;
      float max_data = std::numeric_limits<float>::lowest();
      for (int j = 0; j < axis_size; j++) {
        max_data = x_data[offset] > max_data ? x_data[offset] : max_data;
        offset += inner_num;
      }

      offset = start;
      float sum_data = 0.f;
      for (int j = 0; j < axis_size; j++) {
        output_data[offset] = exp(x_data[offset] - max_data);
        sum_data += output_data[offset];
        offset += inner_num;
      }

      offset = start;
      for (int j = 0; j < axis_size; j++) {
        output_data[offset] /= sum_data;
        offset += inner_num;
      }
    }
  }

  int n_, c_, h_, w_, axis_;
  bool use_cudnn_;
  std::vector<int64_t> shape_;
  lite::Tensor x_ref_, out_ref_;
  lite::Tensor x_gpu_;
  lite::Tensor x_half_;
  lite::Tensor out_cpu_, out_gpu_;

  operators::SoftmaxParam param_;
  std::unique_ptr<KernelContext> ctx_;
  cudaStream_t stream_;
};

TEST_F(SoftmaxTest, TestFP32) {
  InitFloatInput();
  SoftmaxCompute<float, PRECISION(kFloat)> kernel;
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

TEST_F(SoftmaxTest, TestFP16) {
  InitHalfInput();
  SoftmaxCompute<half, PRECISION(kFP16)> kernel;
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
