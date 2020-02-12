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

#include <gtest/gtest.h>
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/core/arena/framework.h"
#include "lite/tests/utils/fill_data.h"

namespace paddle {
namespace lite {

class InstanceNormComputeTest : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string x_ = "x";
  std::string y_ = "y";
  std::string saved_mean_ = "saved_mean";
  std::string saved_variance_ = "saved_variance";
  std::string scale_ = "scale";
  std::string bias_ = "bias";

  DDim dims_{{4, 5, 19, 19}};
  float epsilon_ = 1e-5f;

 public:
  InstanceNormComputeTest(const Place& place,
                          const std::string& alias,
                          DDim dims,
                          float epsilon)
      : TestCase(place, alias), dims_(dims), epsilon_(epsilon) {}

  void RunBaseline(Scope* scope) override {
    auto x = scope->FindTensor(x_);
    auto scale = scope->FindTensor(scale_);
    auto bias = scope->FindTensor(bias_);
    auto y = scope->NewTensor(y_);
    auto saved_mean = scope->NewTensor(saved_mean_);
    auto saved_variance = scope->NewTensor(saved_variance_);
    CHECK(y);
    CHECK(saved_mean);
    CHECK(saved_variance);
    DDim saved_dim({dims_[0] * dims_[1]});
    y->Resize(dims_);
    saved_mean->Resize(saved_dim);
    saved_variance->Resize(saved_dim);

    auto x_data = x->data<float>();
    auto scale_data = scale->data<float>();
    auto bias_data = bias->data<float>();
    auto y_data = y->mutable_data<float>();
    auto saved_mean_data = saved_mean->mutable_data<float>();
    auto saved_variance_data = saved_variance->mutable_data<float>();

    int n = x->dims()[0];
    int c = x->dims()[1];
    int spatial_size = x->dims()[2] * x->dims()[3];

    // compute mean
    for (int i = 0; i < n * c; ++i) {
      const float* x_ptr = x_data + i * spatial_size;
      float sum = 0.f;
      for (int j = 0; j < spatial_size; ++j) {
        sum += x_ptr[j];
      }
      saved_mean_data[i] = sum / spatial_size;
    }
    // compute variance
    for (int i = 0; i < n * c; ++i) {
      const float* x_ptr = x_data + i * spatial_size;
      float sum = 0.f;
      for (int j = 0; j < spatial_size; ++j) {
        sum +=
            (x_ptr[j] - saved_mean_data[i]) * (x_ptr[j] - saved_mean_data[i]);
      }
      saved_variance_data[i] = 1.f / sqrtf(sum / spatial_size + epsilon_);
    }
    // compute out
    for (int i = 0; i < n * c; ++i) {
      const float* x_ptr = x_data + i * spatial_size;
      float* y_ptr = y_data + i * spatial_size;
      float scale_val = scale_data[i % c];
      float bias_val = bias_data[i % c];
      for (int j = 0; j < spatial_size; ++j) {
        y_ptr[j] = scale_val * (x_ptr[j] - saved_mean_data[i]) *
                       saved_variance_data[i] +
                   bias_val;
      }
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("instance_norm");
    op_desc->SetInput("X", {x_});
    op_desc->SetInput("Bias", {bias_});
    op_desc->SetInput("Scale", {scale_});
    op_desc->SetOutput("Y", {y_});
    op_desc->SetOutput("SavedMean", {saved_mean_});
    op_desc->SetOutput("SavedVariance", {saved_variance_});
    op_desc->SetAttr("epsilon", epsilon_);
  }

  void PrepareData() override {
    std::vector<float> x(dims_.production());
    fill_data_rand(x.data(), -1.f, 1.f, dims_.production());

    DDim scale_bias_dims{{dims_[1]}};
    std::vector<float> scale(scale_bias_dims.production());
    fill_data_rand(scale.data(), -1.f, 1.f, scale_bias_dims.production());
    std::vector<float> bias(scale_bias_dims.production());
    fill_data_rand(bias.data(), -1.f, 1.f, scale_bias_dims.production());

    SetCommonTensor(x_, dims_, x.data());
    SetCommonTensor(scale_, scale_bias_dims, scale.data(), {}, true);
    SetCommonTensor(bias_, scale_bias_dims, bias.data(), {}, true);
  }
};

void TestInstanceNorm(Place place,
                      float abs_error = 6e-5,
                      std::vector<std::string> ignored_outs = {}) {
  for (auto& n : {1, 3, 16}) {
    for (auto& c : {1, 4, 16}) {
      for (auto& h : {1, 16, 33, 56}) {
        for (auto& w : {1, 17, 34, 55}) {
          DDim dim_in({n, c, h, w});
          float epsilon = 1e-5f;
          std::unique_ptr<arena::TestCase> tester(
              new InstanceNormComputeTest(place, "def", dim_in, epsilon));
#ifdef LITE_WITH_ARM
          if (place == TARGET(kARM)) {
            auto& ctx = tester->context()->As<ARMContext>();
            ctx.SetRunMode(lite_api::LITE_POWER_HIGH, 4);
          }
#endif
          arena::Arena arena(std::move(tester), place, abs_error);
          if (!arena.TestPrecision(ignored_outs)) {
            LOG(ERROR) << "run n: " << n << ", c: " << c << ", h: " << h
                       << ", w: " << w;
            return;
          }
        }
      }
    }
  }
}

TEST(InstanceNorm, precision) {
  Place place;
  float abs_error = 6e-5;
  std::vector<std::string> ignored_outs = {};
#if defined(LITE_WITH_NPU)
  place = TARGET(kNPU);
  abs_error = 1e-2;  // Using fp16 in NPU
  ignored_outs = {"saved_mean", "saved_variance"};
#elif defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#else
  return;
#endif
  TestInstanceNorm(place, abs_error, ignored_outs);
}

}  // namespace lite
}  // namespace paddle
