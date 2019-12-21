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

class LayerNormComputeTest : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string op_type_ = "layer_norm";
  std::string input_ = "x";
  std::string scale_ = "scale";
  std::string bias_ = "bias";
  std::string output_ = "y";
  std::string mean_ = "mean";
  std::string variance_ = "variance";
  DDim dims_{{4, 5, 19, 19}};
  float epsilon_ = 1e-5f;
  int begin_norm_axis_ = 1;
  bool has_bias_ = true;
  bool has_scale_ = true;

 public:
  LayerNormComputeTest(const Place& place,
                       const std::string& alias,
                       DDim dims,
                       float epsilon,
                       int begin_norm_axis,
                       bool has_bias,
                       bool has_scale)
      : TestCase(place, alias),
        dims_(dims),
        epsilon_(epsilon),
        begin_norm_axis_(begin_norm_axis),
        has_bias_(has_bias),
        has_scale_(has_scale) {}

  void RunBaseline(Scope* scope) override {
    auto x = scope->FindTensor(input_);
    auto scale = scope->FindTensor(scale_);
    auto bias = scope->FindTensor(bias_);

    auto y = scope->NewTensor(output_);
    auto mean = scope->NewTensor(mean_);
    auto variance = scope->NewTensor(variance_);
    CHECK(y);
    CHECK(mean);
    CHECK(variance);
    y->Resize(dims_);

    auto matrix_dim = dims_.Flatten2D(begin_norm_axis_);
    int batch_size = matrix_dim[0];
    int feature_size = matrix_dim[1];
    mean->Resize(std::vector<int64_t>{batch_size});
    variance->Resize(std::vector<int64_t>{batch_size});

    auto* x_data = x->data<float>();
    auto* scale_data = (scale == nullptr ? nullptr : scale->data<float>());
    auto* bias_data = (bias == nullptr ? nullptr : bias->data<float>());
    auto* out_data = y->mutable_data<float>();
    auto* mean_data = mean->mutable_data<float>();
    auto* variance_data = variance->mutable_data<float>();

    for (int i = 0; i < batch_size; ++i) {
      int start = i * feature_size;
      int end = start + feature_size;

      float mean_t = 0;
      float variance_t = 0;
      for (int j = start; j < end; ++j) {
        mean_t += x_data[j];
        variance_t += x_data[j] * x_data[j];
      }
      mean_t /= feature_size;
      variance_t = variance_t / feature_size - mean_t * mean_t;
      mean_data[i] = mean_t;
      variance_data[i] = variance_t;
      variance_t = sqrt(variance_t + epsilon_);
      for (int j = start; j < end; ++j) {
        out_data[j] = (x_data[j] - mean_t) / variance_t;
        if (scale_data) {
          out_data[j] *= scale_data[j - start];
        }
        if (bias_data) {
          out_data[j] += bias_data[j - start];
        }
      }
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType(op_type_);
    op_desc->SetInput("X", {input_});
    op_desc->SetInput("Bias", {bias_});
    op_desc->SetInput("Scale", {scale_});
    op_desc->SetOutput("Y", {output_});
    op_desc->SetOutput("Mean", {mean_});
    op_desc->SetOutput("Variance", {variance_});
    op_desc->SetAttr("epsilon", epsilon_);
    op_desc->SetAttr("begin_norm_axis", begin_norm_axis_);
  }

  void PrepareData() override {
    std::vector<float> din(dims_.production());
    fill_data_rand(din.data(), -1.f, 1.f, dims_.production());

    std::vector<int64_t> scale_v;
    for (size_t i = begin_norm_axis_; i < dims_.size(); i++) {
      scale_v.push_back(dims_[i]);
    }
    DDim scale_dim(scale_v);
    std::vector<float> scale(scale_dim.production());
    fill_data_rand(scale.data(), -1.f, 1.f, scale_dim.production());

    std::vector<float> bias(scale_dim.production());
    fill_data_rand(bias.data(), -1.f, 1.f, scale_dim.production());

    SetCommonTensor(input_, dims_, din.data());
    SetCommonTensor(scale_, scale_dim, scale.data());
    SetCommonTensor(bias_, scale_dim, bias.data());
  }
};

TEST(LayerNorm, precision) {
  LOG(INFO) << "test layer_norm op";
  float abs_error = 2e-5;
  Place place;
#if defined(LITE_WITH_XPU)
  place = TARGET(kXPU);
#elif defined(LITE_WITH_ARM)
  place = TARGET(kARM);
  abs_error = 6e-5;
#else
  return;
#endif

  std::vector<std::vector<int64_t>> dims{{1, 2, 3, 4}, {2, 3, 4}, {3, 4}};
  for (auto dim_in : dims) {
    for (auto epsilon : {1e-5f}) {
      for (auto axis : {0, 1, 2, 3}) {
        for (bool has_bias : {true, false}) {
          for (bool has_scale : {true, false}) {
            if (axis >= dim_in.size()) continue;
            std::unique_ptr<arena::TestCase> tester(
                new LayerNormComputeTest(place,
                                         "def",
                                         DDim(dim_in),
                                         epsilon,
                                         axis,
                                         has_bias,
                                         has_scale));
#ifdef LITE_WITH_ARM
            auto& ctx = tester->context()->As<ARMContext>();
            ctx.SetRunMode(lite_api::LITE_POWER_HIGH, 4);
#endif
            arena::Arena arena(std::move(tester), place, abs_error);
            arena.TestPrecision({"mean", "variance"});
          }
        }
      }
    }
  }
}

}  // namespace lite
}  // namespace paddle
