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
#include "lite/core/test/arena/framework.h"
#include "lite/tests/utils/fill_data.h"

namespace paddle {
namespace lite {

class BatchNormComputeTest : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string op_type_ = "batch_norm";
  std::string input_ = "x";
  std::string scale_ = "scale";
  std::string bias_ = "bias";
  std::string mean_ = "mean";
  std::string variance_ = "variance";
  std::string output_ = "y";
  std::string mean_out_ = "mean_out";
  std::string saved_mean_ = "saved_mean";
  std::string variance_out_ = "variance_out";
  std::string saved_variance_ = "saved_variance";
  DDim dims_{{1, 2, 3, 4}};
  bool use_global_stats_ = false;
  float momentum_ = 0.9;
  float epsilon_ = 1e-5f;
  std::string data_layout_ = "NCHW";
  int is_test_ = 1;

 public:
  BatchNormComputeTest(const Place& place,
                       const std::string& alias,
                       DDim dims,
                       float epsilon)
      : TestCase(place, alias), dims_(dims), epsilon_(epsilon) {}

  void RunBaseline(Scope* scope) override {
    auto x = scope->FindTensor(input_);
    auto scale = scope->FindTensor(scale_);
    auto bias = scope->FindTensor(bias_);
    auto mean = scope->FindTensor(mean_);
    auto variance = scope->FindTensor(variance_);

    auto y = scope->NewTensor(output_);
    auto mean_out = scope->NewTensor(mean_out_);
    auto variance_out = scope->NewTensor(variance_out_);
    auto saved_mean = scope->NewTensor(saved_mean_);
    auto saved_variance = scope->NewTensor(saved_variance_);
    CHECK(y);
    CHECK(mean_out);
    CHECK(variance_out);
    CHECK(saved_mean);
    CHECK(saved_variance);
    y->Resize(dims_);

    int64_t channel_size = 0;
    if (data_layout_ == "NCHW") {
      channel_size = dims_[1];
    } else {
      LOG(FATAL) << "Unknown storage order: " << data_layout_;
    }
    mean_out->Resize({channel_size});
    variance_out->Resize({channel_size});
    saved_mean->Resize({channel_size});
    saved_variance->Resize({channel_size});

    auto x_data = x->data<float>();
    auto y_data = y->mutable_data<float>();
    auto scale_data = scale->data<float>();
    auto bias_data = bias->data<float>();
    auto mean_data = mean->data<float>();
    auto variance_data = variance->data<float>();

    int64_t outer_size = 0;
    int64_t inner_size = 0;
    if (data_layout_ == "NCHW") {
      outer_size = dims_[0];
      inner_size = dims_.Slice(2, dims_.size()).production();
    } else {
      LOG(FATAL) << "Unknown storage order: " << data_layout_;
    }
    auto x_ptr = x_data;
    auto y_ptr = y_data;
    for (int o = 0; o < outer_size; o++) {
      for (int c = 0; c < channel_size; c++) {
        for (int i = 0; i < inner_size; i++) {
          float norm_x =
              (*x_ptr - mean_data[c]) / std::sqrt(variance_data[c] + epsilon_);
          *y_ptr = norm_x * scale_data[c] + bias_data[c];
          x_ptr++;
          y_ptr++;
        }
      }
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType(op_type_);
    op_desc->SetInput("X", {input_});
    op_desc->SetInput("Bias", {bias_});
    op_desc->SetInput("Scale", {scale_});
    op_desc->SetInput("Mean", {mean_});
    op_desc->SetInput("Variance", {variance_});
    op_desc->SetOutput("Y", {output_});
    if (!is_test_) {
      op_desc->SetOutput("MeanOut", {mean_out_});
      op_desc->SetOutput("VarianceOut", {variance_out_});
      op_desc->SetOutput("SavedMean", {saved_mean_});
      op_desc->SetOutput("SavedVariance", {saved_variance_});
    }
    op_desc->SetAttr("epsilon", epsilon_);
    op_desc->SetAttr("momentum", momentum_);
    op_desc->SetAttr("use_global_stats", use_global_stats_);
    op_desc->SetAttr("data_layout", data_layout_);
    op_desc->SetAttr("is_test", is_test_);
  }

  void PrepareData() override {
    std::vector<float> din(dims_.production());
    fill_data_rand(din.data(), -1.f, 1.f, dims_.production());

    DDim scale_dim({dims_[1]});
    std::vector<float> scale(scale_dim.production());
    fill_data_rand(scale.data(), -1.f, 1.f, scale_dim.production());

    std::vector<float> bias(scale_dim.production());
    fill_data_rand(bias.data(), -1.f, 1.f, scale_dim.production());

    std::vector<float> mean(scale_dim.production());
    fill_data_rand(mean.data(), -1.f, 1.f, scale_dim.production());

    std::vector<float> variance(scale_dim.production());
    fill_data_rand(variance.data(), 0.f, 1.f, scale_dim.production());

    SetCommonTensor(input_, dims_, din.data());
    SetCommonTensor(scale_, scale_dim, scale.data(), {}, true);
    SetCommonTensor(bias_, scale_dim, bias.data(), {}, true);
    SetCommonTensor(mean_, scale_dim, mean.data(), {}, true);
    SetCommonTensor(variance_, scale_dim, variance.data(), {}, true);
  }
};

TEST(BatchNorm, precision) {
  float abs_error = 2e-5;
  Place place;
#if defined(LITE_WITH_NNADAPTER)
  place = TARGET(kNNAdapter);
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  abs_error = 1e-1;
#elif defined(NNADAPTER_WITH_CAMBRICON_MLU)
  abs_error = 1e-2;
  // TODO(shentanyue): support later
  return;
#elif defined(NNADAPTER_WITH_VERISILICON_TIMVX)
  abs_error = 1e-2;
#elif defined(NNADAPTER_WITH_HUAWEI_KIRIN_NPU)
  abs_error = 1e-1;
#elif defined(NNADAPTER_WITH_NVIDIA_TENSORRT)
  abs_error = 1e-5;
#elif defined(NNADAPTER_WITH_INTEL_OPENVINO)
  abs_error = 1e-5;
#elif defined(NNADAPTER_WITH_QUALCOMM_QNN)
  abs_error = 1e-2;
#else
  return;
#endif
#elif defined(LITE_WITH_OPENCL)
  place = Place(TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kImageDefault));
  abs_error = 1e-2;  // Using fp16 in OPENCL
#elif defined(LITE_WITH_NPU)
  place = TARGET(kNPU);
#else
  return;
#endif

  for (auto dims :
       std::vector<std::vector<int64_t>>{{1, 2, 3, 4}, {5, 6, 7, 8}}) {
    for (auto epsilon : {1e-5f}) {
      std::unique_ptr<arena::TestCase> tester(
          new BatchNormComputeTest(place, "def", DDim(dims), epsilon));
      arena::Arena arena(std::move(tester), place, abs_error);
      arena.TestPrecision(
          {"mean_out", "saved_mean", "variance_out", "saved_variance"});
    }
  }
}

}  // namespace lite
}  // namespace paddle
