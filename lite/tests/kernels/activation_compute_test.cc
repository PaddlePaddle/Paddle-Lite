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
#include <cmath>
#include <string>
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/core/arena/framework.h"

namespace paddle {
namespace lite {

enum activation_type_test {
  RELU,
  LEAKY_RELU,
  RELU_CLIPPED,
  PRELU,
  SIGMOID,
  TANH,
  SWISH,
  RELU6,
  LOG,
  EXP,
  FLOOR,
  RSQRT,
  GELU,
  SQUARE,
  HARD_SWISH,
  RECIPROCAL,
  THRESHOLDED_RELU,
  ELU
};

class ActivationComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string input_ = "x";
  std::string output_ = "out";
  std::string prelu_alpha_ = "prelu_alpha";
  float leaky_relu_alpha_ = 0.01;
  float relu_clipped_coef_ = 6.;
  std::string prelu_mode_ = "";
  float swish_beta_ = 0.;
  float hard_swish_threshold = 6.0;
  float hard_swish_scale = 6.0;
  float hard_swish_offset = 3.0;
  float relu_threshold_ = 1.0;
  float elu_alpha_ = 1.0;
  DDim dims_{{1}};
  std::string type_ = "";
  activation_type_test act_type_ = RELU;

 public:
  ActivationComputeTester(const Place& place,
                          const std::string& alias,
                          float leaky_relu_alpha,
                          float relu_clipped_coef,
                          std::string prelu_mode,
                          float swish_beta,
                          float elu_alpha,
                          DDim dims,
                          std::string type,
                          activation_type_test act_type)
      : TestCase(place, alias),
        leaky_relu_alpha_(leaky_relu_alpha),
        relu_clipped_coef_(relu_clipped_coef),
        prelu_mode_(prelu_mode),
        swish_beta_(swish_beta),
        elu_alpha_(elu_alpha),
        dims_(dims),
        type_(type),
        act_type_(act_type) {}

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(output_);
    CHECK(out);
    out->Resize(dims_);
    auto* output_data = out->mutable_data<float>();

    auto* x = scope->FindTensor(input_);
    const auto* x_data = x->data<float>();
    LOG(INFO) << act_type_;
    switch (act_type_) {
      case RELU: {
        for (int i = 0; i < dims_.production(); i++) {
          output_data[i] = std::max(0.f, x_data[i]);
        }
        break;
      }
      case LEAKY_RELU: {
        for (int i = 0; i < dims_.production(); i++) {
          output_data[i] =
              x_data[i] > 0.f ? x_data[i] : x_data[i] * leaky_relu_alpha_;
        }
        break;
      }
      case RELU_CLIPPED: {
        for (int i = 0; i < dims_.production(); i++) {
          output_data[i] = x_data[i] > 0.f ? x_data[i] : 0.f;
          output_data[i] = output_data[i] < relu_clipped_coef_
                               ? output_data[i]
                               : relu_clipped_coef_;
        }
        break;
      }
      case PRELU: {
        auto* alpha = scope->FindTensor(prelu_alpha_);
        const auto* alpha_data = alpha->data<float>();

        int num = dims_[0];
        int channel = dims_[1];
        int csize = dims_[2] * dims_[3];
        int bsize = channel * csize;
        if (prelu_mode_ == "all" || prelu_mode_ == "channel") {
          for (int n = 0; n < num; n++) {
            auto x_data_bptr = x_data + n * bsize;
            auto output_data_bptr = output_data + n * bsize;
            for (int c = 0; c < channel; c++) {
              auto x_data_cptr = x_data_bptr + c * csize;
              auto output_data_cptr = output_data_bptr + c * csize;
              float slope =
                  prelu_mode_ == "all" ? alpha_data[0] : alpha_data[c];
              for (int i = 0; i < csize; i++) {
                output_data_cptr[i] = x_data_cptr[i] > 0.f
                                          ? x_data_cptr[i]
                                          : x_data_cptr[i] * slope;
              }
            }
          }
        } else {
          for (int i = 0; i < dims_.production(); i++) {
            output_data[i] =
                x_data[i] > 0.f ? x_data[i] : x_data[i] * alpha_data[i];
          }
        }
        break;
      }
      case SIGMOID: {
        for (int i = 0; i < dims_.production(); i++) {
          output_data[i] = 1.f / (1.f + std::exp(-x_data[i]));
        }
        break;
      }
      case TANH: {
        for (int i = 0; i < dims_.production(); i++) {
          output_data[i] = (std::exp(x_data[i]) - std::exp(-x_data[i])) /
                           (std::exp(x_data[i]) + std::exp(-x_data[i]));
        }
        break;
      }
      case SWISH: {
        for (int i = 0; i < dims_.production(); i++) {
          output_data[i] =
              x_data[i] / (1.f + std::exp(-swish_beta_ * x_data[i]));
        }
        break;
      }
      case RELU6: {
        for (int i = 0; i < dims_.production(); i++) {
          output_data[i] = x_data[i] > 0.f ? x_data[i] : 0.f;
          output_data[i] = output_data[i] < 6.0 ? output_data[i] : 6.0;
        }
        break;
      }
      case LOG: {
        for (int i = 0; i < dims_.production(); i++) {
          output_data[i] = std::log(x_data[i]);
        }
        break;
      }
      case EXP: {
        for (int i = 0; i < dims_.production(); i++) {
          output_data[i] = std::exp(x_data[i]);
        }
        break;
      }
      case FLOOR: {
        for (int i = 0; i < dims_.production(); i++) {
          output_data[i] = std::floor(x_data[i]);
        }
        break;
      }
      case RSQRT: {
        for (int i = 0; i < dims_.production(); i++) {
          output_data[i] = 1.0 / std::sqrt(x_data[i]);
        }
        break;
      }
      case GELU: {
        for (int i = 0; i < dims_.production(); i++) {
          output_data[i] = x_data[i] * 0.5 *
                           (1.0 + std::erf(x_data[i] * 0.70710678118654752440));
        }
        break;
      }
      case SQUARE: {
        for (int i = 0; i < dims_.production(); i++) {
          output_data[i] = x_data[i] * x_data[i];
        }
        break;
      }
      case HARD_SWISH: {
        for (int i = 0; i < dims_.production(); i++) {
          float max_value = std::max(0.f, x_data[i] + hard_swish_offset);
          float min_value = std::min(max_value, hard_swish_threshold);
          output_data[i] = min_value * x_data[i] / hard_swish_scale;
        }
        break;
      }
      case RECIPROCAL: {
        for (int i = 0; i < dims_.production(); i++) {
          output_data[i] = 1.0 / x_data[i];
        }
        break;
      }
      case THRESHOLDED_RELU: {
        for (int i = 0; i < dims_.production(); i++) {
          output_data[i] = x_data[i] > relu_threshold_ ? x_data[i] : 0.f;
        }
        break;
      }
      case ELU: {
        for (int i = 0; i < dims_.production(); i++) {
          float tmp = std::exp(x_data[i]) - 1;
          float max = x_data[i] > 0.f ? x_data[i] : 0.f;
          float min = x_data[i] < 0.f ? elu_alpha_ * tmp : 0.f;
          output_data[i] = min + max;
        }
        break;
      }
      default:
        LOG(INFO) << "the type of activation " << act_type_ << " is unknow.";
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType(type_);
    op_desc->SetInput("X", {input_});
    op_desc->SetOutput("Out", {output_});
    if (act_type_ == PRELU) {
      op_desc->SetInput("Alpha", {prelu_alpha_});
      op_desc->SetAttr("mode", prelu_mode_);
    }
    if (act_type_ == LEAKY_RELU) {
      op_desc->SetAttr("alpha", leaky_relu_alpha_);
    }
    if (act_type_ == RELU_CLIPPED) {
      op_desc->SetAttr("Relu_clipped_coef", relu_clipped_coef_);
    }
    if (act_type_ == SWISH) {
      op_desc->SetAttr("beta", swish_beta_);
    }
    if (act_type_ == HARD_SWISH) {
      op_desc->SetAttr("threshold", hard_swish_threshold);
      op_desc->SetAttr("scale", hard_swish_scale);
      op_desc->SetAttr("offset", hard_swish_offset);
    }
    if (act_type_ == THRESHOLDED_RELU) {
      op_desc->SetAttr("threshold", relu_threshold_);
    }
    if (act_type_ == ELU) {
      op_desc->SetAttr("alpha", elu_alpha_);
    }
  }

  void PrepareData() override {
    std::vector<float> data(dims_.production());
    for (int i = 0; i < dims_.production(); i++) {
      float sign = i % 3 == 0 ? -1.0f : 1.0f;
      sign = (type_ == "log" || type_ == "rsqrt") ? 1 : sign;
      data[i] = sign * static_cast<float>(i % 128) * 0.013f + 0.001;
    }
    SetCommonTensor(input_, dims_, data.data());

    if (type_ == "prelu") {
      int64_t alpha_len = 0;
      DDim alpha_dims;
      if (prelu_mode_ == "all") {
        alpha_len = 1;
        alpha_dims = DDim(std::vector<int64_t>({alpha_len}));
      } else if (prelu_mode_ == "channel") {
        alpha_len = dims_[1];
        alpha_dims = DDim(std::vector<int64_t>({alpha_len}));
      } else if (prelu_mode_ == "element") {
        alpha_len = dims_.production();
        alpha_dims = dims_;
      }
      std::vector<float> prelu_alpha_data(alpha_len);
      for (int i = 0; i < alpha_len; i++) {
        float sign = i % 3 == 0 ? -1.0f : 1.0f;
        prelu_alpha_data[i] =
            sign * static_cast<float>(i % 128) * 0.013f + 0.001;
      }
      SetCommonTensor(prelu_alpha_, alpha_dims, prelu_alpha_data.data());
    }
  }
};

TEST(Activation_relu, precision) {
  LOG(INFO) << "test relu op";
  Place place;
  float abs_error = 2e-5;
#if defined(LITE_WITH_NPU)
  place = TARGET(kNPU);
  abs_error = 1e-2;  // Using fp16 in NPU
#elif defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#elif defined(LITE_WITH_XPU) && defined(LITE_WITH_XTCL)
  place = TARGET(kXPU);
#elif defined(LITE_WITH_HUAWEI_ASCEND_NPU)
  place = TARGET(kHuaweiAscendNPU);
  abs_error = 1e-2;  // precision_mode default is force_fp16
#else
  return;
#endif

  for (auto dims : std::vector<std::vector<int64_t>>{
           {1, 3, 2, 4}, {2, 3, 4}, {5, 4}, {8}}) {
    std::unique_ptr<arena::TestCase> tester(new ActivationComputeTester(
        place, "def", 0.01, 6., "all", 0., 1.0, DDim(dims), "relu", RELU));
    arena::Arena arena(std::move(tester), place, abs_error);
    arena.TestPrecision();
  }
}

TEST(Activation_leaky_relu, precision) {
  LOG(INFO) << "test leaky_relu op";
  Place place;
  float abs_error = 2e-5;
#if defined(LITE_WITH_NPU)
  place = TARGET(kNPU);
  abs_error = 1e-2;  // Using fp16 in NPU
#elif defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#elif defined(LITE_WITH_HUAWEI_ASCEND_NPU)
  place = TARGET(kHuaweiAscendNPU);
  abs_error = 1e-2;  // precision_mode default is force_fp16
#else
  return;
#endif

  for (auto dims : std::vector<std::vector<int64_t>>{
           {1, 3, 2, 4}, {2, 3, 4}, {5, 4}, {8}}) {
    for (auto slope : {0.01, 0.1}) {
      std::unique_ptr<arena::TestCase> tester(
          new ActivationComputeTester(place,
                                      "def",
                                      slope,
                                      6.,
                                      "all",
                                      0.,
                                      1.0,
                                      DDim(dims),
                                      "leaky_relu",
                                      LEAKY_RELU));
      arena::Arena arena(std::move(tester), place, abs_error);
      arena.TestPrecision();
    }
  }
}

TEST(Activation_relu_clipped, precision) {
  LOG(INFO) << "test relu clipped op";
  Place place;
  float abs_error = 2e-5;
#if defined(LITE_WITH_NPU)
  place = TARGET(kNPU);
  abs_error = 1e-2;  // Using fp16 in NPU
#elif defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#else
  return;
#endif

  for (auto dims : std::vector<std::vector<int64_t>>{
           {1, 3, 2, 4}, {2, 3, 4}, {5, 4}, {8}}) {
    for (auto coef : {0.5, 6.}) {
      std::unique_ptr<arena::TestCase> tester(
          new ActivationComputeTester(place,
                                      "def",
                                      0.01,
                                      coef,
                                      "all",
                                      0.,
                                      1.0,
                                      DDim(dims),
                                      "relu_clipped",
                                      RELU_CLIPPED));
      arena::Arena arena(std::move(tester), place, abs_error);
      arena.TestPrecision();
    }
  }
}

TEST(Activation_prelu, precision) {
  LOG(INFO) << "test prelu op";
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));

  for (auto dims : std::vector<std::vector<int64_t>>{{1, 3, 2, 4}}) {
    for (auto mode : {"all", "channel", "element"}) {
      std::unique_ptr<arena::TestCase> tester(new ActivationComputeTester(
          place, "def", 0.01, 6, mode, 0., 1.0, DDim(dims), "prelu", PRELU));
      arena::Arena arena(std::move(tester), place, 2e-5);
      arena.TestPrecision();
    }
  }
#endif
}

TEST(Activation_sigmoid, precision) {
  LOG(INFO) << "test sigmoid op";
  Place place;
  float abs_error = 2e-5;
#if defined(LITE_WITH_NPU)
  place = TARGET(kNPU);
  abs_error = 1e-2;  // Using fp16 in NPU
#elif defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#elif defined(LITE_WITH_HUAWEI_ASCEND_NPU)
  place = TARGET(kHuaweiAscendNPU);
  abs_error = 1e-2;  // precision_mode default is force_fp16
#else
  return;
#endif

  for (auto dims : std::vector<std::vector<int64_t>>{
           {1, 3, 2, 4}, {2, 3, 4}, {5, 4}, {8}}) {
    std::unique_ptr<arena::TestCase> tester(
        new ActivationComputeTester(place,
                                    "def",
                                    0.01,
                                    6.,
                                    "all",
                                    0.,
                                    1.0,
                                    DDim(dims),
                                    "sigmoid",
                                    SIGMOID));
    arena::Arena arena(std::move(tester), place, abs_error);
    arena.TestPrecision();
  }
}

TEST(Activation_tanh, precision) {
  LOG(INFO) << "test tanh op";
  Place place;
  float abs_error = 2e-5;
#if defined(LITE_WITH_NPU)
  place = TARGET(kNPU);
  abs_error = 1e-2;  // Using fp16 in NPU
#elif defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#elif defined(LITE_WITH_XPU) && defined(LITE_WITH_XTCL)
  place = TARGET(kXPU);
#elif defined(LITE_WITH_HUAWEI_ASCEND_NPU)
  place = TARGET(kHuaweiAscendNPU);
  abs_error = 1e-2;  // precision_mode default is force_fp16
#else
  return;
#endif

  for (auto dims : std::vector<std::vector<int64_t>>{
           {1, 3, 2, 4}, {2, 3, 4}, {5, 4}, {8}}) {
    std::unique_ptr<arena::TestCase> tester(new ActivationComputeTester(
        place, "def", 0.01, 6., "all", 0., 1.0, DDim(dims), "tanh", TANH));
    arena::Arena arena(std::move(tester), place, abs_error);
    arena.TestPrecision();
  }
}

TEST(Activation_swish, precision) {
  LOG(INFO) << "test swish op";
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));

  for (auto dims : std::vector<std::vector<int64_t>>{
           {1, 3, 2, 4}, {2, 3, 4}, {5, 4}, {8}}) {
    for (auto coef : {0.01, 0.1}) {
      std::unique_ptr<arena::TestCase> tester(new ActivationComputeTester(
          place, "def", 0.01, 6, "all", coef, 1.0, DDim(dims), "swish", SWISH));
      arena::Arena arena(std::move(tester), place, 2e-5);
      arena.TestPrecision();
    }
  }
#endif
}

TEST(Activation_relu6, precision) {
  LOG(INFO) << "test relu6 op...";
  Place place;
  float abs_error = 2e-5;
#if defined(LITE_WITH_NPU)
  place = TARGET(kNPU);
  abs_error = 1e-2;  // Using fp16 in NPU
#elif defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#elif defined(LITE_WITH_HUAWEI_ASCEND_NPU)
  place = TARGET(kHuaweiAscendNPU);
  abs_error = 1e-2;  // precision_mode default is force_fp16
#else
  return;
#endif

  for (auto dims : std::vector<std::vector<int64_t>>{
           {1, 3, 2, 4}, {2, 3, 4}, {5, 4}, {8}}) {
    std::unique_ptr<arena::TestCase> tester(new ActivationComputeTester(
        place, "def", 0.01, 6., "all", 0., 1.0, DDim(dims), "relu6", RELU6));
    arena::Arena arena(std::move(tester), place, abs_error);
    arena.TestPrecision();
  }
}

TEST(Activation_log, precision) {
  LOG(INFO) << "test log op";
  Place place;
  float abs_error = 2e-5;
#if defined(LITE_WITH_NPU)
  place = TARGET(kNPU);
  abs_error = 1e-2;  // Using fp16 in NPU
#elif defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#else
  return;
#endif

  for (auto dims : std::vector<std::vector<int64_t>>{
           {1, 3, 2, 4}, {2, 3, 4}, {5, 4}, {8}}) {
    std::unique_ptr<arena::TestCase> tester(new ActivationComputeTester(
        place, "def", 0.01, 6., "all", 0., 1.0, DDim(dims), "log", LOG));
    arena::Arena arena(std::move(tester), place, abs_error);
    arena.TestPrecision();
  }
}

TEST(Activation_exp, precision) {
  LOG(INFO) << "test exp op";
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));

  for (auto dims : std::vector<std::vector<int64_t>>{
           {1, 3, 2, 4}, {2, 3, 4}, {5, 4}, {8}}) {
    std::unique_ptr<arena::TestCase> tester(new ActivationComputeTester(
        place, "def", 0.01, 6., "all", 0., 1.0, DDim(dims), "exp", EXP));
    arena::Arena arena(std::move(tester), place, 2e-5);
    arena.TestPrecision();
  }
#endif
}

TEST(Activation_floor, precision) {
  LOG(INFO) << "test floor op";
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
  for (auto dims : std::vector<std::vector<int64_t>>{
           {1, 3, 2, 4}, {2, 3, 4}, {5, 4}, {8}}) {
    std::unique_ptr<arena::TestCase> tester(new ActivationComputeTester(
        place, "def", 0.01, 6., "all", 0., 1.0, DDim(dims), "floor", FLOOR));
    arena::Arena arena(std::move(tester), place, 2e-5);
    arena.TestPrecision();
  }

#endif
}

TEST(Activation_rsqrt, precision) {
  LOG(INFO) << "test rsqrt op";
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
  for (auto dims : std::vector<std::vector<int64_t>>{
           {1, 3, 2, 4}, {2, 3, 4}, {5, 4}, {8}}) {
    std::unique_ptr<arena::TestCase> tester(new ActivationComputeTester(
        place, "def", 0.01, 6., "all", 0., 1.0, DDim(dims), "rsqrt", RSQRT));
    arena::Arena arena(std::move(tester), place, 2e-5);
    arena.TestPrecision();
  }
#endif
}

TEST(Activation_square, precision) {
  LOG(INFO) << "test square op";
  Place place;
  float abs_error = 2e-5;
#if defined(LITE_WITH_NPU)
  place = TARGET(kNPU);
  abs_error = 1e-2;  // Using fp16 in NPU
#elif defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#else
  return;
#endif

  for (auto dims : std::vector<std::vector<int64_t>>{
           {1, 3, 2, 4}, {2, 3, 4}, {5, 4}, {8}}) {
    std::unique_ptr<arena::TestCase> tester(new ActivationComputeTester(
        place, "def", 0.01, 6., "all", 0., 1.0, DDim(dims), "square", SQUARE));
    arena::Arena arena(std::move(tester), place, abs_error);
    arena.TestPrecision();
  }
}

TEST(Activation_gelu, precision) {
  LOG(INFO) << "test gelu op";
  Place place;
  float abs_error = 2e-5;
#if defined(LITE_WITH_XPU) && defined(LITE_WITH_XTCL)
  place = TARGET(kXPU);
#else
  return;
#endif

  for (auto dims : std::vector<std::vector<int64_t>>{
           {1, 3, 2, 4}, {2, 3, 4}, {5, 4}, {8}}) {
    std::unique_ptr<arena::TestCase> tester(new ActivationComputeTester(
        place, "def", 0.01, 6., "all", 0., 1.0, DDim(dims), "gelu", GELU));
    arena::Arena arena(std::move(tester), place, abs_error);
    arena.TestPrecision();
  }
}

TEST(activation_hard_swish, precision) {
  LOG(INFO) << "test hard_swish op";
  Place place;
  float abs_error = 2e-5;

#if defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#else
  return;
#endif

  for (auto dims : std::vector<std::vector<int64_t>>{
           {1, 3, 2, 4}, {2, 3, 4}, {5, 4}, {8}}) {
    std::unique_ptr<arena::TestCase> tester(
        new ActivationComputeTester(place,
                                    "def",
                                    0.01,
                                    6.,
                                    "all",
                                    0.,
                                    1.0,
                                    DDim(dims),
                                    "hard_swish",
                                    HARD_SWISH));
    arena::Arena arena(std::move(tester), place, abs_error);
    arena.TestPrecision();
  }
}

TEST(activation_reciprocal, precision) {
  LOG(INFO) << "test reciprocal op";
  Place place;
  float abs_error = 2e-5;

#if defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#else
  return;
#endif

  for (auto dims : std::vector<std::vector<int64_t>>{
           {1, 3, 2, 4}, {2, 3, 4}, {5, 4}, {8}}) {
    std::unique_ptr<arena::TestCase> tester(
        new ActivationComputeTester(place,
                                    "def",
                                    0.01,
                                    6.,
                                    "all",
                                    0.,
                                    1.0,
                                    DDim(dims),
                                    "reciprocal",
                                    RECIPROCAL));
    arena::Arena arena(std::move(tester), place, abs_error);
    arena.TestPrecision();
  }
}

TEST(Activation_thresholded_relu, precision) {
  LOG(INFO) << "test thresholded_relu op";
  Place place;
  float abs_error = 2e-5;
#if defined(LITE_WITH_NPU)
  place = TARGET(kNPU);
  abs_error = 1e-2;  // Using fp16 in NPU
#elif defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#else
  return;
#endif

  for (auto dims : std::vector<std::vector<int64_t>>{
           {1, 3, 2, 4}, {2, 3, 4}, {5, 4}, {8}}) {
    std::unique_ptr<arena::TestCase> tester(
        new ActivationComputeTester(place,
                                    "def",
                                    0.01,
                                    6.,
                                    "all",
                                    0.,
                                    1.0,
                                    DDim(dims),
                                    "thresholded_relu",
                                    THRESHOLDED_RELU));
    arena::Arena arena(std::move(tester), place, abs_error);
    arena.TestPrecision();
  }
}

TEST(Activation_elu, precision) {
  LOG(INFO) << "test elu op";
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));

  for (auto dims : std::vector<std::vector<int64_t>>{
           {1, 3, 2, 4}, {2, 3, 4}, {5, 4}, {8}}) {
    std::unique_ptr<arena::TestCase> tester(new ActivationComputeTester(
        place, "def", 0.01, 6., "all", 0., 1.0, DDim(dims), "elu", ELU));
    arena::Arena arena(std::move(tester), place, 2e-5);
    arena.TestPrecision();
  }
#endif
}

}  // namespace lite
}  // namespace paddle
