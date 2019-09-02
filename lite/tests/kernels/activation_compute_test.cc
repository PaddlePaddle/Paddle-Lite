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
  FLOOR
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
                          DDim dims,
                          std::string type,
                          activation_type_test act_type)
      : TestCase(place, alias),
        leaky_relu_alpha_(leaky_relu_alpha),
        relu_clipped_coef_(relu_clipped_coef),
        prelu_mode_(prelu_mode),
        swish_beta_(swish_beta),
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
      default:
        LOG(INFO) << "the type of activation is unknow.";
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
  }

  void PrepareData() override {
    std::vector<float> data(dims_.production());
    for (int i = 0; i < dims_.production(); i++) {
      float sign = i % 3 == 0 ? -1.0f : 1.0f;
      sign = type_ == "log" ? 1 : sign;
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
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));

  for (auto n : {1, 3}) {
    for (auto c : {3, 6}) {
      for (auto h : {9, 18}) {
        for (auto w : {9, 18}) {
          for (auto slope : {0.01, 0.1}) {
            std::unique_ptr<arena::TestCase> tester(new ActivationComputeTester(
                place,
                "def",
                0.01,
                6.,
                "all",
                0.,
                DDim(std::vector<int64_t>({n, c, h, w})),
                "relu",
                RELU));
            arena::Arena arena(std::move(tester), place, 2e-5);
            arena.TestPrecision();
          }
        }
      }
    }
  }
#endif
}

TEST(Activation_leaky_relu, precision) {
  LOG(INFO) << "test leaky_relu op";
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));

  for (auto n : {1, 3}) {
    for (auto c : {3, 6}) {
      for (auto h : {9, 18}) {
        for (auto w : {9, 18}) {
          for (auto slope : {0.01, 0.1}) {
            std::unique_ptr<arena::TestCase> tester(new ActivationComputeTester(
                place,
                "def",
                slope,
                6.,
                "all",
                0.,
                DDim(std::vector<int64_t>({n, c, h, w})),
                "leaky_relu",
                LEAKY_RELU));
            arena::Arena arena(std::move(tester), place, 2e-5);
            arena.TestPrecision();
          }
        }
      }
    }
  }
#endif
}

TEST(Activation_relu_clipped, precision) {
  LOG(INFO) << "test relu clipped op";
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));

  for (auto n : {1, 3}) {
    for (auto c : {3, 6}) {
      for (auto h : {9, 18}) {
        for (auto w : {9, 18}) {
          for (auto coef : {0.5, 6.}) {
            std::unique_ptr<arena::TestCase> tester(new ActivationComputeTester(
                place,
                "def",
                0.01,
                coef,
                "all",
                0.,
                DDim(std::vector<int64_t>({n, c, h, w})),
                "relu_clipped",
                RELU_CLIPPED));
            arena::Arena arena(std::move(tester), place, 2e-5);
            arena.TestPrecision();
          }
        }
      }
    }
  }
#endif
}

TEST(Activation_prelu, precision) {
  LOG(INFO) << "test prelu op";
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));

  for (auto n : {1, 3}) {
    for (auto c : {3, 6}) {
      for (auto h : {9, 18}) {
        for (auto w : {9, 18}) {
          for (auto mode : {"all", "channel", "element"}) {
            std::unique_ptr<arena::TestCase> tester(new ActivationComputeTester(
                place,
                "def",
                0.01,
                6,
                mode,
                0.,
                DDim(std::vector<int64_t>({n, c, h, w})),
                "prelu",
                PRELU));
            arena::Arena arena(std::move(tester), place, 2e-5);
            arena.TestPrecision();
          }
        }
      }
    }
  }
#endif
}

TEST(Activation_sigmoid, precision) {
  LOG(INFO) << "test sigmoid op";
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));

  for (auto n : {1, 3}) {
    for (auto c : {3, 6}) {
      for (auto h : {9, 18}) {
        for (auto w : {9, 18}) {
          std::unique_ptr<arena::TestCase> tester(new ActivationComputeTester(
              place,
              "def",
              0.01,
              6.,
              "all",
              0.,
              DDim(std::vector<int64_t>({n, c, h, w})),
              "sigmoid",
              SIGMOID));
          arena::Arena arena(std::move(tester), place, 2e-5);
          arena.TestPrecision();
        }
      }
    }
  }
#endif
}

TEST(Activation_tanh, precision) {
  LOG(INFO) << "test tanh op";
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));

  for (auto n : {1, 3}) {
    for (auto c : {3, 6}) {
      for (auto h : {9, 18}) {
        for (auto w : {9, 18}) {
          std::unique_ptr<arena::TestCase> tester(new ActivationComputeTester(
              place,
              "def",
              0.01,
              6.,
              "all",
              0.,
              DDim(std::vector<int64_t>({n, c, h, w})),
              "tanh",
              TANH));
          arena::Arena arena(std::move(tester), place, 2e-5);
          arena.TestPrecision();
        }
      }
    }
  }
#endif
}

TEST(Activation_swish, precision) {
  LOG(INFO) << "test swish op";
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));

  for (auto n : {1, 3}) {
    for (auto c : {3, 6}) {
      for (auto h : {9, 18}) {
        for (auto w : {9, 18}) {
          for (auto coef : {0.01, 0.1}) {
            std::unique_ptr<arena::TestCase> tester(new ActivationComputeTester(
                place,
                "def",
                0.01,
                6,
                "all",
                coef,
                DDim(std::vector<int64_t>({n, c, h, w})),
                "swish",
                SWISH));
            arena::Arena arena(std::move(tester), place, 2e-5);
            arena.TestPrecision();
          }
        }
      }
    }
  }
#endif
}

TEST(Activation_relu6, precision) {
  LOG(INFO) << "test relu6 op...";
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));

  for (auto n : {1, 3}) {
    for (auto c : {3, 6}) {
      for (auto h : {9, 18}) {
        for (auto w : {9, 18}) {
          for (auto slope : {0.01, 0.1}) {
            std::unique_ptr<arena::TestCase> tester(new ActivationComputeTester(
                place,
                "def",
                0.01,
                6.,
                "all",
                0.,
                DDim(std::vector<int64_t>({n, c, h, w})),
                "relu6",
                RELU6));
            arena::Arena arena(std::move(tester), place, 2e-5);
            arena.TestPrecision();
          }
        }
      }
    }
  }
#endif
}

TEST(Activation_log, precision) {
  LOG(INFO) << "test log op";
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));

  for (auto n : {1, 3}) {
    for (auto c : {3, 6}) {
      for (auto h : {9, 18}) {
        for (auto w : {9, 18}) {
          std::unique_ptr<arena::TestCase> tester(new ActivationComputeTester(
              place,
              "def",
              0.01,
              6.,
              "all",
              0.,
              DDim(std::vector<int64_t>({n, c, h, w})),
              "log",
              LOG));
          arena::Arena arena(std::move(tester), place, 2e-5);
          arena.TestPrecision();
        }
      }
    }
  }
#endif
}

TEST(Activation_exp, precision) {
  LOG(INFO) << "test exp op";
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));

  for (auto n : {1, 3}) {
    for (auto c : {3, 6}) {
      for (auto h : {9, 18}) {
        for (auto w : {9, 18}) {
          std::unique_ptr<arena::TestCase> tester(new ActivationComputeTester(
              place,
              "def",
              0.01,
              6.,
              "all",
              0.,
              DDim(std::vector<int64_t>({n, c, h, w})),
              "exp",
              EXP));
          arena::Arena arena(std::move(tester), place, 2e-5);
          arena.TestPrecision();
        }
      }
    }
  }
#endif
}

TEST(Activation_floor, precision) {
  LOG(INFO) << "test floor op";
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
  for (auto n : {1, 3}) {
    for (auto c : {3, 6}) {
      for (auto h : {9, 18}) {
        for (auto w : {9, 18}) {
          std::unique_ptr<arena::TestCase> tester(new ActivationComputeTester(
              place,
              "def",
              0.01,
              6.,
              "all",
              0.,
              DDim(std::vector<int64_t>({n, c, h, w})),
              "floor",
              FLOOR));
          arena::Arena arena(std::move(tester), place, 2e-5);
          arena.TestPrecision();
        }
      }
    }
  }
#endif
}

}  // namespace lite
}  // namespace paddle
