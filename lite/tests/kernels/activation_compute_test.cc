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

namespace paddle {
namespace lite {

enum activation_type_test {
  RELU,
  LEAKY_RELU,
  RELU_CLIPPED,
  PRELU,
  SIGMOID,
  TANH,
  SWISH
};
class ActivationComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string input_ = "x";
  std::string output_ = "out";
  std::string prelu_channel_slope_ = "prelu_channel_slope";
  float leaky_relu_slope_ = 0.01;
  float relu_clipped_coef_ = 6.;
  bool prelu_channel_shared_ = false;
  float swish_coef_ = 0.;
  DDim dims_{{1}};
  std::string type_ = "relu";
  activation_type_test act_type_ = RELU;

 public:
  ActivationComputeTester(const Place& place,
                          const std::string& alias,
                          float leaky_relu_slope,
                          float relu_clipped_coef,
                          bool prelu_channel_shared,
                          float swish_coef,
                          DDim dims,
                          std::string type,
                          activation_type_test act_type)
      : TestCase(place, alias),
        leaky_relu_slope_(leaky_relu_slope),
        relu_clipped_coef_(relu_clipped_coef),
        prelu_channel_shared_(prelu_channel_shared),
        swish_coef_(swish_coef),
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
              x_data[i] > 0.f ? x_data[i] : x_data[i] * leaky_relu_slope_;
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
        auto* prelu_channel_slope = scope->FindTensor(prelu_channel_slope_);
        const auto* prelu_channel_slope_data =
            prelu_channel_slope->data<float>();

        int num = dims_[0];
        int channel = dims_[1];
        int csize = dims_[2] * dims_[3];
        int bsize = channel * csize;
        for (int n = 0; n < num; n++) {
          auto x_data_bptr = x_data + n * bsize;
          auto output_data_bptr = output_data + n * bsize;
          for (int c = 0; c < channel; c++) {
            auto x_data_cptr = x_data_bptr + c * csize;
            auto output_data_cptr = output_data_bptr + c * csize;
            float slope = prelu_channel_shared_ ? prelu_channel_slope_data[0]
                                                : prelu_channel_slope_data[c];
            for (int i = 0; i < csize; i++) {
              output_data_cptr[i] = x_data_cptr[i] > 0.f
                                        ? x_data_cptr[i]
                                        : x_data_cptr[i] * slope;
            }
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
              x_data[i] / (1.f + std::exp(-swish_coef_ * x_data[i]));
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
    op_desc->SetAttr("Type", type_);
    if (act_type_ == PRELU) {
      op_desc->SetInput("Prelu_channel_slope", {prelu_channel_slope_});
      op_desc->SetAttr("Prelu_channel_shared", prelu_channel_shared_);
    }
    if (act_type_ == LEAKY_RELU) {
      op_desc->SetAttr("Leaky_relu_slope", leaky_relu_slope_);
    }
    if (act_type_ == RELU_CLIPPED) {
      op_desc->SetAttr("Relu_clipped_coef", relu_clipped_coef_);
    }
    if (act_type_ == SWISH) {
      op_desc->SetAttr("Swish_coef", swish_coef_);
    }
  }

  void PrepareData() override {
    std::vector<float> data(dims_.production());
    for (int i = 0; i < dims_.production(); i++) {
      float sign = i % 3 == 0 ? -1.0f : 1.0f;
      data[i] = sign * static_cast<float>(i % 128) * 0.013f + 0.001;
    }
    SetCommonTensor(input_, dims_, data.data());

    if (type_ == "prelu") {
      int64_t slope_len = prelu_channel_shared_ ? 1 : dims_[1];
      std::vector<float> slope_data(slope_len);
      for (int i = 0; i < slope_len; i++) {
        float sign = i % 3 == 0 ? -1.0f : 1.0f;
        slope_data[i] = sign * static_cast<float>(i % 128) * 0.013f + 0.001;
      }
      SetCommonTensor(prelu_channel_slope_,
                      DDim(std::vector<int64_t>({slope_len})),
                      slope_data.data());
    }
  }
};

TEST(Activation_relu, precision) {
#ifdef LITE_WITH_X86
  Place place(TARGET(kX86));
#endif
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
#endif

  for (auto n : {1, 3}) {
    for (auto c : {3, 6}) {
      for (auto h : {9, 18}) {
        for (auto w : {9, 18}) {
          std::unique_ptr<arena::TestCase> tester(new ActivationComputeTester(
              place,
              "def",
              0.01,
              6.,
              false,
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

TEST(Activation_leaky_relu, precision) {
#ifdef LITE_WITH_X86
  Place place(TARGET(kX86));
#endif
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
#endif

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
                false,
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
}

TEST(Activation_relu_clipped, precision) {
#ifdef LITE_WITH_X86
  Place place(TARGET(kX86));
#endif
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
#endif

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
                false,
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
}

TEST(Activation_prelu, precision) {
#ifdef LITE_WITH_X86
  Place place(TARGET(kX86));
#endif
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
#endif

  for (auto n : {1, 3}) {
    for (auto c : {3, 6}) {
      for (auto h : {9, 18}) {
        for (auto w : {9, 18}) {
          for (auto flag : {false, true}) {
            std::unique_ptr<arena::TestCase> tester(new ActivationComputeTester(
                place,
                "def",
                0.01,
                6,
                flag,
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
}

TEST(Activation_sigmoid, precision) {
#ifdef LITE_WITH_X86
  Place place(TARGET(kX86));
#endif
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
#endif

  for (auto n : {1, 3}) {
    for (auto c : {3, 6}) {
      for (auto h : {9, 18}) {
        for (auto w : {9, 18}) {
          std::unique_ptr<arena::TestCase> tester(new ActivationComputeTester(
              place,
              "def",
              0.01,
              6.,
              false,
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
}

TEST(Activation_tanh, precision) {
#ifdef LITE_WITH_X86
  Place place(TARGET(kX86));
#endif
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
#endif

  for (auto n : {1, 3}) {
    for (auto c : {3, 6}) {
      for (auto h : {9, 18}) {
        for (auto w : {9, 18}) {
          std::unique_ptr<arena::TestCase> tester(new ActivationComputeTester(
              place,
              "def",
              0.01,
              6.,
              false,
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
}

TEST(Activation_swish, precision) {
#ifdef LITE_WITH_X86
  Place place(TARGET(kX86));
#endif
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
#endif

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
                false,
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
}

}  // namespace lite
}  // namespace paddle
