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
#include "lite/core/test/arena/framework.h"
#include "lite/tests/utils/fill_data.h"

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
  SQRT,
  RSQRT,
  GELU,
  SQUARE,
  HARD_SWISH,
  RECIPROCAL,
  THRESHOLDED_RELU,
  ELU,
  SOFTSIGN,
  HARD_SIGMOID,
  ABS,
  MISH,
  SOFTPLUS
};

template <class T = float>
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
  float threshold_ = 6.0;

  float hard_sigmoid_slope_ = 0.2f;
  float hard_sigmoid_offset_ = 0.5f;
  float softplus_beta_ = 1.f;
  float softplus_threshold_ = 20.f;
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
    auto* output_data = out->template mutable_data<T>();

    auto* x = scope->FindTensor(input_);
    const auto* x_data = x->template data<T>();
    switch (act_type_) {
      case RELU: {
        for (int i = 0; i < dims_.production(); i++) {
          output_data[i] = x_data[i] > 0.f ? x_data[i] : 0.f;
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
        const auto* alpha_data = alpha->template data<T>();

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
              T slope = prelu_mode_ == "all" ? alpha_data[0] : alpha_data[c];
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
          output_data[i] =
              output_data[i] < threshold_ ? output_data[i] : threshold_;
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
      case SQRT: {
        for (int i = 0; i < dims_.production(); i++) {
          output_data[i] = std::sqrt(x_data[i]);
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
      case SOFTSIGN: {
        for (int i = 0; i < dims_.production(); i++) {
          output_data[i] = x_data[i] / (1 + std::fabs(x_data[i]));
        }
        break;
      }
      case HARD_SIGMOID: {
        for (int i = 0; i < dims_.production(); ++i) {
          T tmp = x_data[i] * hard_sigmoid_slope_ + hard_sigmoid_offset_;
          if (tmp < 1.f && tmp > 0.f) {
            output_data[i] = tmp;
          } else if (tmp <= 0.f) {
            output_data[i] = 0.f;
          } else if (tmp >= 1.f) {
            output_data[i] = 1.f;
          }
        }
        break;
      }
      case ABS: {
        for (int i = 0; i < dims_.production(); i++) {
          output_data[i] = std::fabs(x_data[i]);
        }
        break;
      }
      case MISH: {
        for (int i = 0; i < dims_.production(); i++) {
          float x = x_data[i];
          float sp = 0.0f;
          if (threshold_ > 0 && x > threshold_)
            sp = x;
          else if (threshold_ > 0 && x < -threshold_)
            sp = std::exp(x);
          else
            sp = std::log1p(std::exp(x));
          output_data[i] = x * std::tanh(sp);
        }
        break;
      }
      case SOFTPLUS: {
        for (int i = 0; i < dims_.production(); i++) {
          output_data[i] =
              x_data[i] * softplus_beta_ > softplus_threshold_
                  ? x_data[i]
                  : std::log(1 + std::exp(x_data[i] * softplus_beta_)) /
                        softplus_beta_;
        }
        break;
      }
      default:
        LOG(FATAL) << "the type of activation " << act_type_ << " is unknow.";
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
    if (act_type_ == RELU6) {
      op_desc->SetAttr("threshold", threshold_);
    }
    if (act_type_ == GELU) {
      op_desc->SetAttr("approximate", false);
    }
    if (act_type_ == HARD_SIGMOID) {
      op_desc->SetAttr("slope", hard_sigmoid_slope_);
      op_desc->SetAttr("offset", hard_sigmoid_offset_);
    }
    if (act_type_ == MISH) {
      op_desc->SetAttr("threshold", threshold_);
    }
    if (act_type_ == SOFTPLUS) {
      op_desc->SetAttr("beta", softplus_beta_);
      op_desc->SetAttr("threshold", softplus_threshold_);
    }
  }

  void PrepareData() override {
    std::vector<T> data(dims_.production());
    for (int i = 0; i < dims_.production(); i++) {
      T sign = i % 3 == 0 ? -1.0f : 1.0f;
      sign = (type_ == "log" || type_ == "rsqrt" || type_ == "sqrt") ? 1 : sign;
      data[i] = sign * static_cast<T>(i % 128) * 0.013f + 0.001;
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
      SetCommonTensor(
          prelu_alpha_, alpha_dims, prelu_alpha_data.data(), {}, true);
    }
  }
};

template <class T = float>
void TestAct(const Place& place,
             const std::string& alias,
             float leaky_relu_alpha,
             float relu_clipped_coef,
             std::string prelu_mode,
             float swish_beta,
             float elu_alpha,
             DDim dims,
             std::string type,
             activation_type_test act_type,
             float abs_error = 2e-5) {
  std::unique_ptr<arena::TestCase> tester(
      new ActivationComputeTester<T>(place,
                                     "def",
                                     leaky_relu_alpha,
                                     relu_clipped_coef,
                                     prelu_mode,
                                     swish_beta,
                                     elu_alpha,
                                     dims,
                                     type,
                                     act_type));
  arena::Arena arena(std::move(tester), place, abs_error);
  arena.TestPrecision();
}

template <class T = float>
void TestActPerformance(const Place& place,
                        const std::string& alias,
                        float leaky_relu_alpha,
                        float relu_clipped_coef,
                        std::string prelu_mode,
                        float swish_beta,
                        float elu_alpha,
                        DDim dims,
                        std::string type,
                        activation_type_test act_type,
                        float abs_error = 2e-5) {
  std::unique_ptr<arena::TestCase> tester(
      new ActivationComputeTester<T>(place,
                                     "def",
                                     leaky_relu_alpha,
                                     relu_clipped_coef,
                                     prelu_mode,
                                     swish_beta,
                                     elu_alpha,
                                     dims,
                                     type,
                                     act_type));
  arena::Arena arena(std::move(tester), place, abs_error);
  arena.TestPerformance();
}

TEST(Activation_relu, precision) {
  Place place;
  float abs_error = 2e-5;
#if defined(LITE_WITH_NNADAPTER)
  place = TARGET(kNNAdapter);
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  abs_error = 5e-2;
#elif defined(NNADAPTER_WITH_HUAWEI_KIRIN_NPU)
  abs_error = 5e-2;
#elif defined(NNADAPTER_WITH_CAMBRICON_MLU)
  abs_error = 1e-3;
#elif defined(NNADAPTER_WITH_VERISILICON_TIMVX)
  abs_error = 2e-5;
#elif defined(NNADAPTER_WITH_KUNLUNXIN_XTCL)
  abs_error = 2e-5;
#elif defined(NNADAPTER_WITH_ANDROID_NNAPI)
  abs_error = 5e-2;
#elif defined(NNADAPTER_WITH_GOOGLE_XNNPACK)
  abs_error = 5e-2;
#elif defined(NNADAPTER_WITH_INTEL_OPENVINO)
  abs_error = 1e-5;
#else
  return;
#endif
#elif defined(LITE_WITH_NPU)
  place = TARGET(kNPU);
  abs_error = 1e-2;  // Using fp16 in NPU
#elif defined(LITE_WITH_OPENCL)
  place = Place(TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kNCHW));
  abs_error = 1e-2;  // Using fp16 in OPENCL
#elif defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#elif defined(LITE_WITH_X86)
  place = TARGET(kX86);
#else
  return;
#endif

  for (auto dims : std::vector<std::vector<int64_t>>{
           {1, 3, 2, 4}, {2, 3, 4}, {5, 4}, {8}}) {
    TestAct(place,
            "def",
            0.01,
            6.,
            "all",
            0.,
            1.0,
            DDim(dims),
            "relu",
            RELU,
            abs_error);
  }
}

TEST(Activation_leaky_relu, precision) {
  Place place;
  float abs_error = 2e-5;
  std::vector<std::vector<int64_t>> test_dims{
      {1, 3, 2, 4}, {2, 3, 4}, {5, 4}, {8}};
#if defined(LITE_WITH_NNADAPTER)
  place = TARGET(kNNAdapter);
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  abs_error = 1e-2;
#elif defined(NNADAPTER_WITH_HUAWEI_KIRIN_NPU)
  abs_error = 1e-2;
#elif defined(NNADAPTER_WITH_QUALCOMM_QNN)
  abs_error = 1e-2;
#elif defined(NNADAPTER_WITH_VERISILICON_TIMVX)
  abs_error = 1e-5;
#elif defined(NNADAPTER_WITH_CAMBRICON_MLU)
  abs_error = 1e-3;
#elif defined(NNADAPTER_WITH_NVIDIA_TENSORRT)
  abs_error = 2e-5;
#elif defined(NNADAPTER_WITH_INTEL_OPENVINO)
  abs_error = 1e-5;
  for (auto& dims : test_dims) {
    if (dims.size() == 1) {
      dims.push_back(1);
    }
  }
#else
  return;
#endif
#elif defined(LITE_WITH_NPU)
  place = TARGET(kNPU);
  abs_error = 1e-2;  // Using fp16 in NPU
#elif defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#elif defined(LITE_WITH_X86)
  place = TARGET(kX86);
#else
  return;
#endif

  for (auto dims : test_dims) {
    for (auto slope : {0.01, 0.1}) {
      TestAct(place,
              "def",
              slope,
              6.,
              "all",
              0.,
              1.0,
              DDim(dims),
              "leaky_relu",
              LEAKY_RELU,
              abs_error);
    }
  }
}

TEST(Activation_relu_clipped, precision) {
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
      TestAct(place,
              "def",
              0.01,
              coef,
              "all",
              0.,
              1.0,
              DDim(dims),
              "relu_clipped",
              RELU_CLIPPED,
              abs_error);
    }
  }
}

TEST(Activation_prelu, precision) {
  LOG(INFO) << "test prelu op";
  Place place;
  float abs_error = 2e-5;
  std::vector<std::string> modes{"all", "channel", "element"};
#if defined(LITE_WITH_NNADAPTER)
  place = TARGET(kNNAdapter);
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  abs_error = 1e-2;
  modes = {"all", "channel"};
#elif defined(NNADAPTER_WITH_HUAWEI_KIRIN_NPU)
  abs_error = 1e-2;
  modes = {"all", "channel"};
#else
  return;
#endif
#elif defined(LITE_WITH_OPENCL)
  place = Place(TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kImageDefault));
  abs_error = 1e-2;  // Using fp16 in OPENCL
#elif defined(LITE_WITH_XPU)
  place = TARGET(kXPU);
#elif defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#else
  return;
#endif
  for (auto dims : std::vector<std::vector<int64_t>>{{1, 3, 2, 4}}) {
    for (auto mode : modes) {
      TestAct(place,
              "def",
              0.01,
              6,
              mode,
              0.,
              1.0,
              DDim(dims),
              "prelu",
              PRELU,
              abs_error);
    }
  }
}

TEST(Activation_sigmoid, precision) {
  Place place;
  float abs_error = 2e-5;
  std::vector<std::vector<int64_t>> test_dims{
      {1, 3, 2, 4}, {2, 3, 4}, {5, 4}, {8}};
#if defined(LITE_WITH_NNADAPTER)
  place = TARGET(kNNAdapter);
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  abs_error = 5e-2;
#elif defined(NNADAPTER_WITH_HUAWEI_KIRIN_NPU)
  abs_error = 5e-2;
#elif defined(NNADAPTER_WITH_CAMBRICON_MLU)
  abs_error = 1e-3;
#elif defined(NNADAPTER_WITH_NVIDIA_TENSORRT)
  abs_error = 2e-5;
#elif defined(NNADAPTER_WITH_INTEL_OPENVINO)
  abs_error = 1e-5;
  for (auto& dims : test_dims) {
    if (dims.size() == 1) {
      dims.push_back(1);
    }
  }
#else
  return;
#endif
#elif defined(LITE_WITH_NPU)
  place = TARGET(kNPU);
  abs_error = 1e-2;  // Using fp16 in NPU
#elif defined(LITE_WITH_OPENCL)
  place = Place(TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kNCHW));
  abs_error = 1e-2;  // Using fp16 in OPENCL
#elif defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#elif defined(LITE_WITH_X86)
  place = TARGET(kX86);
#else
  return;
#endif

  for (auto dims : test_dims) {
    TestAct(place,
            "def",
            0.01,
            6.,
            "all",
            0.,
            1.0,
            DDim(dims),
            "sigmoid",
            SIGMOID,
            abs_error);
  }
}

TEST(Activation_tanh, precision) {
  Place place;
  float abs_error = 2e-5;
  std::vector<std::vector<int64_t>> test_dims{
      {1, 3, 2, 4}, {2, 3, 4}, {5, 4}, {8}};
#if defined(LITE_WITH_NNADAPTER)
  place = TARGET(kNNAdapter);
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  abs_error = 5e-2;
#elif defined(NNADAPTER_WITH_HUAWEI_KIRIN_NPU)
  abs_error = 5e-2;
#elif defined(NNADAPTER_WITH_CAMBRICON_MLU)
  abs_error = 1e-3;
#elif defined(NNADAPTER_WITH_VERISILICON_TIMVX)
  abs_error = 2e-5;
#elif defined(NNADAPTER_WITH_NVIDIA_TENSORRT)
  abs_error = 2e-5;
#elif defined(NNADAPTER_WITH_QUALCOMM_QNN)
  abs_error = 1e-2;
#elif defined(NNADAPTER_WITH_INTEL_OPENVINO)
  abs_error = 1e-5;
  for (auto& dims : test_dims) {
    if (dims.size() == 1) {
      dims.push_back(1);
    }
  }
#else
  return;
#endif
#elif defined(LITE_WITH_NPU)
  place = TARGET(kNPU);
  abs_error = 1e-2;  // Using fp16 in NPU
#elif defined(LITE_WITH_OPENCL)
  place = Place(TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kNCHW));
  abs_error = 1e-2;  // Using fp16 in OPENCL
#elif defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#elif defined(LITE_WITH_X86)
  place = TARGET(kX86);
#else
  return;
#endif

  for (auto dims : test_dims) {
    TestAct(place,
            "def",
            0.01,
            6.,
            "all",
            0.,
            1.0,
            DDim(dims),
            "tanh",
            TANH,
            abs_error);
  }
}

TEST(Activation_swish, precision) {
  Place place;
  float abs_error = 2e-5;
  std::vector<float> coefs{0.01, 0.1};
  std::vector<std::vector<int64_t>> test_dims{
      {1, 3, 2, 4}, {2, 3, 4}, {5, 4}, {8}};
#if defined(LITE_WITH_NNADAPTER)
  place = TARGET(kNNAdapter);
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  abs_error = 1e-2;
  coefs = {1.};
#elif defined(NNADAPTER_WITH_HUAWEI_KIRIN_NPU)
  abs_error = 1e-2;
  coefs = {1.};
#elif defined(NNADAPTER_WITH_QUALCOMM_QNN)
  abs_error = 1e-2;
  coefs = {1.};
#elif defined(NNADAPTER_WITH_INTEL_OPENVINO)
  abs_error = 1e-5;
  coefs = {1.};
#elif defined(NNADAPTER_WITH_NVIDIA_TENSORRT)
  abs_error = 2e-5;
  coefs = {1.};
  for (auto& dims : test_dims) {
    if (dims.size() == 1) {
      dims.push_back(1);
    }
  }
#else
  return;
#endif
#elif defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#else
  return;
#endif

  for (auto dims : test_dims) {
    for (auto coef : coefs) {
      TestAct(place,
              "def",
              0.01,
              6,
              "all",
              coef,
              1.0,
              DDim(dims),
              "swish",
              SWISH,
              abs_error);
    }
  }
}

TEST(Activation_relu6, precision) {
  Place place;
  float abs_error = 2e-5;
  std::vector<std::vector<int64_t>> test_dims{
      {1, 3, 2, 4}, {2, 3, 4}, {5, 4}, {8}};
#if defined(LITE_WITH_NNADAPTER)
  place = TARGET(kNNAdapter);
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  abs_error = 1e-2;
#elif defined(NNADAPTER_WITH_HUAWEI_KIRIN_NPU)
  abs_error = 1e-2;
#elif defined(NNADAPTER_WITH_CAMBRICON_MLU)
  abs_error = 1e-3;
#elif defined(NNADAPTER_WITH_VERISILICON_TIMVX)
  abs_error = 2e-5;
#elif defined(NNADAPTER_WITH_KUNLUNXIN_XTCL)
  abs_error = 2e-5;
#elif defined(NNADAPTER_WITH_NVIDIA_TENSORRT)
  abs_error = 2e-5;
#elif defined(NNADAPTER_WITH_INTEL_OPENVINO)
  abs_error = 1e-5;
  for (auto& dims : test_dims) {
    if (dims.size() == 1) {
      dims.push_back(1);
    }
  }
#elif defined(NNADAPTER_WITH_QUALCOMM_QNN)
  abs_error = 1e-2;
#else
  return;
#endif
#elif defined(LITE_WITH_NPU)
  place = TARGET(kNPU);
  abs_error = 1e-2;  // Using fp16 in NPU
#elif defined(LITE_WITH_OPENCL)
  place = Place(TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kNCHW));
  abs_error = 1e-2;  // Using fp16 in OPENCL
#elif defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#elif defined(LITE_WITH_X86)
  place = TARGET(kX86);
#else
  return;
#endif

  for (auto dims : test_dims) {
    TestAct(place,
            "def",
            0.01,
            6.,
            "all",
            0.,
            1.0,
            DDim(dims),
            "relu6",
            RELU6,
            abs_error);
  }
}

TEST(Activation_log, precision) {
  Place place;
  float abs_error = 2e-5;
  std::vector<std::vector<int64_t>> test_dims{
      {1, 3, 2, 4}, {2, 3, 4}, {5, 4}, {8}};
#if defined(LITE_WITH_NNADAPTER)
  place = TARGET(kNNAdapter);
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  abs_error = 1e-2;
#elif defined(NNADAPTER_WITH_HUAWEI_KIRIN_NPU)
  abs_error = 5e-2;
#elif defined(NNADAPTER_WITH_CAMBRICON_MLU)
  abs_error = 1e-3;
#elif defined(NNADAPTER_WITH_NVIDIA_TENSORRT)
  abs_error = 2e-5;
#elif defined(NNADAPTER_WITH_INTEL_OPENVINO)
  abs_error = 1e-5;
  for (auto& dims : test_dims) {
    if (dims.size() == 1) {
      dims.push_back(1);
    }
  }
#else
  return;
#endif
#elif defined(LITE_WITH_NPU)
  place = TARGET(kNPU);
  abs_error = 1e-2;  // Using fp16 in NPU
#elif defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#else
  return;
#endif

  for (auto dims : test_dims) {
    TestAct(place,
            "def",
            0.01,
            6.,
            "all",
            0.,
            1.0,
            DDim(dims),
            "log",
            LOG,
            abs_error);
  }
}

TEST(Activation_exp, precision) {
  Place place;
  float abs_error = 2e-5;
  std::vector<std::vector<int64_t>> test_dims{
      {1, 3, 2, 4}, {2, 3, 4}, {5, 4}, {8}};
#if defined(LITE_WITH_NNADAPTER)
  place = TARGET(kNNAdapter);
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  abs_error = 5e-2;
#elif defined(NNADAPTER_WITH_HUAWEI_KIRIN_NPU)
  abs_error = 5e-2;
#elif defined(NNADAPTER_WITH_NVIDIA_TENSORRT)
  abs_error = 2e-5;
#elif defined(NNADAPTER_WITH_INTEL_OPENVINO)
  abs_error = 1e-5;
  for (auto& dims : test_dims) {
    if (dims.size() == 1) {
      dims.push_back(1);
    }
  }
#elif defined(NNADAPTER_WITH_CAMBRICON_MLU)
  abs_error = 1e-3;
#else
  return;
#endif
#elif defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#else
  return;
#endif
  for (auto dims : test_dims) {
    TestAct(place,
            "def",
            0.01,
            6.,
            "all",
            0.,
            1.0,
            DDim(dims),
            "exp",
            EXP,
            abs_error);
  }
}

TEST(Activation_floor, precision) {
  Place place;
  float abs_error = 2e-5;
#if defined(LITE_WITH_NNADAPTER)
  place = TARGET(kNNAdapter);
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  abs_error = 1e-2;
#elif defined(NNADAPTER_WITH_HUAWEI_KIRIN_NPU)
  abs_error = 5e-2;
#else
  return;
#endif
#elif defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#else
  return;
#endif

  for (auto dims : std::vector<std::vector<int64_t>>{
           {1, 3, 2, 4}, {2, 3, 4}, {5, 4}, {8}}) {
    TestAct(place,
            "def",
            0.01,
            6.,
            "all",
            0.,
            1.0,
            DDim(dims),
            "floor",
            FLOOR,
            abs_error);
  }
}

TEST(Activation_rsqrt, precision) {
  Place place;
  float abs_error = 2e-5;
#if defined(LITE_WITH_OPENCL)
  place = Place(TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kImageDefault));
  abs_error = 1e-2;  // Using fp16 in OPENCL
#elif defined(LITE_WITH_XPU)
  place = TARGET(kXPU);
#elif defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#elif defined(LITE_WITH_X86)
  place = TARGET(kX86);
#else
  return;
#endif

  for (auto dims : std::vector<std::vector<int64_t>>{
           {1, 3, 2, 4}, {2, 3, 4}, {5, 4}, {8}}) {
    TestAct(place,
            "def",
            0.01,
            6.,
            "all",
            0.,
            1.0,
            DDim(dims),
            "rsqrt",
            RSQRT,
            abs_error);
  }
}

TEST(Activation_sqrt, precision) {
  Place place;
  float abs_error = 2e-5;
#if defined(LITE_WITH_NNADAPTER)
  place = TARGET(kNNAdapter);
#if defined(NNADAPTER_WITH_INTEL_OPENVINO)
#else
  return;
#endif
#endif
#if defined(LITE_WITH_OPENCL)
  place = Place(TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kImageDefault));
  abs_error = 1e-2;  // Using fp16 in OPENCL
#elif defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#elif defined(LITE_WITH_X86)
  place = TARGET(kX86);
#else
  return;
#endif

  for (auto dims : std::vector<std::vector<int64_t>>{
           {1, 3, 2, 4}, {2, 3, 4}, {5, 4}, {8}}) {
    TestAct(place,
            "def",
            0.01,
            6.,
            "all",
            0.,
            1.0,
            DDim(dims),
            "sqrt",
            SQRT,
            abs_error);
  }
}

TEST(Activation_square, precision) {
  Place place;
  float abs_error = 2e-5;
#if defined(LITE_WITH_NNADAPTER)
  place = TARGET(kNNAdapter);
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  abs_error = 1e-2;
#elif defined(NNADAPTER_WITH_HUAWEI_KIRIN_NPU)
  abs_error = 5e-2;
#else
  return;
#endif
#elif defined(LITE_WITH_OPENCL)
  place = Place(TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kImageDefault));
  abs_error = 1e-2;
#elif defined(LITE_WITH_NPU)
  place = TARGET(kNPU);
  abs_error = 1e-2;  // Using fp16 in NPU
#elif defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#elif defined(LITE_WITH_X86)
  place = TARGET(kX86);
#else
  return;
#endif

  for (auto dims : std::vector<std::vector<int64_t>>{
           {1, 3, 2, 4}, {2, 3, 4}, {5, 4}, {8}}) {
    TestAct(place,
            "def",
            0.01,
            6.,
            "all",
            0.,
            1.0,
            DDim(dims),
            "square",
            SQUARE,
            abs_error);
  }
}

TEST(Activation_gelu, precision) {
  Place place;
  float abs_error = 2e-5;
#if defined(LITE_WITH_NNADAPTER)
  place = TARGET(kNNAdapter);
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  abs_error = 1e-2;
#elif defined(NNADAPTER_WITH_HUAWEI_KIRIN_NPU)
  abs_error = 5e-2;
#elif defined(NNADAPTER_WITH_CAMBRICON_MLU)
  abs_error = 1e-3;
#elif defined(NNADAPTER_WITH_INTEL_OPENVINO)
  abs_error = 1e-5;
#elif defined(NNADAPTER_WITH_QUALCOMM_QNN)
  abs_error = 1e-2;
#else
  return;
#endif
#elif defined(LITE_WITH_X86)
  place = TARGET(kX86);
#elif defined(LITE_WITH_OPENCL)
  place = Place(TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kNCHW));
  abs_error = 1e-2;  // Using fp16 in OPENCL
#elif defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#else
  return;
#endif

  for (auto dims : std::vector<std::vector<int64_t>>{
           {1, 3, 2, 4}, {2, 3, 4}, {5, 4}, {8}}) {
    TestAct(place,
            "def",
            0.01,
            6.,
            "all",
            0.,
            1.0,
            DDim(dims),
            "gelu",
            GELU,
            abs_error);
  }
}

TEST(Activation_mish, precision) {
  Place place;
  float abs_error = 2e-5;
#if defined(LITE_WITH_X86)
  place = TARGET(kX86);
#elif defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#else
  return;
#endif
  for (auto dims : std::vector<std::vector<int64_t>>{
           {1, 3, 2, 4}, {2, 3, 4}, {5, 4}, {8}}) {
    TestAct(place,
            "def",
            0.0,
            0.,
            "all",
            0.,
            0.0,
            DDim(dims),
            "mish",
            MISH,
            abs_error);
  }
}

TEST(Activation_softplus, precision) {
  Place place;
  float abs_error = 2e-5;
#if defined(LITE_WITH_NNADAPTER)
  place = TARGET(kNNAdapter);
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  abs_error = 1e-2;
#elif defined(NNADAPTER_WITH_HUAWEI_KIRIN_NPU)
  abs_error = 5e-2;
#elif defined(NNADAPTER_WITH_INTEL_OPENVINO)
  abs_error = 1e-5;
#else
  return;
#endif
#elif defined(LITE_WITH_X86)
  place = TARGET(kHost);
#elif defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#else
  return;
#endif
  for (auto dims : std::vector<std::vector<int64_t>>{
           {1, 3, 2, 4}, {2, 3, 4}, {5, 4}, {8}}) {
    TestAct(place,
            "def",
            0.0,
            0.,
            "all",
            0.,
            0.0,
            DDim(dims),
            "softplus",
            SOFTPLUS,
            abs_error);
  }
}

TEST(Activation_hard_swish, precision) {
  Place place;
  float abs_error = 2e-5;
#if defined(LITE_WITH_NNADAPTER)
  place = TARGET(kNNAdapter);
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  abs_error = 1e-2;
#elif defined(NNADAPTER_WITH_HUAWEI_KIRIN_NPU)
  abs_error = 5e-2;
#elif defined(NNADAPTER_WITH_VERISILICON_TIMVX)
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
#elif defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#elif defined(LITE_WITH_X86)
  place = TARGET(kX86);
#else
  return;
#endif

  for (auto dims : std::vector<std::vector<int64_t>>{
           {1, 3, 2, 4}, {2, 3, 4}, {5, 4}, {8}}) {
    TestAct(place,
            "def",
            0.01,
            6.,
            "all",
            0.,
            1.0,
            DDim(dims),
            "hard_swish",
            HARD_SWISH,
            abs_error);
  }
}

TEST(activation_reciprocal, precision) {
  Place place;
  float abs_error = 2e-5;

#if defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#else
  return;
#endif

  for (auto dims : std::vector<std::vector<int64_t>>{
           {1, 3, 2, 4}, {2, 3, 4}, {5, 4}, {8}}) {
    TestAct(place,
            "def",
            0.01,
            6.,
            "all",
            0.,
            1.0,
            DDim(dims),
            "reciprocal",
            RECIPROCAL,
            abs_error);
  }
}

TEST(Activation_thresholded_relu, precision) {
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
    TestAct(place,
            "def",
            0.01,
            6.,
            "all",
            0.,
            1.0,
            DDim(dims),
            "thresholded_relu",
            THRESHOLDED_RELU,
            abs_error);
  }
}

TEST(Activation_elu, precision) {
#ifdef LITE_WITH_ARM
  // "This operator's definition is different from Paddle."
  // "So the output has diff with Paddle. We need to fix it as soon as
  // possible."
  // "Host is fix, but arm is not."
  return;
  Place place(TARGET(kARM));

  for (auto dims : std::vector<std::vector<int64_t>>{
           {1, 3, 2, 4}, {2, 3, 4}, {5, 4}, {8}}) {
    TestAct(place, "def", 0.01, 6., "all", 0., 1.0, DDim(dims), "elu", ELU);
  }
#endif
}

TEST(Activation_softsign, precision) {
  Place place;
  float abs_error = 2e-5;
#if defined(LITE_WITH_X86)
  place = TARGET(kX86);
#else
  return;
#endif

  for (auto dims : std::vector<std::vector<int64_t>>{
           {1, 3, 2, 4}, {2, 3, 4}, {5, 4}, {8}}) {
    TestAct(place,
            "def",
            0.01,
            6.,
            "all",
            0.,
            1.0,
            DDim(dims),
            "relu",
            RELU,
            abs_error);
  }
}

TEST(Activation_abs, precision) {
  Place place;
  float abs_error = 2e-5;
#if defined(LITE_WITH_NNADAPTER)
  place = TARGET(kNNAdapter);
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  abs_error = 5e-2;
#elif defined(NNADAPTER_WITH_HUAWEI_KIRIN_NPU)
  abs_error = 5e-2;
#elif defined(NNADAPTER_WITH_QUALCOMM_QNN)
  abs_error = 1e-2;
#else
  return;
#endif
#else
  return;
#endif

  for (auto dims : std::vector<std::vector<int64_t>>{
           {1, 3, 2, 4}, {2, 3, 4}, {5, 4}, {8}}) {
    TestAct(place,
            "def",
            0.01,
            6.,
            "all",
            0.,
            1.0,
            DDim(dims),
            "abs",
            ABS,
            abs_error);
  }
}

TEST(Activation_hard_sigmoid_fp32, precision) {
  Place place;
  float abs_error = 2e-5;
#if defined(LITE_WITH_NNADAPTER)
  place = TARGET(kNNAdapter);
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  abs_error = 1e-2;
#elif defined(NNADAPTER_WITH_VERISILICON_TIMVX)
  abs_error = 1e-5;
#elif defined(NNADAPTER_WITH_INTEL_OPENVINO)
  abs_error = 1e-5;
#elif defined(NNADAPTER_WITH_QUALCOMM_QNN)
  abs_error = 1e-2;
#else
  return;
#endif
#elif defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#else
  return;
#endif

  for (auto dims : std::vector<std::vector<int64_t>>{{1, 3, 32, 32},
                                                     {1, 2, 3, 4},
                                                     {1, 3, 2, 4},
                                                     {2, 3, 4},
                                                     {5, 4},
                                                     {8}}) {
    TestAct<float>(place,
                   "def",
                   0.01,
                   6.,
                   "all",
                   0.,
                   1.0,
                   DDim(dims),
                   "hard_sigmoid",
                   HARD_SIGMOID,
                   abs_error);
  }
}

TEST(Activation_hard_sigmoid_fp32, performance) {
  Place place;
  float abs_error = 2e-5;
#if defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#else
  return;
#endif

  for (auto dims : std::vector<std::vector<int64_t>>{{1, 32, 544, 544}}) {
    TestActPerformance<float>(place,
                              "def",
                              0.01,
                              6.,
                              "all",
                              0.,
                              1.0,
                              DDim(dims),
                              "hard_sigmoid",
                              HARD_SIGMOID,
                              abs_error);
  }
}

#if defined(LITE_WITH_ARM) && defined(ENABLE_ARM_FP16)
TEST(Activation_relu_fp16, precision) {
  Place place(TARGET(kARM), PRECISION(kFP16));
  float abs_error = 2e-5;

  for (auto dims : std::vector<std::vector<int64_t>>{
           {1, 3, 2, 4}, {2, 3, 4}, {5, 4}, {8}}) {
    TestAct<float16_t>(place,
                       "def",
                       0.01,
                       6.,
                       "all",
                       0.,
                       1.0,
                       DDim(dims),
                       "relu",
                       RELU,
                       abs_error);
  }
}

TEST(Activation_hard_sigmoid_fp16, precision) {
  Place place;
  float abs_error = 2e-3;
#if defined(LITE_WITH_NNADAPTER)
  place = Place(TARGET(kNNAdapter), PRECISION(kFP16));
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  abs_error = 1e-2;
#elif defined(NNADAPTER_WITH_INTEL_OPENVINO)
  abs_error = 1e-5;
#else
  return;
#endif
#elif defined(LITE_WITH_ARM)
  place = Place(TARGET(kARM), PRECISION(kFP16));
#else
  return;
#endif

  for (auto dims : std::vector<std::vector<int64_t>>{{1, 3, 32, 32},
                                                     {1, 2, 3, 4},
                                                     {1, 3, 2, 4},
                                                     {2, 3, 4},
                                                     {5, 4},
                                                     {8}}) {
    TestAct<float16_t>(place,
                       "def",
                       0.01,
                       6.,
                       "all",
                       0.,
                       1.0,
                       DDim(dims),
                       "hard_sigmoid",
                       HARD_SIGMOID,
                       abs_error);
  }
}

TEST(Activation_prelu_fp16, precision) {
  Place place(TARGET(kARM), PRECISION(kFP16));
  float abs_error = 2e-3;

  for (auto dims : std::vector<std::vector<int64_t>>{
           {1, 2, 3, 4}, {1, 3, 2, 4}, {1, 1, 2, 32}}) {
    for (auto mode : {"all", "channel", "element"}) {
      TestAct<float16_t>(place,
                         "def",
                         0.01,
                         6,
                         mode,
                         0.,
                         1.0,
                         DDim(dims),
                         "prelu",
                         PRELU,
                         abs_error);
    }
  }
}

TEST(Activation_hard_sigmoid_fp16, performance) {
  Place place;
  float abs_error = 2e-3;
#if defined(LITE_WITH_ARM)
  place = Place(TARGET(kARM), PRECISION(kFP16));
#else
  return;
#endif

  for (auto dims : std::vector<std::vector<int64_t>>{{1, 3, 32, 32},
                                                     {1, 2, 3, 4},
                                                     {1, 3, 2, 4},
                                                     {2, 3, 4},
                                                     {5, 4},
                                                     {8}}) {
    TestActPerformance<float16_t>(place,
                                  "def",
                                  0.01,
                                  6.,
                                  "all",
                                  0.,
                                  1.0,
                                  DDim(dims),
                                  "hard_sigmoid",
                                  HARD_SIGMOID,
                                  abs_error);
  }
}

TEST(Activation_prelu_fp16, performance) {
  Place place(TARGET(kARM), PRECISION(kFP16));
  float abs_error = 2e-5;

  for (auto dims : std::vector<std::vector<int64_t>>{
           {1, 3, 32, 32}, {1, 2, 3, 4}, {1, 3, 2, 4}}) {
    for (auto mode : {"all", "channel", "element"}) {
      TestActPerformance<float16_t>(place,
                                    "def",
                                    0.01,
                                    6,
                                    mode,
                                    0.,
                                    1.0,
                                    DDim(dims),
                                    "prelu",
                                    PRELU,
                                    abs_error);
    }
  }
}

TEST(Activation_hard_swish_fp16, precision) {
  Place place;
  float abs_error = 2e-3;
#ifdef LITE_WITH_ARM
  place = Place(TARGET(kARM), PRECISION(kFP16));
#else
  return;
#endif
  for (auto dims : std::vector<std::vector<int64_t>>{{1, 3, 32, 32},
                                                     {1, 2, 3, 4},
                                                     {1, 3, 2, 4},
                                                     {2, 3, 4},
                                                     {5, 4},
                                                     {8}}) {
    TestAct<float16_t>(place,
                       "def",
                       0.01,
                       6.,
                       "all",
                       0.,
                       1.0,
                       DDim(dims),
                       "hard_swish",
                       HARD_SWISH,
                       abs_error);
  }
}

TEST(Activation_hard_swish_fp16, performance) {
  Place place;
  float abs_error = 2e-3;
#ifdef LITE_WITH_ARM
  place = Place(TARGET(kARM), PRECISION(kFP16));
#else
  return;
#endif
  for (auto dims : std::vector<std::vector<int64_t>>{{1, 3, 32, 32},
                                                     {1, 2, 3, 4},
                                                     {1, 3, 2, 4},
                                                     {2, 3, 4},
                                                     {5, 4},
                                                     {8}}) {
    TestActPerformance<float16_t>(place,
                                  "def",
                                  0.01,
                                  6.,
                                  "all",
                                  0.,
                                  1.0,
                                  DDim(dims),
                                  "hard_swish",
                                  HARD_SWISH,
                                  abs_error);
  }
}
#endif

}  // namespace lite
}  // namespace paddle
