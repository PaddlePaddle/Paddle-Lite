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

#include "lite/kernels/arm/activation_compute.h"
#include <gtest/gtest.h>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

enum activation_type {
  RELU,
  RELU_NEG,
  RELU_CLIPPED,
  PRELU,
  SIGMOID,
  TANH,
  SWISH
};

template <typename dtype>
void activation_compute_ref(const operators::ActivationParam& param,
                            activation_type type) {
  auto x_data = param.x->data<dtype>();
  auto output_data = param.out->mutable_data<dtype>();
  DDim x_dims = param.x->dims();
  DDim output_dims = param.out->dims();
  ASSERT_EQ(x_dims.data(), output_dims.data());

  switch (type) {
    case RELU: {
      for (int i = 0; i < output_dims.production(); i++) {
        output_data[i] = std::max(0.f, x_data[i]);
      }
      break;
    }
    case RELU_NEG: {
      float neg_slope = param.relu_neg_slope;
      for (int i = 0; i < output_dims.production(); i++) {
        output_data[i] = x_data[i] > 0.f ? x_data[i] : x_data[i] * neg_slope;
      }
      break;
    }
    case RELU_CLIPPED: {
      float clipped_coef = param.relu_clipped_coef;
      for (int i = 0; i < output_dims.production(); i++) {
        output_data[i] = x_data[i] > 0.f ? x_data[i] : 0.f;
        output_data[i] =
            output_data[i] < clipped_coef ? output_data[i] : clipped_coef;
      }
      break;
    }
    case PRELU: {
      bool channel_shared = param.prelu_channel_shared;
      auto channel_slope = param.prelu_channel_slope->data<dtype>();
      int num = x_dims[0];
      int channel = x_dims[1];
      int csize = x_dims[2] * x_dims[3];
      int bsize = channel * csize;
      for (int n = 0; n < num; n++) {
        auto x_data_bptr = x_data + n * bsize;
        auto output_data_bptr = output_data + n * bsize;
        for (int c = 0; c < channel; c++) {
          auto x_data_cptr = x_data_bptr + c * csize;
          auto output_data_cptr = output_data_bptr + c * csize;
          float slope = channel_shared ? channel_slope[0] : channel_slope[c];
          for (int i = 0; i < csize; i++) {
            output_data_cptr[i] =
                x_data_cptr[i] > 0.f ? x_data_cptr[i] : x_data_cptr[i] * slope;
          }
        }
      }
      break;
    }
    case SIGMOID: {
      for (int i = 0; i < output_dims.production(); i++) {
        output_data[i] = 1.f / (1.f + std::exp(-x_data[i]));
      }
      break;
    }
    case TANH: {
      for (int i = 0; i < output_dims.production(); i++) {
        output_data[i] = (std::exp(x_data[i]) - std::exp(-x_data[i])) /
                         (std::exp(x_data[i]) + std::exp(-x_data[i]));
      }
      break;
    }
    case SWISH: {
      float coef = param.swish_coef;
      for (int i = 0; i < output_dims.production(); i++) {
        output_data[i] = x_data[i] / (1.f + std::exp(-coef * x_data[i]));
      }
      break;
    }
    default:
      LOG(INFO) << "the type of activation is wrong";
  }
}

void test_activation_compute(
    KernelLite<TARGET(kARM), PRECISION(kFloat)>* activation,
    operators::ActivationParam* param,
    activation_type type) {
  DeviceInfo::Init();
  for (auto n : {1, 2}) {
    for (auto c : {6, 32 /*, 128*/}) {
      for (auto h : {9, 18 /*, 56 , 112, 224, 512*/}) {
        for (auto w : {9, 18 /*, 56, 112, 224, 512*/}) {
          Tensor x;
          Tensor output;
          Tensor output_ref;
          // set the dims of input, output, ref output tensors
          x.Resize({n, c, h, w});
          output.Resize({n, c, h, w});
          output_ref.Resize({n, c, h, w});
          // initialize the data of input tensors
          auto* x_data = x.mutable_data<float>();
          auto* output_data = output.mutable_data<float>();
          for (int i = 0; i < x.dims().production(); i++) {
            float sign = i % 3 == 0 ? -1.0f : 1.0f;
            x_data[i] = sign * static_cast<float>(i % 128) * 0.013f;
          }
          // prepare kernel params and run
          std::unique_ptr<KernelContext> ctx(new KernelContext);
          ctx->As<ARMContext>();
          activation->SetContext(std::move(ctx));
          param->x = &x;
          param->out = &output;
          if (type == PRELU) {
            Tensor channel_slope;
            channel_slope.Resize({c});
            auto* channel_slope_data = channel_slope.mutable_data<float>();
            for (int j = 0; j < channel_slope.dims().production(); j++) {
              float sign = j % 3 == 0 ? -1.0f : 1.0f;
              channel_slope_data[j] =
                  sign * static_cast<float>(j % 128) * 0.013f;
            }
            param->prelu_channel_slope = &channel_slope;
          }
          activation->SetParam(*param);
          activation->Launch();
          // invoking ref implementation and compare results
          param->out = &output_ref;
          activation_compute_ref<float>(*param, type);
          auto* output_ref_data = output_ref.mutable_data<float>();

          for (int i = 0; i < output.dims().production(); i++) {
            EXPECT_NEAR(output_data[i], output_ref_data[i], 1e-5);
          }
        }
      }
    }
  }
}

TEST(activation_arm, retrive_op) {
  for (auto activation_name : {"relu",
                               "relu_neg",
                               "relu_clipped",
                               "prelu",
                               "sigmoid",
                               "tanh",
                               "swish"}) {
    LOG(INFO) << "test " << activation_name;
    auto activation =
        KernelRegistry::Global().Create<TARGET(kARM), PRECISION(kFloat)>(
            activation_name);
    ASSERT_FALSE(activation.empty());
    ASSERT_TRUE(activation.front());
  }
}

TEST(activation_arm, init) {
  ReluCompute activation_relu;
  ASSERT_EQ(activation_relu.precision(), PRECISION(kFloat));
  ASSERT_EQ(activation_relu.target(), TARGET(kARM));

  ReluNegCompute activation_relu_neg;
  ASSERT_EQ(activation_relu_neg.precision(), PRECISION(kFloat));
  ASSERT_EQ(activation_relu_neg.target(), TARGET(kARM));

  ReluClippedCompute activation_relu_clipped;
  ASSERT_EQ(activation_relu_clipped.precision(), PRECISION(kFloat));
  ASSERT_EQ(activation_relu_clipped.target(), TARGET(kARM));

  PReluCompute activation_prelu;
  ASSERT_EQ(activation_prelu.precision(), PRECISION(kFloat));
  ASSERT_EQ(activation_prelu.target(), TARGET(kARM));

  SigmoidCompute activation_sigmoid;
  ASSERT_EQ(activation_sigmoid.precision(), PRECISION(kFloat));
  ASSERT_EQ(activation_sigmoid.target(), TARGET(kARM));

  TanhCompute activation_tanh;
  ASSERT_EQ(activation_tanh.precision(), PRECISION(kFloat));
  ASSERT_EQ(activation_tanh.target(), TARGET(kARM));

  SwishCompute activation_swish;
  ASSERT_EQ(activation_swish.precision(), PRECISION(kFloat));
  ASSERT_EQ(activation_swish.target(), TARGET(kARM));
}

TEST(relu_activation_arm, compute) {
  ReluCompute activation;
  operators::ActivationParam param;
  activation_type type = RELU;
  test_activation_compute(&activation, &param, type);
}

TEST(relu_neg_activation_arm, compute) {
  ReluNegCompute activation;
  operators::ActivationParam param;
  for (float slope : {0.001, 0.01, 0.1}) {
    param.relu_neg_slope = slope;
    activation_type type = RELU_NEG;
    test_activation_compute(&activation, &param, type);
  }
}

TEST(relu_clipped_activation_arm, compute) {
  ReluClippedCompute activation;
  operators::ActivationParam param;
  for (float coef : {1, 3, 6}) {
    param.relu_clipped_coef = coef;
    activation_type type = RELU_CLIPPED;
    test_activation_compute(&activation, &param, type);
  }
}

TEST(prelu_activation_arm, compute) {
  PReluCompute activation;
  operators::ActivationParam param;
  for (bool flag : {false, true}) {
    param.prelu_channel_shared = flag;
    activation_type type = PRELU;
    test_activation_compute(&activation, &param, type);
  }
}

TEST(sigmoid_activation_arm, compute) {
  SigmoidCompute activation;
  operators::ActivationParam param;
  activation_type type = SIGMOID;
  test_activation_compute(&activation, &param, type);
}

TEST(tanh_activation_arm, compute) {
  TanhCompute activation;
  operators::ActivationParam param;
  activation_type type = TANH;
  test_activation_compute(&activation, &param, type);
}

TEST(swish_activation_arm, compute) {
  SwishCompute activation;
  operators::ActivationParam param;
  for (float coef : {0.01, 0.1}) {
    param.swish_coef = coef;
    activation_type type = SWISH;
    test_activation_compute(&activation, &param, type);
  }
}
}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
