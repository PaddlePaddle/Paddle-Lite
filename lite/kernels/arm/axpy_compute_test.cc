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

#include <cstdlib>
#include <memory>
#include <utility>
#include <vector>

#include "lite/core/op_registry.h"
#include "lite/kernels/arm/axpy_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

template <typename dtype>
void axpy_compute_ref(const operators::AxpyParam& param) {
  lite::Tensor* scale = param.Scale;
  lite::Tensor* x = param.X;
  lite::Tensor* bias = param.Bias;
  lite::Tensor* output = param.Out;

  auto scale_data = scale->data<dtype>();
  auto x_data = x->data<dtype>();
  auto bias_data = bias->data<dtype>();
  auto output_data = output->mutable_data<dtype>();

  DDim x_dims = x->dims();
  int num = x_dims[0];
  int channel = x_dims[1];
  int size = x_dims[2] * x_dims[3];
  int in_channel = channel * size;

  for (int i = 0; i < num; i++) {
    auto scale_data_i = scale_data + i * channel;
    auto x_data_i = x_data + i * in_channel;
    auto bias_data_i = bias_data + i * in_channel;
    auto output_data_i = output_data + i * in_channel;
    for (int j = 0; j < channel; j++) {
      auto scale_data_j = scale_data_i + j;
      auto x_data_j = x_data_i + j * size;
      auto bias_data_j = bias_data_i + j * size;
      auto output_data_j = output_data_i + j * size;
      for (int k = 0; k < size; k++) {
        output_data_j[k] = scale_data_j[0] * x_data_j[k] + bias_data_j[k];
      }
    }
  }
}

TEST(axpy_arm, retrive_op) {
  auto axpy = KernelRegistry::Global().Create("axpy");
  ASSERT_FALSE(axpy.empty());
  ASSERT_TRUE(axpy.front());
}

TEST(axpy_arm, init) {
  AxpyCompute axpy;
  ASSERT_EQ(axpy.precision(), PRECISION(kFloat));
  ASSERT_EQ(axpy.target(), TARGET(kARM));
}
TEST(axpy_arm, compute) {
  DeviceInfo::Init();
  int iter = 10;
  for (int i = 0; i < iter; i++) {
    Tensor scale;
    Tensor x;
    Tensor bias;
    Tensor output;
    Tensor output_ref;

    // set the dims of scale, x, bias and output_ref
    int n = 2, c = 3, h = 4, w = 5;
    scale.Resize({n, c});
    x.Resize({n, c, h, w});
    bias.Resize({n, c, h, w});
    output.Resize({n, c, h, w});
    output_ref.Resize({n, c, h, w});

    // initialize the data of scale, x, bias
    // initialize_random_data<float>(scale);
    // initialize_random_data<float>(x);
    // initialize_random_data<float>(bias);
    auto* scale_data = scale.mutable_data<float>();
    for (int i = 0; i < scale.dims().production(); i++) {
      float sign = i % 3 == 0 ? -1.0f : 1.0f;
      scale_data[i] = sign * static_cast<float>(i % 128) * 0.010f;
    }
    auto* x_data = x.mutable_data<float>();
    for (int i = 0; i < x.dims().production(); i++) {
      float sign = i % 4 == 0 ? -1.0f : 1.0f;
      x_data[i] = sign * static_cast<float>(i % 128) * 0.007f;
    }
    auto* bias_data = bias.mutable_data<float>();
    for (int i = 0; i < bias.dims().production(); i++) {
      float sign = i % 5 == 0 ? -1.0f : 1.0f;
      bias_data[i] = sign * static_cast<float>(i % 128) * 0.005f;
    }

    // prepare kernel params and run to obtain output_data
    AxpyCompute axpy_op;
    std::unique_ptr<KernelContext> ctx(new KernelContext);
    ctx->As<ARMContext>();
    axpy_op.SetContext(std::move(ctx));
    operators::AxpyParam param;
    param.Scale = &scale;
    param.X = &x;
    param.Bias = &bias;
    param.Out = &output;
    axpy_op.SetParam(param);
    axpy_op.Launch();
    auto* output_data = output.mutable_data<float>();

    // invoking ref implementation and compare results
    param.Out = &output_ref;
    axpy_compute_ref<float>(param);
    auto* output_ref_data = output_ref.mutable_data<float>();

    for (int i = 0; i < output.dims().production(); i++) {
      EXPECT_NEAR(output_data[i], output_ref_data[i], 1e-5);
    }
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
USE_LITE_KERNEL(axpy, kARM, kFloat, kNCHW, def);
