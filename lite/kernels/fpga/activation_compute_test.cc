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

#include "lite/kernels/fpga/activation_compute.h"
#include <gtest/gtest.h>
#include <memory>
#include <utility>
#include <vector>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace fpga {

template <typename dtype>
void activation_compute_ref(const operators::ActivationParam& param) {
  auto x_data = param.X->data<dtype>();
  auto output_data = param.Out->mutable_data<dtype>();
  DDim x_dims = param.X->dims();
  DDim output_dims = param.Out->dims();
  ASSERT_EQ(x_dims.data(), output_dims.data());
  for (int i = 0; i < output_dims.production(); i++) {
    output_data[i] = std::max(0.f, x_data[i]);
  }
}

TEST(activation_fpga, retrive_op) {
  auto activation =
      KernelRegistry::Global().Create<TARGET(kFPGA), PRECISION(kFP16)>("relu");
  ASSERT_FALSE(activation.empty());
  ASSERT_TRUE(activation.front());
}

TEST(activation_fpga, init) {
  ReluCompute activation;
  ASSERT_EQ(activation.precision(), PRECISION(kFP16));
  ASSERT_EQ(activation.target(), TARGET(kFPGA));
}

TEST(activation_fpga, compute) {
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
          ReluCompute activation;
          operators::ActivationParam param;
          param.X = &x;
          param.Out = &output;
          activation.SetParam(param);
          activation.Launch();
          // invoking ref implementation and compare results
          param.Out = &output_ref;
          activation_compute_ref<float>(param);
          auto* output_ref_data = output_ref.mutable_data<float>();
          for (int i = 0; i < output.dims().production(); i++) {
            EXPECT_NEAR(output_data[i], output_ref_data[i], 1e-5);
          }
        }
      }
    }
  }
}

}  // namespace fpga
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(relu, kFPGA, kFP16, kNHWC, def);
