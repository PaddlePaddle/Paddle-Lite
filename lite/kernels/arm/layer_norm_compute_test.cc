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
#include <limits>
#include <vector>

#include "lite/core/op_registry.h"
#include "lite/kernels/arm/layer_norm_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void LayerNormComputeRef(const operators::LayerNormParam& param) {
  auto* x = param.X;
  auto* y = param.Y;
  auto* scale_tensor = param.Scale;
  auto* bias_tensor = param.Bias;
  auto* mean_tensor = param.Mean;
  auto* var_tensor = param.Variance;

  int begin_norm_axis = param.begin_norm_axis;
  float epsilon = param.epsilon;

  auto* x_data = x->data<float>();
  auto* scale_data =
      (scale_tensor == nullptr ? nullptr : scale_tensor->data<float>());
  auto* bias_data =
      (bias_tensor == nullptr ? nullptr : bias_tensor->data<float>());
  auto* out_data = y->mutable_data<float>();
  auto* mean_data = mean_tensor->mutable_data<float>();
  auto* var_data = var_tensor->mutable_data<float>();

  auto matrix_dim = x->dims().Flatten2D(begin_norm_axis);
  int batch_size = matrix_dim[0];
  int feature_size = matrix_dim[1];
  for (int i = 0; i < batch_size; ++i) {
    int start = i * feature_size;
    int end = start + feature_size;

    float mean = 0;
    float var = 0;
    for (int j = start; j < end; ++j) {
      mean += x_data[j];
      var += x_data[j] * x_data[j];
    }
    mean /= feature_size;
    var = var / feature_size - mean * mean;
    mean_data[i] = mean;
    var_data[i] = var;
    var = sqrt(var + epsilon);
    for (int j = start; j < end; ++j) {
      out_data[j] = (x_data[j] - mean) / var;
      if (scale_data) {
        out_data[j] *= scale_data[j - start];
      }
      if (bias_data) {
        out_data[j] += bias_data[j - start];
      }
    }
  }
}

TEST(layer_norm_arm, init) {
  LayerNormCompute layer_norm;
  ASSERT_EQ(layer_norm.precision(), PRECISION(kFloat));
  ASSERT_EQ(layer_norm.target(), TARGET(kARM));
}

TEST(layer_norm_arm, compute) {
  LayerNormCompute layer_norm;
  operators::LayerNormParam param;

  lite::Tensor x;
  lite::Tensor output;
  lite::Tensor output_mean;
  lite::Tensor output_var;
  lite::Tensor output_ref;
  lite::Tensor output_mean_ref;
  lite::Tensor output_var_ref;
  lite::Tensor bias;
  lite::Tensor scale;
  lite::Tensor* bias_ptr;
  lite::Tensor* scale_ptr;

  for (auto n : {1, 3}) {
    for (auto c : {1, 3, 5}) {
      for (auto h : {3, 16, 20, 32}) {
        for (auto w : {3, 16, 20, 32}) {
          for (auto axis : {0, 1, 2}) {
            for (auto has_bias : {true, false}) {
              for (auto has_scale : {true, false}) {
                auto dims = DDim(std::vector<int64_t>({n, c, h, w}));
                auto out_size = dims.Flatten2D(axis)[0];
                auto inner_size = dims.Flatten2D(axis)[1];
                bias_ptr = nullptr;
                scale_ptr = nullptr;
                if (has_bias) {
                  bias.Resize(std::vector<int64_t>({inner_size, 1, 1, 1}));
                  float* bias_data = bias.mutable_data<float>();
                  for (int i = 0; i < inner_size; ++i) {
                    bias_data[i] = 0.01;
                  }
                  bias_ptr = &bias;
                }
                if (has_scale) {
                  scale.Resize(std::vector<int64_t>({inner_size, 1, 1, 1}));
                  float* scale_data = scale.mutable_data<float>();
                  for (int i = 0; i < inner_size; ++i) {
                    scale_data[i] = 0.2;
                  }
                  scale_ptr = &scale;
                }

                x.Resize(dims);
                output.Resize(DDim(std::vector<int64_t>({n, c, h, w})));
                output_ref.Resize(DDim(std::vector<int64_t>({n, c, h, w})));
                output_mean.Resize(std::vector<int64_t>({out_size, 1, 1, 1}));
                output_mean_ref.Resize(
                    std::vector<int64_t>({out_size, 1, 1, 1}));
                output_var.Resize(std::vector<int64_t>({out_size, 1, 1, 1}));
                output_var_ref.Resize(
                    std::vector<int64_t>({out_size, 1, 1, 1}));

                auto* x_data = x.mutable_data<float>();
                auto* output_data = output.mutable_data<float>();
                auto* output_mean_data = output_mean.mutable_data<float>();
                auto* output_var_data = output_var.mutable_data<float>();
                auto* output_data_ref = output_ref.mutable_data<float>();
                auto* output_mean_data_ref =
                    output_mean_ref.mutable_data<float>();
                auto* output_var_data_ref =
                    output_var_ref.mutable_data<float>();

                for (int i = 0; i < x.dims().production(); i++) {
                  x_data[i] = i % 255 * 0.001;
                }
                param.X = &x;
                param.Y = &output;
                param.begin_norm_axis = axis;
                param.Bias = bias_ptr;
                param.Scale = scale_ptr;
                param.Mean = &output_mean;
                param.Variance = &output_var;
                param.epsilon = 0.00001;
                layer_norm.SetParam(param);
                layer_norm.Run();

                param.Y = &output_ref;
                param.Mean = &output_mean_ref;
                param.Variance = &output_var_ref;
                LayerNormComputeRef(param);
                for (int i = 0; i < output.dims().production(); i++) {
                  EXPECT_NEAR(output_data[i], output_data_ref[i], 1e-4);
                }
                for (int i = 0; i < output_mean.dims().production(); ++i) {
                  EXPECT_NEAR(
                      output_mean_data[i], output_mean_data_ref[i], 1e-5);
                  EXPECT_NEAR(output_var_data[i], output_var_data_ref[i], 1e-5);
                }
              }
            }
          }
        }
      }
    }
  }
}

TEST(layer_norm, retrive_op) {
  auto layer_norm = KernelRegistry::Global().Create("layer_norm");
  ASSERT_FALSE(layer_norm.empty());
  ASSERT_TRUE(layer_norm.front());
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(layer_norm, kARM, kFloat, kNCHW, def);
