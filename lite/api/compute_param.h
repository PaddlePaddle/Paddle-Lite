// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once
#include <memory>
#include <string>
#include <vector>
#include "paddle_api.h"    // NOLINT
#include "paddle_place.h"  // NOLINT

namespace paddle {
namespace lite_api {

class LITE_API ParamBase {
 public:
  PrecisionType out_ptype{PRECISION(kFloat)};
  virtual int GetKernelIndex() { return 0; }
  virtual void* AttachRawParam() {}
  virtual ~ParamBase() = default;
};

class LITE_API ActivationParam : public ParamBase {
 public:
  Tensor* X{};
  Tensor* Out{};
  ActivationType active_type{ActivationType::kIndentity};
  bool has_active{false};
  float Leaky_relu_alpha{0};   // leaky_relu param
  float Relu_clipped_coef{6};  // relu_clipped param
  const char* Prelu_mode{
      "channel"};         // prelu param, can be "all", "channel" or "element"
  Tensor* Prelu_alpha{};  // prelu param
  float Swish_beta;       // swish param
  // hard_sigmoid param
  float hard_sigmoid_slope{0.2f};
  float hard_sigmoid_offset{0.5f};
  // hard_swish param
  float hard_swish_threshold{6.0};
  float hard_swish_scale{6.0};
  float hard_swish_offset{3.0};

  ActivationParam() = default;
  virtual ~ActivationParam() = default;
  void* AttachRawParam() override;
};

class LITE_API ConvParam : public ParamBase {
 public:
  Tensor* x{};
  Tensor* filter{};
  Tensor* bias{nullptr};
  Tensor* residualData{nullptr};
  Tensor* output{};
  std::vector<int> strides{1, 1};
  std::shared_ptr<std::vector<int>> paddings;
  int groups{1};
  std::shared_ptr<std::vector<int>> dilations;
  bool fuse_residual_connection{false};
  const char* data_format{"Anylayout"};
  // for activation
  ActivationParam activation_param;
  // only used in conv_transpose.
  std::vector<int> output_size;
  // for int8
  bool enable_int8{false};
  float input_scale{1.0f};
  std::vector<float> weight_scale{};
  float output_scale{1.0f};
  int bit_length{8};

  ConvParam() = default;
  virtual ~ConvParam() = default;
  void* AttachRawParam() override;
  int GetKernelIndex() override;
};

}  // namespace lite_api
}  // namespace paddle
