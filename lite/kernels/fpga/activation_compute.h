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

#pragma once
#include <algorithm>
#include <map>
#include <string>
#include "lite/backends/fpga/KD/float16.hpp"
#include "lite/backends/fpga/KD/pes/relu_pe.hpp"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace fpga {

static std::map<std::string, zynqmp::ActiveType> activation_map = {
    {"relu", zynqmp::TYPE_RELU},
    {"relu6", zynqmp::TYPE_RELU6},
    {"leaky_relu", zynqmp::TYPE_LEAKY_RELU},
    {"sigmoid", zynqmp::TYPE_SIGMOID},
    {"", zynqmp::TYPE_NONE}};

class ReluCompute
    : public KernelLite<TARGET(kFPGA), PRECISION(kFP16), DATALAYOUT(kNHWC)> {
 public:
  using param_t = operators::ActivationParam;

  void Run() override;
  void PrepareForRun() override;

  virtual ~ReluCompute() = default;

 private:
  zynqmp::ReluPE pe_;
  zynqmp::Tensor input_;
  zynqmp::Tensor output_;
};

class SigmoidCompute
    : public KernelLite<TARGET(kFPGA), PRECISION(kFP16), DATALAYOUT(kNHWC)> {
 public:
  using param_t = operators::ActivationParam;

  void Run() override;

  virtual ~SigmoidCompute() = default;
};

}  // namespace fpga
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
