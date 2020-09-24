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

#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

class XPUResNet50Compute : public KernelLite<TARGET(kXPU), PRECISION(kFloat)> {
 public:
  using param_t = operators::XPUResNet50Param;

  virtual void PrepareForRun();

  virtual void Run();

 private:
  std::vector<const int16_t *> arg_filter_;
  std::vector<const float *> arg_max_filter_;
  std::vector<const float *> arg_bias_;
};

class XPUResNet50DtypeCompute
    : public KernelLite<TARGET(kXPU), PRECISION(kFloat)> {
 public:
  using param_t = operators::XPUResNet50Param;

  virtual void PrepareForRun();

  virtual void Run();

 private:
  std::vector<const int16_t *> arg_filter_;
  std::vector<const float *> arg_max_filter_;
  std::vector<const float *> arg_bias_;
};

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
