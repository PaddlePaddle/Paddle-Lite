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
#include "lite/core/kernel.h"
#include "lite/core/op_lite.h"
#include "lite/operators/op_params.h"
#include "lite/backends/xpu/xpu_header_sitter.h"

namespace paddle {
namespace lite {

namespace operators {

class ResNet50Op : public OpLite {
 public:
  ResNet50Op() {}
  explicit ResNet50Op(const std::string &op_type) : OpLite(op_type) {}

  bool CheckShape() const override;

  bool InferShape() const override;

  bool AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) override;

  void AttachKernel(KernelBase *kernel) override { kernel->SetParam(param_); }
  std::string DebugString() const override { return "ResNet50"; }

 private:
  mutable ResNet50Param param_;
};

}  // namespace operators

namespace kernels {
namespace xpu {

class ResNet50Compute : public KernelLite<TARGET(kXPU), PRECISION(kFloat)> {
 public:
  ResNet50Compute();

  using param_t = operators::ResNet50Param;

  virtual void PrepareForRun() override;

  virtual void Run() override;

 private:
  std::vector<const int16_t*> arg_filter_;
  std::vector<const float*> arg_max_filter_;
  std::vector<const float*> arg_bias_;
};

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
