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
#include <string>
#include <vector>
#include "lite/core/op_lite.h"
#include "lite/core/scope.h"
#include "lite/utils/all.h"

namespace paddle {
namespace lite {
namespace operators {

class SequencePoolGradOp : public OpLite {
 public:
  SequencePoolGradOp() {}
  explicit SequencePoolGradOp(const std::string &op_type) : OpLite(op_type) {}
  bool CheckShape() const override;
  bool InferShapeImpl() const override;
  bool AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) override;

  void AttachKernel(KernelBase *kernel) override { kernel->SetParam(param_); }
  std::string DebugString() const override { return "sequence_pool_grad"; }

 private:
  mutable SequencePoolGradParam param_;
};

}  // namespace operators
}  // namespace lite
}  // namespace paddle
