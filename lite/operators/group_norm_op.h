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

class GroupNormOp : public OpLite {
 public:
  GroupNormOp() {}

  explicit GroupNormOp(const std::string &op_type) : OpLite(op_type) {}

  bool CheckShape() const override;

  bool InferShapeImpl() const override;

  bool AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) override;

  void AttachKernel(KernelBase *kernel) override { kernel->SetParam(param_); }

  std::string DebugString() const override { return "group_norm"; }

#ifdef LITE_WITH_PROFILE
  void GetOpRuntimeInfo(paddle::lite::profile::OpCharacter *ch) {
    ch->input_shape = ch->DimToStr(param_.x->dims());
    ch->output_shape = ch->DimToStr(param_.out->dims());
    // ch->remark = "";
    auto x_dims = param_.x->dims();
    auto nc = x_dims[0] * x_dims[1];
    auto hw = x_dims[2] * x_dims[3];
    auto nchw = x_dims.production();
    ch->macs = 5.f * nchw + 3.f * (nc + hw);
  }
#endif

 private:
  mutable GroupNormParam param_;
};

} /* namespace operators */
} /* namespace lite */
} /* namespace paddle */
