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

namespace paddle {
namespace lite {
namespace operators {

class LrnOpLite : public OpLite {
 public:
  LrnOpLite() {}
  explicit LrnOpLite(const std::string &op_type) : OpLite(op_type) {}

  bool CheckShape() const override;

  bool InferShapeImpl() const override;

  bool AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) override;

  void AttachKernel(KernelBase *kernel) override { kernel->SetParam(param_); }

  std::string DebugString() const override { return "lrn"; }

#ifdef LITE_WITH_PROFILE
  void GetOpRuntimeInfo(paddle::lite::profile::OpCharacter *ch) {
    ch->input_shape = ch->DimToStr(param_.X->dims());
    ch->output_shape = ch->DimToStr(param_.Out->dims());
    ch->remark = "n" + std::to_string(param_.n) + param_.norm_region;
    ch->macs = param_.Out->numel() * param_.k * 2.f;
  }
#endif

 private:
  mutable LrnParam param_;
};

}  // namespace operators
}  // namespace lite
}  // namespace paddle
