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

class SplitOp : public OpLite {
 public:
  SplitOp() {}
  explicit SplitOp(const std::string &op_type) : OpLite(op_type) {}

  bool CheckShape() const override;

  bool InferShapeImpl() const override;

  bool InferShapeWithCache() const override { return false; }

  bool AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) override;

  void AttachKernel(KernelBase *kernel) override { kernel->SetParam(param_); }
  std::string DebugString() const override { return "split"; }

#ifdef LITE_WITH_PROFILE
  void GetOpRuntimeInfo(paddle::lite::profile::OpCharacter *ch) {
    ch->input_shape = ch->DimToStr(param_.x->dims());

    std::string outputs_shape = "";
    for (size_t i = 0; i < param_.output.size(); ++i) {
      outputs_shape += ch->DimToStr(param_.output[i]->dims());
      if (i != param_.output.size() - 1) outputs_shape += "/";
    }
    ch->output_shape = outputs_shape;

    std::string sections = "";
    for (size_t i = 0; i < param_.sections.size(); ++i) {
      sections += std::to_string(param_.sections[i]);
      if (i != param_.sections.size() - 1) sections += "/";
    }
    ch->remark = "axis" + std::to_string(param_.axis) + "num" +
                 std::to_string(param_.num) + "sections" + sections;
  }
#endif

 private:
  mutable SplitParam param_;
};

}  // namespace operators
}  // namespace lite
}  // namespace paddle
