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

class ConcatOpLite : public OpLite {
 public:
  ConcatOpLite() {}
  explicit ConcatOpLite(const std::string &op_type) : OpLite(op_type) {}

  bool CheckShape() const override;

  bool InferShapeImpl() const override;

  bool AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) override;

  void AttachKernel(KernelBase *kernel) override { kernel->SetParam(param_); }
  std::string DebugString() const override { return "concat"; }

#ifdef LITE_WITH_PROFILE
  void GetOpRuntimeInfo(paddle::lite::profile::OpCharacter *ch) {
    auto output_dims = param_.output->dims();
    std::string inputs_shape = "";
    for (size_t i = 0; i < param_.x.size(); ++i) {
      inputs_shape += ch->DimToStr(param_.x[i]->dims());
      if (i != param_.x.size() - 1) inputs_shape += "/";
    }
    ch->input_shape = inputs_shape;
    ch->output_shape = ch->DimToStr(output_dims);
    ch->remark = "axis" + std::to_string(param_.axis);
    ch->macs = 0.f;  // no calc. only io operation
  }
#endif

 private:
  mutable ConcatParam param_;
};

}  // namespace operators
}  // namespace lite
}  // namespace paddle
