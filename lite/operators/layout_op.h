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
#include "lite/core/op_lite.h"

namespace paddle {
namespace lite {
namespace operators {

class LayoutOp : public OpLite {
 public:
  explicit LayoutOp(const std::string &type) : OpLite(type) {}
  bool CheckShape() const override;
  bool InferShapeImpl() const override;
  bool Run() override;
  std::string DebugString() const override;

  void AttachKernel(KernelBase *kernel) override { kernel->SetParam(param_); }

#ifdef LITE_WITH_PROFILE
  void GetOpRuntimeInfo(paddle::lite::profile::OpCharacter *ch) {
    auto input_dims = param_.x->dims();
    auto output_dims = param_.y->dims();
    ch->input_shape = ch->DimToStr(input_dims);
    ch->output_shape = ch->DimToStr(output_dims);
    ch->remark = "type" + std::to_string(param_.process_type);
  }
#endif

 protected:
  bool AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) override;

 private:
  operators::LayoutParam param_;
};

}  // namespace operators
}  // namespace lite
}  // namespace paddle
