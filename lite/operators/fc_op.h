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
#include "lite/core/kernel.h"
#include "lite/core/op_lite.h"
#include "lite/core/scope.h"
#include "lite/core/tensor.h"
#include "lite/operators/op_params.h"
#include "lite/utils/all.h"

namespace paddle {
namespace lite {
namespace operators {

class FcOpLite : public OpLite {
 public:
  FcOpLite() {}

  explicit FcOpLite(const std::string &type) : OpLite(type) {}

  bool CheckShape() const override;

  bool InferShapeImpl() const override;

  bool AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) override;

  void AttachKernel(KernelBase *kernel) override { kernel->SetParam(param_); }

  std::string DebugString() const override { return "fc"; }

#ifdef LITE_WITH_PROFILE
  void GetOpRuntimeInfo(paddle::lite::profile::OpCharacter *ch) {
    auto m = param_.input->dims().count(0, param_.in_num_col_dims);
    ch->input_shape = ch->DimToStr(param_.input->dims());
    ch->filter_shape = ch->DimToStr(param_.w->dims());
    ch->output_shape = ch->DimToStr(param_.output->dims());
    ch->remark = (param_.bias ? "Bias" : "") + param_.activation_type;
    ch->macs = m * param_.w->dims()[0] * param_.w->dims()[1] * 3.0f;
  }
#endif

 private:
  mutable FcParam param_;
};

}  // namespace operators
}  // namespace lite
}  // namespace paddle
