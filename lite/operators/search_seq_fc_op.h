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

class SearchSeqFcOpLite : public OpLite {
 public:
  SearchSeqFcOpLite() {}

  explicit SearchSeqFcOpLite(const std::string &type) : OpLite(type) {}

  bool CheckShape() const override;

  bool InferShapeImpl() const override;

  void AttachKernel(KernelBase *kernel) override { kernel->SetParam(param_); }

  bool AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) override;

  std::string DebugString() const override { return "search_seq_fc"; }

#ifdef LITE_WITH_PROFILE
  void GetOpRuntimeInfo(paddle::lite::profile::OpCharacter *ch) {
    ch->input_shape = ch->DimToStr(param_.x->dims());
    ch->filter_shape = ch->DimToStr(param_.w->dims());
    ch->output_shape = ch->DimToStr(param_.out->dims());
    ch->remark = "out_size" + std::to_string(param_.out_size);
    auto x_dims = param_.x->dims();
    auto w_dims = param_.w->dims();
    ch->macs = 2.f * x_dims[0] * x_dims[1] * w_dims[0];
  }
#endif

 private:
  mutable SearchSeqFcParam param_;
};

}  // namespace operators
}  // namespace lite
}  // namespace paddle
