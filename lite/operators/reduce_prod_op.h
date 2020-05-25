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
#include "lite/operators/op_params.h"

namespace paddle {
namespace lite {
namespace operators {

class ReduceProdOpLite : public OpLite {
 public:
  ReduceProdOpLite() {}

  explicit ReduceProdOpLite(const std::string &op_type) : OpLite(op_type) {}

  bool CheckShape() const override;

  bool InferShapeImpl() const override;

  bool AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) override;

  void AttachKernel(KernelBase *kernel) override { kernel->SetParam(param_); }

  std::string DebugString() const override { return "reduce_prod"; }

#ifdef LITE_WITH_PROFILE
  void GetOpRuntimeInfo(paddle::lite::profile::OpCharacter *ch) {
    ch->input_shape = ch->DimToStr(param_.x->dims());
    ch->output_shape = ch->DimToStr(param_.output->dims());
    ch->remark = "keep_dim" + std::to_string(param_.keep_dim) + "reduce_all" +
                 std::to_string(param_.reduce_all);

    auto dims = param_.dim;
    auto in_sum = param_.x->numel();
    if (dims.size() == 0 || dims.size() == 1) {
      ch->macs = 1.f * in_sum;
    } else if (dims.size() == 2) {
      ch->macs = 2.f * in_sum;
    } else {
      LOG(FATAL) << "This dims size of ReduceProd: " << dims.size()
                 << " doesn't support";
      ch->macs = 0.f;
    }
  }
#endif

 private:
  mutable ReduceParam param_;
};

}  // namespace operators
}  // namespace lite
}  // namespace paddle
