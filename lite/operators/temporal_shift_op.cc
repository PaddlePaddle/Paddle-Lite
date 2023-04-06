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

#include "lite/operators/temporal_shift_op.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool TemporalShiftOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Out);
  // seg_num must > 0
  CHECK_OR_FALSE(param_.seg_num > 0);
  // shift_radio must in [0, 0.5]
  CHECK_OR_FALSE(param_.shift_ratio >= 0.0f && param_.shift_ratio <= 0.5f);
  CHECK(param_.data_format == "NCHW" || param_.data_format == "NHWC")
      << "Invilid data format.";
  return true;
}

bool TemporalShiftOpLite::InferShapeImpl() const { return true; }

bool TemporalShiftOpLite::AttachImpl(const cpp::OpDesc &op_desc,
                                     lite::Scope *scope) {
  param_.X = scope->FindVar(op_desc.Input("X").front())->GetMutable<Tensor>();
  param_.Out =
      scope->FindVar(op_desc.Output("Out").front())->GetMutable<Tensor>();

  if (op_desc.HasAttr("seg_num")) {
    param_.seg_num = op_desc.GetAttr<int>("seg_num");
  }
  if (op_desc.HasAttr("shift_ratio")) {
    param_.shift_ratio = op_desc.GetAttr<float>("shift_ratio");
  }
  if (op_desc.HasAttr("data_format")) {
    param_.data_format = op_desc.GetAttr<std::string>("data_format");
  }
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(temporal_shift, paddle::lite::operators::TemporalShiftOpLite);
