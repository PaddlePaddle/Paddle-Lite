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

#include "lite/operators/search_grnn_op.h"
#include <vector>
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool SearchGrnnOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.x);
  CHECK_OR_FALSE(param_.wi);
  CHECK_OR_FALSE(param_.wh);
  CHECK_OR_FALSE(param_.out);
  CHECK_OR_FALSE(param_.tmp_buffer);
  CHECK_OR_FALSE(param_.idx_sorted_by_width);
  CHECK_OR_FALSE(param_.layout_input);

  int _cap_h = param_.num_hidden;
  int _cap_e = param_.num_input;

  const auto& x_dims = param_.x->dims();
  CHECK_OR_FALSE(x_dims.size() == 2);
  CHECK_OR_FALSE(x_dims[1] == _cap_e);

  const auto& wi_dims = param_.wi->dims();
  CHECK_OR_FALSE(wi_dims.size() == 3);
  CHECK_OR_FALSE(wi_dims[0] == 3);
  CHECK_OR_FALSE(wi_dims[1] == _cap_h);
  CHECK_OR_FALSE(wi_dims[2] == _cap_e);

  const auto& wh_dims = param_.wh->dims();
  CHECK_OR_FALSE(wh_dims.size() == 3);
  CHECK_OR_FALSE(wh_dims[0] == 3);
  CHECK_OR_FALSE(wh_dims[1] == _cap_h);
  CHECK_OR_FALSE(wh_dims[2] == _cap_h);

  return true;
}

bool SearchGrnnOpLite::InferShapeImpl() const {
  const auto& x_dims = param_.x->dims();
  const auto& x_lod = param_.x->lod();
  CHECK_OR_FALSE(!x_lod.empty());
  CHECK_OR_FALSE(x_dims[0] == x_lod[0].back());
  param_.out->set_lod(x_lod);

  return true;
}

bool SearchGrnnOpLite::AttachImpl(const cpp::OpDesc& op_desc,
                                  lite::Scope* scope) {
  auto x = op_desc.Input("X").front();
  auto wi = op_desc.Input("Wi").front();
  auto wh = op_desc.Input("Wh").front();
  param_.x = scope->FindVar(x)->GetMutable<lite::Tensor>();
  param_.wi = scope->FindVar(wi)->GetMutable<lite::Tensor>();
  param_.wh = scope->FindVar(wh)->GetMutable<lite::Tensor>();

  param_.num_input = op_desc.GetAttr<int>("num_input");
  param_.num_hidden = op_desc.GetAttr<int>("num_hidden");

  auto out = op_desc.Output("Out").front();
  auto tmp_buffer = op_desc.Output("tmp_buffer").front();
  auto idx_sorted_by_width = op_desc.Output("idx_sorted_by_width").front();
  auto layout_input = op_desc.Output("layout_input").front();
  param_.out = scope->FindVar(out)->GetMutable<lite::Tensor>();
  param_.tmp_buffer = scope->FindVar(tmp_buffer)->GetMutable<lite::Tensor>();
  param_.idx_sorted_by_width =
      scope->FindVar(idx_sorted_by_width)->GetMutable<lite::Tensor>();
  param_.layout_input =
      scope->FindVar(layout_input)->GetMutable<lite::Tensor>();

#ifdef LITE_WITH_XPU
  if (op_desc.HasAttr("__xpu__float_to_fix")) {
    param_.__xpu__float_to_fix = op_desc.GetAttr<bool>("__xpu__float_to_fix");
  }
  if (op_desc.HasAttr("__xpu__wi_max")) {
    param_.__xpu__wi_max = op_desc.GetAttr<std::vector<float>>("__xpu__wi_max");
  }
  if (op_desc.HasAttr("__xpu__wh_max")) {
    param_.__xpu__wh_max = op_desc.GetAttr<std::vector<float>>("__xpu__wh_max");
  }
#endif

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(search_grnn, paddle::lite::operators::SearchGrnnOpLite);
