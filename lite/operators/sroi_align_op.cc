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

#include "lite/operators/sroi_align_op.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool SroiAlignOpLite::CheckShape() const {
  CHECK_GT_OR_FALSE(param_.X.size(), 2UL);
  // CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Out);
  return true;
}

bool SroiAlignOpLite::InferShape() const {
  std::vector<lite::DDim> input_dims;
  for (auto p : param_.X) {
    input_dims.push_back(p->dims());
  }
  lite::DDim output_dims;
  int num_index = input_dims[0][0];
  int channel_index = input_dims[0][1];
  int height_index = input_dims[0][2];
  int width_index = input_dims[0][3];
  output_dims[num_index] = input_dims[1][0];
  output_dims[channel_index] = input_dims[0][1];
  output_dims[height_index] = param_.pooled_h;
  output_dims[width_index] = param_.pooled_w;

  param_.Out->Resize(lite::DDim(output_dims));
  return true;
}

// TODO(Superjomn) replace framework::OpDesc with a lite one.
bool SroiAlignOpLite::AttachImpl(const cpp::OpDesc &op_desc,
                                 lite::Scope *scope) {
  // 1、读入数据
  auto inputs = op_desc.Input("X");
  auto out = op_desc.Output("Out").front();

  for (auto var : inputs) {
    param_.X.push_back(scope->FindVar(var)->GetMutable<lite::Tensor>());
  }
  CHECK(scope->FindVar(out));
  param_.Out = scope->FindVar(out)->GetMutable<lite::Tensor>();
  // 2、其他变量
  param_.pooled_h = op_desc.GetAttr<int>("pooled_h");
  param_.pooled_w = op_desc.GetAttr<int>("pooled_w");
  param_.spatial_scale = op_desc.GetAttr<float>("spatial_scale");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(sroi_align, paddle::lite::operators::SroiAlignOpLite);
