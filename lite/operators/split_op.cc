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

#include "lite/operators/split_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool SplitOp::CheckShape() const {
  CHECK_OR_FALSE(param_.x);
  auto x_dims = param_.x->dims();
  auto x_rank = x_dims.size();
  CHECK_GE(param_.axis, -static_cast<int>(x_rank));
  CHECK_LT(param_.axis, static_cast<int>(x_rank));
  return true;
}

bool SplitOp::InferShapeImpl() const {
  const auto &outs = param_.output;
  auto in_dims = param_.x->dims();
  int num = param_.num;
  auto &sections = param_.sections;

  int axis = 0;
  if (param_.axis_tensor != nullptr) {
    axis = param_.axis_tensor->data<int>()[0];
  } else {
    axis = param_.axis;
  }
  if (axis < 0) {
    axis += in_dims.size();
  }
  // update sections
  int infer_num = std::count(sections.begin(), sections.end(), -1);
  CHECK_LT(infer_num, 2);
  for (int i = 0; i < sections.size(); i++) {
    if (sections[i] == -1) {
      sections[i] =
          in_dims[axis] - std::accumulate(sections.begin(), sections.end(), 1);
    }
  }
  // update sections end

  const int outs_number = outs.size();
  std::vector<lite::DDim> outs_dims;
  outs_dims.reserve(outs_number);
  std::vector<lite::Tensor *> sections_tensor_list_ =
      param_.sections_tensor_list;
  if (sections.size() > 0 && sections_tensor_list_.size() > 0) {
    std::vector<int> vec_sections;
    for (size_t i = 0; i < sections_tensor_list_.size(); ++i) {
      auto dim = in_dims;
      dim[axis] = sections_tensor_list_[i]->data<int>()[0];
      outs_dims.push_back(dim);
    }
  } else if (num > 0) {
    int out_axis_dim = in_dims[axis] / num;
    for (int i = 0; i < outs_number; ++i) {
      auto dim = in_dims;
      dim[axis] = out_axis_dim;
      outs_dims.push_back(dim);
    }
  } else if (sections.size() > 0) {
    for (int i = 0; i < outs_number; ++i) {
      auto dim = in_dims;
      dim[axis] = sections[i];
      outs_dims.push_back(dim);
    }
  }

  for (size_t j = 0; j < outs_dims.size(); ++j) {
    outs[j]->Resize(outs_dims[j]);
    if (axis != 0) {
      outs[j]->set_lod(param_.x->lod());
    }
  }
  return true;
}

bool SplitOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  param_.axis = opdesc.GetAttr<int>("axis");
  param_.num = opdesc.GetAttr<int>("num");
  param_.sections = opdesc.GetAttr<std::vector<int>>("sections");

  param_.x = scope->FindTensor(opdesc.Input("X").front());
  if (opdesc.HasInput("AxisTensor") && !opdesc.Input("AxisTensor").empty()) {
    param_.axis_tensor = scope->FindTensor(opdesc.Input("AxisTensor").front());
  }
  param_.sections_tensor_list.clear();
  if (opdesc.HasInput("SectionsTensorList")) {
    auto names = opdesc.Input("SectionsTensorList");
    for (auto name : names) {
      param_.sections_tensor_list.push_back(scope->FindMutableTensor(name));
    }
  }

  param_.output.clear();
  auto outs_name = opdesc.Output("Out");
  for (auto name : outs_name) {
    param_.output.push_back(scope->FindMutableTensor(name));
    output_tensor_ptrs_cache_.push_back(scope->FindMutableTensor(name));
  }
  input_tensor_ptrs_cache_.push_back(param_.x);

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(split, paddle::lite::operators::SplitOp);
