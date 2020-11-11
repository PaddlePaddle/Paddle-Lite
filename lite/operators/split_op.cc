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
  CHECK_GT_OR_FALSE(param_.output.size(), 1UL);
  auto x_dims = param_.x->dims();
  auto x_rank = x_dims.size();
  CHECK_OR_FALSE(param_.axis >= -static_cast<int>(x_rank) &&
                 param_.axis < static_cast<int>(x_rank));
  return true;
}

bool SplitOp::InferShapeImpl() const {
  const auto &outs = param_.output;
  auto in_dims = param_.x->dims();
  int num = param_.num;
  const auto &sections = param_.sections;

  int axis = 0;
  if (param_.axis_tensor != nullptr) {
    axis = param_.axis_tensor->data<int>()[0];
  } else {
    axis = param_.axis;
  }
  if (axis < 0) {
    axis += in_dims.size();
  }

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
  }

  return true;
}

bool SplitOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  AttachParam(&param_);
  param_.axis = opdesc.GetAttr<int>("axis");
  param_.num = opdesc.GetAttr<int>("num");
  param_.sections = opdesc.GetAttr<std::vector<int>>("sections");
  auto input = opdesc.Input("X").front();
  auto outs = opdesc.Output("Out");
  param_.x = scope->FindVar(input)->GetMutable<lite::Tensor>();
  param_.output.clear();
  for (auto var : outs) {
    param_.output.push_back(scope->FindVar(var)->GetMutable<lite::Tensor>());
  }
  std::vector<std::string> input_arg_names = opdesc.InputArgumentNames();
  if (std::find(input_arg_names.begin(), input_arg_names.end(), "AxisTensor") !=
      input_arg_names.end()) {
    auto args = opdesc.Input("AxisTensor");
    if (!args.empty()) {
      auto *var = scope->FindVar(args.front());
      param_.axis_tensor = var->GetMutable<lite::Tensor>();
    }
  }
  if (std::find(input_arg_names.begin(),
                input_arg_names.end(),
                "SectionsTensorList") != input_arg_names.end()) {
    auto args = opdesc.Input("SectionsTensorList");
    if (!args.empty()) {
      auto *var = scope->FindVar(args.front());
      param_.sections_tensor_list =
          *(var->GetMutable<std::vector<lite::Tensor *>>());
    }
  }
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(split, paddle::lite::operators::SplitOp);
