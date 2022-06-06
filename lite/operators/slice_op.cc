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
#include "lite/operators/slice_op.h"

#include <algorithm>

#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool SliceOp::CheckShape() const {
  CHECK(!(param_.X == nullptr && param_.XTensorList == nullptr));
  CHECK(!(param_.Out == nullptr && param_.OutTensorList == nullptr));
  if (param_.X) {
    CHECK_LT(param_.X->dims().size(), 7u)
        << "The rank of input X should be less than 7";
  }
  return true;
}

bool SliceOp::InferShapeImpl() const {
  // TODO(Superjomn) Enable data sharing.
  if (param_.XTensorList) {
    return true;
  } else if (param_.X) {
    auto in_dims = param_.X->dims();
    auto out_dims = in_dims;
    CHECK_EQ(param_.starts.size(), param_.ends.size())
        << "for slice op starts and ends must be equal";
    int dim_value, start, end;
    auto axes = param_.axes;
    auto starts = param_.starts;
    auto ends = param_.ends;
    auto decrease_axis = param_.decrease_axis;
    for (size_t i = 0; i < axes.size(); ++i) {
      CHECK_LT(param_.axes[i], in_dims.size()) << "The index of dimension in "
                                                  "axes must be less than the "
                                                  "size of input shape.";
      if (param_.infer_flags.size() > i && param_.infer_flags[i] == -1) {
        out_dims[axes[i]] = -1;
      } else {
        // infer out_dim shape
        dim_value = out_dims[axes[i]];
        if (dim_value > 0) {
          start = starts[i] < 0 ? (starts[i] + dim_value) : starts[i];
          end = ends[i] < 0 ? (ends[i] + dim_value) : ends[i];
          start = (std::max)(start, 0);
          end = (std::max)(end, 0);
          end = (std::min)(end, dim_value);
          out_dims[axes[i]] = end - start;
        }
      }
    }
    // generate new shape
    if (decrease_axis.size() > 0) {
      std::vector<int64_t> new_out_shape;
      for (size_t i = 0; i < decrease_axis.size(); ++i) {
        if (param_.infer_flags[i] != -1) {
          CHECK_EQ(out_dims[decrease_axis[i]], 1) << "decrease dim should be 1";
        }
        out_dims[decrease_axis[i]] = 0;
      }
      for (size_t i = 0; i < out_dims.size(); ++i) {
        if (out_dims[i] != 0) {
          new_out_shape.push_back(out_dims[i]);
        }
      }
      if (new_out_shape.size() == 0) {
        new_out_shape.push_back(1);
      }
      DDim new_dims;
      new_dims.ConstructFrom(new_out_shape);
      out_dims = new_dims;
    }
    param_.Out->Resize(out_dims);
    if (axes[0] != 0) {
      param_.Out->set_lod(param_.X->lod());
    }
  } else {
    LOG(FATAL) << "x or x_array must be set.";
  }
  return true;
}

bool SliceOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  auto input_var = scope->FindVar(opdesc.Input("Input").front());
  auto output_var = scope->FindVar(opdesc.Output("Out").front());
  bool input_is_array = input_var->IsType<std::vector<lite::Tensor>>();
  bool out_is_array = output_var->IsType<std::vector<lite::Tensor>>();
  if (input_is_array) {
    param_.XTensorList = input_var->GetMutable<std::vector<lite::Tensor>>();
    CHECK(param_.XTensorList);
  } else {
    param_.X = scope->FindVar(opdesc.Input("Input").front())
                   ->GetMutable<lite::Tensor>();
    CHECK(param_.X);
  }
  if (out_is_array) {
    param_.OutTensorList = output_var->GetMutable<std::vector<lite::Tensor>>();
    CHECK(param_.OutTensorList);
  } else {
    param_.Out = scope->FindVar(opdesc.Output("Out").front())
                     ->GetMutable<lite::Tensor>();
    CHECK(param_.Out);
  }
  param_.axes = opdesc.GetAttr<std::vector<int>>("axes");

  if (opdesc.HasAttr("infer_flags")) {
    param_.infer_flags = opdesc.GetAttr<std::vector<int>>("infer_flags");
  } else {
    // Initialize infer_flags with 1.
    // To be compatible with other op tests in which infer_flags is not set.
    param_.infer_flags = std::vector<int>(param_.axes.size(), 1);
  }
  if (opdesc.HasAttr("decrease_axis")) {
    param_.decrease_axis = opdesc.GetAttr<std::vector<int>>("decrease_axis");
  }

  // The priority: StartsTensor > StartsTensorList > attr(starts).
  // The priority: EndsTensor > EndsTensorList > attr(ends).
  size_t starts_size, ends_size;
  if (opdesc.HasAttr("starts")) {
    param_.starts = opdesc.GetAttr<std::vector<int>>("starts");
  }
  if (opdesc.HasAttr("ends")) {
    param_.ends = opdesc.GetAttr<std::vector<int>>("ends");
  }
  starts_size = param_.starts.size();
  ends_size = param_.ends.size();

  param_.StartsTensorList.clear();
  if (opdesc.HasInput("StartsTensorList") &&
      !opdesc.Input("StartsTensorList").empty()) {
    param_.StartsTensorList.clear();
    auto StartsTensorList = opdesc.Input("StartsTensorList");
    if (!StartsTensorList.empty() &&
        scope->FindVar(StartsTensorList[0])
            ->IsType<std::vector<lite::Tensor>>()) {
      auto tmp_tensor_list = scope->FindVar(StartsTensorList[0])
                                 ->GetMutable<std::vector<lite::Tensor>>();
      for (auto tensor : *tmp_tensor_list) {
        param_.StartsTensorList.push_back(&tensor);
      }
    } else {
      for (auto var : StartsTensorList) {
        param_.StartsTensorList.push_back(
            scope->FindVar(var)->GetMutable<lite::Tensor>());
      }
    }
    CHECK_GT(param_.StartsTensorList.size(), 0u)
        << "StartsTensorList size can't be zero";
    starts_size = param_.StartsTensorList.size();
  }
  param_.EndsTensorList.clear();
  if (opdesc.HasInput("EndsTensorList") &&
      !opdesc.Input("EndsTensorList").empty()) {
    param_.EndsTensorList.clear();
    auto EndsTensorList = opdesc.Input("EndsTensorList");
    if (!EndsTensorList.empty() &&
        scope->FindVar(EndsTensorList[0])
            ->IsType<std::vector<lite::Tensor>>()) {
      auto tmp_tensor_list = scope->FindVar(EndsTensorList[0])
                                 ->GetMutable<std::vector<lite::Tensor>>();
      for (auto tensor : *tmp_tensor_list) {
        param_.EndsTensorList.push_back(&tensor);
      }
    } else {
      for (auto var : EndsTensorList) {
        param_.EndsTensorList.push_back(
            scope->FindVar(var)->GetMutable<lite::Tensor>());
      }
    }
    CHECK_GT(param_.EndsTensorList.size(), 0u)
        << "EndsTensorList size can't be zero";
    ends_size = param_.EndsTensorList.size();
  }

  if (opdesc.HasInput("StartsTensor") &&
      !opdesc.Input("StartsTensor").empty()) {
    param_.StartsTensor = scope->FindVar(opdesc.Input("StartsTensor").front())
                              ->GetMutable<lite::Tensor>();
  } else {
    CHECK_EQ(starts_size, param_.axes.size())
        << "The size of starts must be equal to the size of axes.";
  }
  if (opdesc.HasInput("EndsTensor") && !opdesc.Input("EndsTensor").empty()) {
    param_.EndsTensor = scope->FindVar(opdesc.Input("EndsTensor").front())
                            ->GetMutable<lite::Tensor>();
  } else {
    CHECK_EQ(ends_size, param_.axes.size())
        << "The size of ends must be equal to the size of axes.";
  }
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(slice, paddle::lite::operators::SliceOp);
