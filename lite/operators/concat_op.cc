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

#include "lite/operators/concat_op.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool ConcatOpLite::CheckShape() const {
  CHECK_GE_OR_FALSE(param_.x.size(), 1UL);
  CHECK_OR_FALSE(param_.output);
  return true;
}

bool ConcatOpLite::SmartInferShape() {
  if (!last_input_shapes.empty()){
    const std::vector<Tensor *> &inputs = param_.x;
    bool same = true;
    for (int i = 0; i < inputs.size(); i++){
      if (last_input_shapes[i] == inputs[i]->dims() && 
          last_input_lods[i] == inputs[i]->lod()){
        continue;
      } else{
        same = false;
        break;
      }
    }
    if (same) {
      param_.output->Resize(last_output_shapes[0]);
      param_.output->set_lod(last_output_lods[0]);
      return true;
    }
  }

  this->InferShape();
  
  if (!last_input_shapes.empty()) {
    last_input_shapes.clear();
    last_input_lods.clear();
  }
  const std::vector<Tensor *> &inputs = param_.x;
  for (int i = 0; i < inputs.size(); i++){
    last_input_shapes.push_back(inputs[i]->dims());
    last_input_lods.push_back(inputs[i]->lod());
  }

  if (!last_output_shapes.empty()) {
    last_output_shapes.clear();
    last_output_lods.clear();
  }
  last_output_shapes.push_back(param_.output->dims());
  last_output_lods.push_back(param_.output->lod());

  return true;
}

bool ConcatOpLite::InferShape() const {
  const std::vector<Tensor *> &inputs = param_.x;
  const size_t n = inputs.size();
  CHECK_GT_OR_FALSE(n, 0);

  int axis = 0;
  if (param_.axis_tensor == nullptr) {
    axis = param_.axis;
  } else {
    auto *axis_tensor_val = param_.axis_tensor->data<int>();
    axis = axis_tensor_val[0];
  }
  if (axis < 0) {
    axis += inputs[0]->dims().size();
  }

  auto out_dims = inputs[0]->dims();
  size_t in_zero_dims_size = out_dims.size();
  for (size_t i = 1; i < n; i++) {
    const auto &input_dims_i = inputs[i]->dims();
    for (size_t j = 0; j < in_zero_dims_size; j++) {
      if (j == static_cast<size_t>(axis)) {
        out_dims[axis] += input_dims_i[j];
      } else {
        CHECK_EQ_OR_FALSE(out_dims[j], input_dims_i[j]);
      }
    }
  }
  if (out_dims[axis] < 0) {
    out_dims[axis] = -1;
  }
  // Set output dims
  param_.output->Resize(out_dims);
  param_.output->set_lod(param_.x[0]->lod());
  auto out_lod = param_.output->mutable_lod();
  *out_lod = param_.x[0]->lod();
  return true;
}

// TODO(Superjomn) replace framework::OpDesc with a lite one.
bool ConcatOpLite::AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) {
  auto inputs = op_desc.Input("X");
  auto out = op_desc.Output("Out").front();

  param_.x.clear();
  for (auto var : inputs) {
    param_.x.push_back(scope->FindVar(var)->GetMutable<lite::Tensor>());
  }
  CHECK(scope->FindVar(out));
  param_.output = scope->FindVar(out)->GetMutable<lite::Tensor>();
  param_.axis = op_desc.GetAttr<int>("axis");

  std::vector<std::string> input_arg_names = op_desc.InputArgumentNames();
  if (std::find(input_arg_names.begin(), input_arg_names.end(), "AxisTensor") !=
      input_arg_names.end()) {
    auto arguments = op_desc.Input("AxisTensor");
    if (arguments.size() > 0) {
      auto var = scope->FindVar(arguments.front());
      if (var != nullptr) {
        param_.axis_tensor = var->GetMutable<lite::Tensor>();
      }
    }
  }
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(concat, paddle::lite::operators::ConcatOpLite);
