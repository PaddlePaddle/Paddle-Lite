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

#include "lite/operators/unsqueeze_op.h"
#include "lite/core/op_registry.h"
namespace paddle {
namespace lite {
namespace operators {

static DDim GetOutputShape(const std::vector<int> &unsqz_dims,
                           const DDim &in_dims) {
  int output_size = in_dims.size() + static_cast<int>(unsqz_dims.size());
  int cur_output_size = in_dims.size();
  std::vector<int64_t> output_shape(output_size, 0);

  // Validate Check: rank range.
  CHECK_LE(output_size, 6) << "The output tensor's rank should be less than 6.";

  for (int axis : unsqz_dims) {
    int cur = axis < 0 ? axis + cur_output_size + 1 : axis;
    // Validate Check: the axis bound
    CHECK((cur >= 0) && (cur <= cur_output_size))
        << "The unsqueeze dims must be within range of current rank.";
    // Move old axis, and insert new axis
    for (int i = cur_output_size; i >= cur; --i) {
      if (output_shape[i] == 1) {
        // Move axis
        output_shape[i + 1] = 1;
        output_shape[i] = 0;
      }
    }

    output_shape[cur] = 1;
    // Add the output size.
    cur_output_size++;
  }

  // Make output shape
  for (int in_idx = 0, out_idx = 0; out_idx < output_size; ++out_idx) {
    if (output_shape[out_idx] == 0) {
      output_shape[out_idx] = in_dims[in_idx++];
    }
  }

  return DDim(output_shape);
}

bool UnsqueezeOp::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Out);
  return true;
}

bool UnsqueezeOp::InferShapeImpl() const {
  std::vector<int> final_axes;
  auto axes = param_.axes;
  auto *axes_tensor = param_.axes_tensor;
  auto axes_tensor_vct = param_.axes_tensor_vct;

  if (!axes.empty()) {
    final_axes = axes;
  } else if (axes_tensor != nullptr) {
    auto *axes_tensor_data = axes_tensor->data<int>();
    final_axes = std::vector<int>(axes_tensor_data,
                                  axes_tensor_data + axes_tensor->numel());
  } else if (!axes_tensor_vct.empty()) {
    for (size_t i = 0; i < axes_tensor_vct.size(); i++) {
      final_axes.push_back(axes_tensor_vct[i]->data<int>()[0]);
    }
  } else {
    LOG(FATAL) << "Input axis error";
  }

  DDim in_dims = param_.X->dims();
  DDim out_dim = GetOutputShape(final_axes, in_dims);
  param_.Out->Resize(out_dim);
  return true;
}

bool UnsqueezeOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  param_.X = scope->FindTensor(opdesc.Input("X").front());
  param_.Out = scope->FindMutableTensor(opdesc.Output("Out").front());

  if (opdesc.HasAttr("axes")) {
    param_.axes = opdesc.GetAttr<std::vector<int>>("axes");
  }

  if (opdesc.HasInput("AxesTensor") && opdesc.Input("AxesTensor").size() > 0) {
    auto var = scope->FindVar(opdesc.Input("AxesTensor").front());
    if (var != nullptr) {
      param_.axes_tensor = var->GetMutable<lite::Tensor>();
      VLOG(5) << "load AxesTensor";
    }
  }

  if (opdesc.HasInput("AxesTensorList") &&
      opdesc.Input("AxesTensorList").size() > 0) {
    auto args = opdesc.Input("AxesTensorList");
    for (auto arg : args) {
      auto *var = scope->FindVar(arg);
      if (var != nullptr) {
        param_.axes_tensor_vct.push_back(var->GetMutable<lite::Tensor>());
      }
    }
  }
  CHECK(param_.X) << "Input(X) of UnsqueezeOp should not be null.";
  CHECK(param_.Out) << "Output(Out) of UnsqueezeOp should not be null.";
  if (opdesc.HasAttr("inplace")) {
    param_.inplace = opdesc.GetAttr<bool>("inplace");
  }
  input_tensor_ptrs_cache_.push_back(param_.X);
  output_tensor_ptrs_cache_.push_back(param_.Out);
  return true;
}

bool Unsqueeze2Op::CheckShape() const {
  UnsqueezeOp::CheckShape();
  // CHECK_OR_FALSE(param_.XShape);
  return true;
}

bool Unsqueeze2Op::InferShapeImpl() const {
  UnsqueezeOp::InferShapeImpl();
  auto x_dims = param_.X->dims();
  std::vector<DDim::value_type> xshape_dims(x_dims.size() + 1, 0);
  for (size_t i = 0; i < x_dims.size(); i++) {
    xshape_dims[i + 1] = x_dims[i];
  }
  if (param_.XShape) param_.XShape->Resize(DDim(xshape_dims));
  return true;
}

bool Unsqueeze2Op::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  UnsqueezeOp::AttachImpl(opdesc, scope);
  if (opdesc.HasOutput("XShape")) {
    param_.XShape = scope->FindMutableTensor(opdesc.Output("XShape").front());
    // CHECK(param_.XShape) << "Output(XShape) of Unsqueeze2Op should not be
    // null.";
  }
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(unsqueeze, paddle::lite::operators::UnsqueezeOp);
REGISTER_LITE_OP(unsqueeze2, paddle::lite::operators::Unsqueeze2Op);
