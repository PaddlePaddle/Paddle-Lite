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

#include "lite/operators/squeeze_op.h"
#include "lite/core/op_registry.h"
namespace paddle {
namespace lite {
namespace operators {

static DDim GetOutputShape(const std::vector<int> &squeeze_dims,
                           const DDim &in_dims,
                           bool is_runtime) {
  size_t num_squeeze_dims = squeeze_dims.size();
  bool should_squeeze[9] = {false};

  // Determines number of dimensions of output tensor after squeeze.
  // Mark and count the dimensions need to be squeezed
  if (num_squeeze_dims == 0) {
    for (size_t idx = 0; idx < in_dims.size(); ++idx) {
      if (in_dims[idx] == 1) {
        should_squeeze[idx] = true;
      }
    }
  } else {
    for (size_t idx = 0; idx < num_squeeze_dims; ++idx) {
      int current = squeeze_dims[idx] < 0 ? squeeze_dims[idx] + in_dims.size()
                                          : squeeze_dims[idx];
      // Check current index, the upper limit has been checked.
      CHECK_GE(current, 0)
          << "Invalid axis, the negative axis is out of range.";

      if (!should_squeeze[current]) {
        if (is_runtime) {
          if (in_dims[current] == 1) {
            should_squeeze[current] = true;
          }
        } else {
          // At compile time, dim of -1 or 1 is allowed to squeeze
          if (in_dims[current] == 1 || in_dims[current] == -1) {
            should_squeeze[current] = true;
          }
        }
      }
    }
  }

  // Make output dimensions
  std::vector<int64_t> output_shape;
  for (int i = 0; i < in_dims.size(); ++i) {
    if (!should_squeeze[i]) {
      output_shape.push_back(in_dims[i]);
    }
  }
  return DDim(output_shape);
}

bool SqueezeOp::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Out);
  for (int a : param_.axes) {
    CHECK_LT(a, static_cast<int>(param_.X->dims().size()))
        << "The squeeze axis should be less than input tensor's rank.";
  }
  return true;
}

bool SqueezeOp::InferShapeImpl() const {
  std::vector<int> squeeze_dims = param_.axes;
  DDim in_dims = param_.X->dims();
  DDim out_dim = GetOutputShape(squeeze_dims, in_dims, true);
  param_.Out->Resize(out_dim);
  return true;
}

bool SqueezeOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  param_.X = scope->FindTensor(opdesc.Input("X").front());
  param_.Out = scope->FindMutableTensor(opdesc.Output("Out").front());

  if (opdesc.HasAttr("axes")) {
    param_.axes = opdesc.GetAttr<std::vector<int>>("axes");
  }
  CHECK(param_.X) << "Input(X) of SqueezeOp should not be null.";
  CHECK(param_.Out) << "Output(Out) of SqueezeOp should not be null.";
  if (opdesc.HasAttr("inplace")) {
    param_.inplace = opdesc.GetAttr<bool>("inplace");
  }
  input_tensor_ptrs_cache_.push_back(param_.X);
  output_tensor_ptrs_cache_.push_back(param_.Out);
  return true;
}

bool Squeeze2Op::CheckShape() const {
  SqueezeOp::CheckShape();
  return true;
}

bool Squeeze2Op::InferShapeImpl() const {
  SqueezeOp::InferShapeImpl();
  auto x_dims = param_.X->dims();
  std::vector<DDim::value_type> xshape_dims(x_dims.size() + 1, 0);
  for (size_t i = 0; i < x_dims.size(); i++) {
    xshape_dims[i + 1] = x_dims[i];
  }
  if (param_.XShape) param_.XShape->Resize(DDim(xshape_dims));
  return true;
}

bool Squeeze2Op::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  SqueezeOp::AttachImpl(opdesc, scope);
  if (opdesc.HasOutput("XShape"))
    param_.XShape = scope->FindMutableTensor(opdesc.Output("XShape").front());
  else
    LOG(INFO) << "PaddleLiteV2.12 remove XShape OutputTensor for SqueezeOp.";
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(squeeze, paddle::lite::operators::SqueezeOp);
REGISTER_LITE_OP(squeeze2, paddle::lite::operators::Squeeze2Op);
