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
  int cnt_squeezed_dims = 0;
  bool should_squeeze[9] = {false};

  // Determines number of dimensions of output tensor after squeeze.
  // Mark and count the dimensions need to be squeezed
  if (num_squeeze_dims == 0) {
    for (int idx = 0; idx < in_dims.size(); ++idx) {
      if (in_dims[idx] == 1) {
        should_squeeze[idx] = true;
        ++cnt_squeezed_dims;
      }
    }
  } else {
    for (size_t idx = 0; idx < num_squeeze_dims; ++idx) {
      int current = squeeze_dims[idx] < 0 ? squeeze_dims[idx] + in_dims.size()
                                          : squeeze_dims[idx];
      // Check current index, the upper limit has been checked.
      CHECK_GE(current, 0)
          << "Invalid axis, the negative axis is out of range.";

      if (is_runtime) {
        CHECK_EQ(in_dims[current], 1) << "Invalid axis index, the axis that "
                                         "will be squeezed should be equal "
                                         "to 1.";
      }

      if (!(should_squeeze[current])) {
        ++cnt_squeezed_dims;
      }
      should_squeeze[current] = true;
    }
  }

  // Make output dimensions
  std::vector<int64_t> output_shape(in_dims.size() - cnt_squeezed_dims, 0);
  for (int in_idx = 0, out_idx = 0; in_idx < in_dims.size(); ++in_idx) {
    if (!should_squeeze[in_idx]) {
      output_shape[out_idx++] = in_dims[in_idx];
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

bool SqueezeOp::InferShape() const {
  std::vector<int> squeeze_dims = param_.axes;
  DDim in_dims = param_.X->dims();
  DDim out_dim = GetOutputShape(squeeze_dims, in_dims, true);
  param_.Out->Resize(out_dim);
  return true;
}

bool SqueezeOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  auto x_var = scope->FindVar(opdesc.Input("X").front());
  auto output_var = scope->FindVar(opdesc.Output("Out").front());
  CHECK(x_var);
  CHECK(output_var);
  param_.X = const_cast<lite::Tensor *>(&(x_var->Get<lite::Tensor>()));
  param_.Out = output_var->GetMutable<lite::Tensor>();

  if (opdesc.HasAttr("axes")) {
    param_.axes = opdesc.GetAttr<std::vector<int>>("axes");
  }
  CHECK(param_.X) << "Input(X) of SqueezeOp should not be null.";
  CHECK(param_.Out) << "Output(Out) of SqueezeOp should not be null.";
  return true;
}

bool Squeeze2Op::CheckShape() const {
  SqueezeOp::CheckShape();
  CHECK_OR_FALSE(param_.XShape);
  return true;
}

bool Squeeze2Op::InferShape() const {
  SqueezeOp::InferShape();
  auto x_dims = param_.X->dims();
  std::vector<DDim::value_type> xshape_dims(x_dims.size() + 1, 1);
  for (size_t i = 0; i < x_dims.size(); i++) {
    xshape_dims[i + 1] = x_dims[i];
  }
  param_.XShape->Resize(DDim(xshape_dims));
  return true;
}

bool Squeeze2Op::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  SqueezeOp::AttachImpl(opdesc, scope);
  auto xshape_var = scope->FindVar(opdesc.Output("XShape").front());
  CHECK(xshape_var);
  param_.XShape = xshape_var->GetMutable<lite::Tensor>();
  CHECK(param_.XShape) << "Output(XShape) of ReshapeOp should not be null.";
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(squeeze, paddle::lite::operators::SqueezeOp);
REGISTER_LITE_OP(squeeze2, paddle::lite::operators::Squeeze2Op);
