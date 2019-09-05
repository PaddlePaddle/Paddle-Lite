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

#include "lite/operators/elementwise_ops.h"
#include <gtest/gtest.h>
#include <random>
#include "lite/backends/npu/bridge/registry.h"
#include "lite/backends/npu/bridge/test_helper.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace npu {
namespace bridge {

template <typename dtype>
void elementwise_add_ref(const std::shared_ptr<operators::ElementwiseOp> op) {
  Scope* scope = op->scope();
  const OpInfo* op_info = op->op_info();
  auto x = scope->FindVar(op_info->Input("X").front())->GetMutable<Tensor>();
  auto y = scope->FindVar(op_info->Input("Y").front())->GetMutable<Tensor>();
  auto out =
      scope->FindVar(op_info->Output("Out").front())->GetMutable<Tensor>();

  auto x_data = x->data<dtype>();
  auto y_data = y->data<dtype>();
  dtype* out_data = out->mutable_data<dtype>();

  auto x_dims = x->dims();
  auto y_dims = y->dims();
  int axis = op_info->GetAttr<int>("axis");

  if (axis < 0) {
    axis = x_dims.size() - y_dims.size();
  }
  int batch = 1;
  int channels = 1;
  int num = 1;
  for (int i = 0; i < axis; ++i) {
    batch *= x_dims[i];
  }
  for (int i = 0; i < y_dims.size(); ++i) {
    channels *= y_dims[i];
  }
  for (int i = y_dims.size() + axis; i < x_dims.size(); ++i) {
    num *= x_dims[i];
  }
  // do elementwise add/sub/max...
  std::string elt_type = "add";
  if (elt_type == "add") {
    for (int i = 0; i < batch; ++i) {
      for (int j = 0; j < channels; ++j) {
        int offset = (i * channels + j) * num;
        const dtype* din_ptr = x_data + offset;
        const dtype diny_data = y_data[j];
        dtype* dout_ptr = out_data + offset;
        for (int k = 0; k < num; ++k) {
          *dout_ptr = *din_ptr + diny_data;
          dout_ptr++;
          din_ptr++;
        }
      }
    }
  } else if (elt_type == "sub") {
    for (int i = 0; i < batch; ++i) {
      for (int j = 0; j < channels; ++j) {
        int offset = (i * channels + j) * num;
        const dtype* din_ptr = x_data + offset;
        const dtype diny_data = y_data[j];
        dtype* dout_ptr = out_data + offset;
        for (int k = 0; k < num; ++k) {
          *dout_ptr = *din_ptr - diny_data;
          dout_ptr++;
          din_ptr++;
        }
      }
    }
  } else if (elt_type == "mul") {
    for (int i = 0; i < batch; ++i) {
      for (int j = 0; j < channels; ++j) {
        int offset = (i * channels + j) * num;
        const dtype* din_ptr = x_data + offset;
        const dtype diny_data = y_data[j];
        dtype* dout_ptr = out_data + offset;
        for (int k = 0; k < num; ++k) {
          *dout_ptr = *din_ptr * diny_data;
          dout_ptr++;
          din_ptr++;
        }
      }
    }
  } else if (elt_type == "max") {
    for (int i = 0; i < batch; ++i) {
      for (int j = 0; j < channels; ++j) {
        int offset = (i * channels + j) * num;
        const dtype* din_ptr = x_data + offset;
        const dtype diny_data = y_data[j];
        dtype* dout_ptr = out_data + offset;
        for (int k = 0; k < num; ++k) {
          *dout_ptr = std::max(*din_ptr, diny_data);
          dout_ptr++;
          din_ptr++;
        }
      }
    }
  } else {
    LOG(FATAL) << "unsupported Elementwise type: " << elt_type;
  }
}

void test_elementwise_add(int bs, int ic, int ih, int iw, int axis) {
  // prepare input&output variables
  Scope scope;
  std::string x_var_name = "x";
  std::string y_var_name = "y";
  std::string out_var_name = "out";
  std::string out_ref_var_name = "out_ref";
  auto* x = scope.Var(x_var_name)->GetMutable<Tensor>();
  auto* y = scope.Var(y_var_name)->GetMutable<Tensor>();
  auto* out = scope.Var(out_var_name)->GetMutable<Tensor>();
  auto* out_ref = scope.Var(out_ref_var_name)->GetMutable<Tensor>();
  x->Resize({bs, ic, ih, iw});
  y->Resize({bs, ic, ih, iw});

  // initialize input&output data
  FillTensor<float>(x);
  FillTensor<float>(y);

  // initialize op desc
  cpp::OpDesc opdesc;
  opdesc.SetType("elementwise_add");
  opdesc.SetInput("X", {x_var_name});
  opdesc.SetInput("Y", {y_var_name});
  opdesc.SetOutput("Out", {out_var_name});
  opdesc.SetAttr("axis", axis);

  // create and convert op to NPU model, then run it on NPU
  auto op = CreateOp<operators::ElementwiseOp>(opdesc, &scope);
  LauchOp(op, {x_var_name}, {out_var_name});
  out_ref->CopyDataFrom(*out);

  // execute reference implementation and save to output tensor
  elementwise_add_ref<float>(op);

  // compare results
  auto* out_data = out->mutable_data<float>();
  auto* out_ref_data = out_ref->mutable_data<float>();
  for (int i = 0; i < out->dims().production(); i++) {
    EXPECT_NEAR(out_data[i], out_ref_data[i], 1e-1);
  }
}

TEST(NPUBridges, elementwise_add) {
  for (auto bs : {1, 4, 7}) {
    for (auto ic : {1, 4, 7}) {
      for (auto ih : {1, 4, 7}) {
        for (auto iw : {1, 4, 7}) {
          for (auto axis : {-1}) test_elementwise_add(bs, ic, ih, iw, axis);
        }
      }
    }
  }
}

}  // namespace bridge
}  // namespace npu
}  // namespace lite
}  // namespace paddle

USE_LITE_OP(elementwise_add);
USE_NPU_BRIDGE(elementwise_add);
