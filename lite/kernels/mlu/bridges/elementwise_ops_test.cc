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
#include "lite/core/op_registry.h"
#include "lite/core/subgraph_bridge_registry.h"
#include "lite/kernels/mlu/bridges/test_helper.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace mlu {

template <typename dtype>
void elementwise_add_ref(const std::shared_ptr<operators::ElementwiseOp> op) {
  Scope* scope = op->scope();
  const OpInfo* op_info = op->op_info();
  auto x = scope->FindTensor("x");
  auto y = scope->FindTensor("y");
  auto out = scope->FindMutableTensor("out_ref");
  out->Resize(x->dims());

  auto x_data = x->data<dtype>();
  auto y_data = y->data<dtype>();
  auto out_data = out->mutable_data<dtype>();

  auto x_dims = x->dims();
  auto y_dims = y->dims();
  int axis = op_info->GetAttr<int>("axis");

  if (axis < 0) {
    axis += x_dims.size();
  }
  int batch = 1;
  int channels = y->numel();
  int num = x->numel() / channels / batch;
  // do elementwise add/sub/max...
  std::string op_type = op_info->Type();
  if (op_type == "elementwise_add") {
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
  } else if (op_type == "elementwise_sub") {
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
  } else if (op_type == "elementwise_mul") {
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
  } else if (op_type == "elementwise_div") {
    for (int i = 0; i < batch; ++i) {
      for (int j = 0; j < channels; ++j) {
        int offset = (i * channels + j) * num;
        const dtype* din_ptr = x_data + offset;
        const dtype diny_data = y_data[j];
        dtype* dout_ptr = out_data + offset;
        for (int k = 0; k < num; ++k) {
          *dout_ptr = *din_ptr / diny_data;
          dout_ptr++;
          din_ptr++;
        }
      }
    }
  } else if (op_type == "elementwise_max") {
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
    LOG(FATAL) << "unsupported Elementwise type: " << op_type;
  }
}

void test_elementwise_add(const std::vector<int64_t>& x_shape,
                          const std::vector<int64_t>& y_shape,
                          int axis,
                          std::string elt_type) {
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
  x->Resize(x_shape);
  y->Resize(y_shape);

  // initialize input&output data
  FillTensor<float>(x, 1, 3);
  FillTensor<float>(y, 1, 3);

  // initialize op desc
  cpp::OpDesc opdesc;
  opdesc.SetType("elementwise_" + elt_type);
  opdesc.SetInput("X", {x_var_name});
  opdesc.SetInput("Y", {y_var_name});
  opdesc.SetOutput("Out", {out_var_name});
  opdesc.SetAttr("axis", axis);

  // create and convert op to MLU model, then run it on MLU
  auto op = CreateOp<operators::ElementwiseOp>(opdesc, &scope);

  // execute reference implementation and save to output tensor
  elementwise_add_ref<float>(op);
  out_ref->CopyDataFrom(*out);

  LaunchOp(op, {x_var_name, y_var_name}, {out_var_name});
  // compare results
  auto* out_data = out->mutable_data<float>();
  auto* out_ref_data = out_ref->mutable_data<float>();
  for (int i = 0; i < out->dims().production(); i++) {
    EXPECT_NEAR(out_data[i], out_ref_data[i], 1e-2);
  }
}

TEST(MLUBridges, elementwise_add) {
  for (auto elt_type : {"add", "sub", "mul", "div"}) {
    // test_elementwise_add({1, 2, 3, 4}, {2}, 1, elt_type);
    // test_elementwise_add({1, 2, 3, 4}, {1, 2, 1, 1}, 1, elt_type);
    test_elementwise_add({1, 2, 3, 4}, {1, 2, 3, 4}, 3, elt_type);
  }
}

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

USE_SUBGRAPH_BRIDGE(elementwise_add, kMLU)
USE_SUBGRAPH_BRIDGE(elementwise_sub, kMLU)
USE_SUBGRAPH_BRIDGE(elementwise_mul, kMLU)
USE_SUBGRAPH_BRIDGE(elementwise_div, kMLU)
