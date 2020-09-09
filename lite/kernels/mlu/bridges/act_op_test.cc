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

#include <gtest/gtest.h>

#include <random>

#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"
#include "lite/core/subgraph_bridge_registry.h"
#include "lite/kernels/mlu/bridges/test_helper.h"
#include "lite/operators/activation_ops.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace mlu {

template void FillTensor<float, int>(Tensor* x,
                                     float lower = -2,
                                     float upper = -2);

void act_ref(const std::shared_ptr<operators::ActivationOp> op) {
  Scope* scope = op->scope();
  const OpInfo* op_info = op->op_info();
  auto op_type = op_info->Type();
  auto x = scope->FindTensor("x");
  auto out = scope->FindMutableTensor("out");
  auto out_ref = scope->FindMutableTensor("out_ref");
  out->Resize(x->dims());
  out_ref->Resize(x->dims());
  auto x_data = x->data<float>();
  auto out_data = out->mutable_data<float>();
  CHECK_EQ(x->numel(), out->numel());

  // "sigmoid","relu","tanh","relu_clipped","leaky_relu","softsign","hard_sigmoid"
  if (op_type == "sigmoid") {
    for (int i = 0; i < out->numel(); i++) {
      out_data[i] = 1.f / (1.f + std::exp(-x_data[i]));
    }
  } else if (op_type == "relu") {
    for (int i = 0; i < out->numel(); i++) {
      out_data[i] = std::max(0.f, x_data[i]);
    }
  } else if (op_type == "tanh") {
    for (int i = 0; i < out->numel(); i++) {
      out_data[i] = (std::exp(x_data[i]) - std::exp(-x_data[i])) /
                    (std::exp(x_data[i]) + std::exp(-x_data[i]));
    }
  } else if (op_type == "relu_clipped") {
    auto relu_clipped_coef = op_info->GetAttr<float>("Relu_clipped_coef");
    for (int i = 0; i < out->numel(); i++) {
      out_data[i] = std::min(std::max(0.f, x_data[i]), relu_clipped_coef);
    }
  } else if (op_type == "relu6") {
    for (int i = 0; i < out->numel(); i++) {
      out_data[i] = std::min(std::max(0.f, x_data[i]), 6.f);
    }
  } else if (op_type == "leaky_relu") {
    auto alpha = op_info->GetAttr<float>("alpha");
    for (int i = 0; i < out->numel(); i++) {
      out_data[i] = std::max(x_data[i], x_data[i] * alpha);
    }
  } else if (op_type == "softsign") {
    for (int i = 0; i < out->numel(); i++) {
      out_data[i] = x_data[i] / (1 + std::abs(x_data[i]));
    }
  } else if (op_type == "hard_sigmoid") {
    auto slope = op_info->GetAttr<float>("slope");
    auto offset = op_info->GetAttr<float>("offset");
    for (int i = 0; i < out->numel(); i++) {
      out_data[i] = std::min(1.f, slope * x_data[i] + offset);
      out_data[i] = std::max(0.f, out_data[i]);
    }
  } else {
    LOG(FATAL) << "unsupported activation type: " << op_type;
  }
}

void test_act(std::vector<int64_t> x_shape, std::string op_type) {
  // prepare input&output variables
  Scope scope;
  std::string x_var_name("x");
  std::string out_var_name("out");
  std::string out_ref_var_name("out_ref");
  auto* x = scope.NewTensor(x_var_name);
  auto* out = scope.NewTensor(out_var_name);
  auto* out_ref = scope.NewTensor(out_ref_var_name);
  x->Resize(x_shape);

  // initialize input&output data
  FillTensor<float>(x, 2, 8);

  // initialize op desc
  cpp::OpDesc opdesc;
  opdesc.SetType(op_type);
  opdesc.SetInput("X", {x_var_name});
  opdesc.SetOutput("Out", {out_var_name});
  if (op_type == "relu_clipped") {
    opdesc.SetAttr("Relu_clipped_coef", 3.f);
  } else if (op_type == "relu6") {
    opdesc.SetAttr("Relu_clipped_coef", 6.f);
  } else if (op_type == "leaky_relu") {
    opdesc.SetAttr("alpha", 0.02f);
  } else if (op_type == "hard_sigmoid") {
    opdesc.SetAttr("slope", 0.2f);
    opdesc.SetAttr("offset", 0.5f);
  }

  // create and convert op to MLU model, then run it on MLU
  auto op = CreateOp<operators::ActivationOp>(opdesc, &scope);
  // execute reference implementation and save to output tensor
  act_ref(op);
  out_ref->CopyDataFrom(*out);

  LaunchOp(op, {x_var_name}, {out_var_name});

  // compare results
  auto* out_data = out->mutable_data<float>();
  auto* out_ref_data = out_ref->mutable_data<float>();
  for (int i = 0; i < out->dims().production(); i++) {
    EXPECT_NEAR(out_data[i], out_ref_data[i], 1e-2);
  }
}

TEST(MLUBridges, activation) {
  std::vector<std::vector<int64_t>> shapes{{1}, {2, 3}, {1, 2, 3, 4}};
  std::vector<std::string> types{
      "sigmoid", "relu", "relu6", "tanh", "leaky_relu"};
  for (auto x_shape : shapes) {
    for (auto op_type : types) {
      test_act(x_shape, op_type);
    }
  }
}

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

USE_SUBGRAPH_BRIDGE(sigmoid, kMLU)
USE_SUBGRAPH_BRIDGE(relu, kMLU)
USE_SUBGRAPH_BRIDGE(relu6, kMLU)
USE_SUBGRAPH_BRIDGE(tanh, kMLU)
USE_SUBGRAPH_BRIDGE(leaky_relu, kMLU)
