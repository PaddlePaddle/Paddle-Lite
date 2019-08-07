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

#include "lite/operators/mul_op.h"
#include <gtest/gtest.h>
#include <random>
#include "ai_ddk_lib/include/graph/op/all_ops.h"
#include "lite/core/op_registry.h"
#include "lite/npu/bridge/registry.h"
#include "lite/npu/bridge/utils.h"
#include "lite/operators/graph_op.h"

namespace paddle {
namespace lite {
namespace npu {
namespace bridge {

void mul_ref(const std::shared_ptr<operators::MulOpLite> op) {
  Scope* scope = op->scope();
  const OpInfo* op_info = op->op_info();
  auto x = scope->FindVar(op_info->Input("X").front())->GetMutable<Tensor>();
  auto y = scope->FindVar(op_info->Input("Y").front())->GetMutable<Tensor>();
  auto out =
      scope->FindVar(op_info->Output("Out").front())->GetMutable<Tensor>();
  int32_t x_num_col_dims = op_info->GetAttr<int32_t>("x_num_col_dims");
  int32_t y_num_col_dims = op_info->GetAttr<int32_t>("y_num_col_dims");
  auto x_data = x->mutable_data<float>();
  auto y_data = y->mutable_data<float>();
  auto out_data = out->mutable_data<float>();
  auto x_mat_dims = x->dims().Flattern2D(x_num_col_dims);
  auto y_mat_dims = y->dims().Flattern2D(y_num_col_dims);
  CHECK_EQ(x_mat_dims[1], y_mat_dims[0]);
  const int M = x_mat_dims[0];
  const int K = x_mat_dims[1];
  const int N = y_mat_dims[1];
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      out_data[m * N + n] = 0;
      for (int k = 0; k < K; ++k) {
        out_data[m * N + n] += x_data[m * K + k] * y_data[k * N + n];
      }
    }
  }
}

void test_mul(
    int n, int c, int h, int w, int o, int x_num_col_dims, int y_num_col_dims) {
  const auto& bridges = lite::npu::bridge::Factory::Instance();
  const auto& supported_lists = bridges.AllFunctions();
  CHECK(bridges.HasType("mul"));

  // prepare input&output variables
  Scope scope;
  std::string x_var_name("x");
  std::string y_var_name("y");
  std::string out_var_name("out");
  std::string out_ref_var_name("out_ref");
  auto* x = scope.Var(x_var_name)->GetMutable<Tensor>();
  auto* y = scope.Var(y_var_name)->GetMutable<Tensor>();
  auto* out = scope.Var(out_var_name)->GetMutable<Tensor>();
  auto* out_ref = scope.Var(out_ref_var_name)->GetMutable<Tensor>();
  x->Resize({n, c, h, w});

  // get y shape
  auto x_mat_dims = x->dims().Flattern2D(x_num_col_dims);
  std::vector<int64_t> y_shape;
  for (int i = 0; i < y_num_col_dims - 1; i++) {
    y_shape.push_back(1);
  }
  y_shape.push_back(x_mat_dims[1]);
  y_shape.push_back(o);
  y->Resize(y_shape);

  // initialize input&output data
  std::default_random_engine rand_eng;
  std::uniform_real_distribution<float> rand_dist(-5.0f, 5.0f);
  for (int i = 0; i < x->dims().production(); i++) {
    float rand_value = half2float(float2half(rand_dist(rand_eng)));
    x->mutable_data<float>()[i] = rand_value;
  }
  for (int i = 0; i < y->dims().production(); i++) {
    float rand_value = half2float(float2half(rand_dist(rand_eng)));
    y->mutable_data<float>()[i] = rand_value;
  }

  // create mul op
  cpp::OpDesc mul_op_desc;
  mul_op_desc.SetType("mul");
  mul_op_desc.SetInput("X", {x_var_name});
  mul_op_desc.SetInput("Y", {y_var_name});
  mul_op_desc.SetOutput("Out", {out_var_name});
  mul_op_desc.SetAttr("x_num_col_dims", static_cast<int>(x_num_col_dims));
  mul_op_desc.SetAttr("y_num_col_dims", static_cast<int>(y_num_col_dims));

  auto mul_op = std::make_shared<operators::MulOpLite>(mul_op_desc.Type());
  mul_op->SetValidPlaces({Place{TARGET(kHost), PRECISION(kFloat)},
                          Place{TARGET(kARM), PRECISION(kFloat)}});
  CHECK(mul_op->Attach(mul_op_desc, &scope));
  CHECK(mul_op->CheckShape());
  CHECK(mul_op->InferShape());

  // convert mul op and build IR graph
  ge::TensorDesc x_desc(
      ge::Shape(x->dims().Vectorize()), ge::FORMAT_NCHW, ge::DT_FLOAT);
  ge::TensorDesc y_desc(
      ge::Shape(y->dims().Vectorize()), ge::FORMAT_NCHW, ge::DT_FLOAT);
  auto x_node = std::make_shared<ge::op::Data>(x_var_name);
  auto y_node = std::make_shared<ge::op::Data>(y_var_name);
  x_node->update_input_desc_x(x_desc);
  y_node->update_input_desc_x(y_desc);
  node_map_type inputs_map;
  inputs_map[x_var_name] = x_node;
  inputs_map[y_var_name] = y_node;
  auto outputs_map =
      supported_lists.at(mul_op->op_info()->Type())(mul_op, inputs_map);
  CHECK_GT(outputs_map.size(), 0);

  // compile IR graph to om model
  std::vector<ge::Operator> graph_inputs{*inputs_map[x_var_name],
                                         *inputs_map[y_var_name]};
  std::vector<ge::Operator> graph_outputs{*outputs_map[out_var_name]};
  std::string model_name(UniqueName("test_mul") + ".om");
  CHECK(npu::BuildNPUClient(graph_inputs, graph_outputs, model_name));

  // create graph op
  cpp::OpDesc graph_op_desc;
  graph_op_desc.SetType("graph_op");
  graph_op_desc.SetInput("Inputs", {x_var_name});
  graph_op_desc.SetOutput("Outputs", {y_var_name});
  graph_op_desc.SetAttr("model_name", model_name);

  auto graph_op =
      std::make_shared<operators::GraphOpLite>(graph_op_desc.Type());
  graph_op->SetValidPlaces({Place{TARGET(kNPU), PRECISION(kFloat)}});
  CHECK(graph_op->Attach(graph_op_desc, &scope));
  CHECK(graph_op->CheckShape());
  CHECK(graph_op->InferShape());

  // create graph op kernel
  auto graph_kernels =
      graph_op->CreateKernels({Place{TARGET(kNPU), PRECISION(kFloat)}});
  CHECK(!graph_kernels.empty());
  auto graph_kernel =
      std::move(graph_kernels.front());  // use the first kernel by default
  auto graph_ctx = ContextScheduler::Global().NewContext(TARGET(kNPU));
  graph_kernel->SetContext(std::move(graph_ctx));

  // perform graph op kernel and copy output tensor('out') to 'out_ref'
  graph_kernel->Launch();
  out_ref->CopyDataFrom(*out);

  // execute reference implementation and save to output tensor('out')
  mul_ref(mul_op);

  // compare results
  auto* out_data = out->mutable_data<float>();
  auto* out_ref_data = out_ref->mutable_data<float>();
  for (int i = 0; i < out->dims().production(); i++) {
    EXPECT_NEAR(out_data[i], out_ref_data[i], 1e-5);
  }

  // model release
  npu::OpList::Global().clear();
  npu::DeviceInfo::Global().Clear();
}

TEST(NPUBridges, mul) {
  for (auto n : {1, 3}) {
    for (auto c : {2, 4}) {
      for (auto h : {5, 7}) {
        for (auto w : {8, 9}) {
          for (auto o : {3, 7}) {
            for (auto x_num_col_dims : {1 /*, 2, 3*/}) {
              for (auto y_num_col_dims : {1 /*, 2, 3*/}) {
                test_mul(n, c, h, w, o, x_num_col_dims, y_num_col_dims);
              }
            }
          }
        }
      }
    }
  }
}

}  // namespace bridge
}  // namespace npu
}  // namespace lite
}  // namespace paddle

USE_LITE_OP(mul);
USE_NPU_BRIDGE(mul);

USE_LITE_OP(graph_op);
USE_LITE_KERNEL(graph_op, kNPU, kFloat, kNCHW, def);
