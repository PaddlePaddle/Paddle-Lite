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

#include "lite/operators/fc_op.h"
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

void fc_ref(const std::shared_ptr<operators::FcOpLite> op) {
  Scope* scope = op->scope();
  const OpInfo* op_info = op->op_info();
  auto input =
      scope->FindVar(op_info->Input("Input").front())->GetMutable<Tensor>();
  auto w = scope->FindVar(op_info->Input("W").front())->GetMutable<Tensor>();
  auto out =
      scope->FindVar(op_info->Output("Out").front())->GetMutable<Tensor>();
  int32_t in_num_col_dims = op_info->GetAttr<int32_t>("in_num_col_dims");
  Tensor* bias = nullptr;
  float* bias_data = nullptr;
  if (op_info->HasInput("Bias")) {
    auto bias_var_names = op_info->Input("Bias");
    if (bias_var_names.size() > 0) {
      auto bias_var_name = bias_var_names.front();
      bias = scope->FindVar(bias_var_name)->GetMutable<lite::Tensor>();
      bias_data = bias->mutable_data<float>();
    }
  }
  auto input_data = input->data<float>();
  auto w_data = w->mutable_data<float>();
  auto out_data = out->mutable_data<float>();
  auto in_mat_dims = input->dims().Flattern2D(in_num_col_dims);
  int out_num_classes = w->dims()[1];
  const int M = in_mat_dims[0];
  const int K = in_mat_dims[1];
  const int N = out_num_classes;
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      out_data[m * N + n] = 0;
      for (int k = 0; k < K; ++k) {
        out_data[m * N + n] += input_data[m * K + k] * w_data[k * N + n];
      }
    }
  }
  if (bias_data != nullptr) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        out_data[m * N + n] += bias_data[n];
      }
    }
  }
}

void test_fc(int bs,
             int ic,
             int ih,
             int iw,
             int in_num_col_dims,
             int out_num_classes,
             bool has_bias) {
  const auto& bridges = lite::npu::bridge::Factory::Instance();
  const auto& supported_lists = bridges.AllFunctions();
  CHECK(bridges.HasType("fc"));

  // prepare input&output variables
  Scope scope;
  std::string input_var_name("input");
  std::string w_var_name("w");
  std::string bias_var_name("bias");
  std::string out_var_name("out");
  std::string out_ref_var_name("out_ref");
  auto* input = scope.Var(input_var_name)->GetMutable<Tensor>();
  auto* w = scope.Var(w_var_name)->GetMutable<Tensor>();
  auto* bias = scope.Var(bias_var_name)->GetMutable<Tensor>();
  auto* out = scope.Var(out_var_name)->GetMutable<Tensor>();
  auto* out_ref = scope.Var(out_ref_var_name)->GetMutable<Tensor>();
  input->Resize({bs, ic, ih, iw});

  // get w shape
  auto in_mat_dims = input->dims().Flattern2D(in_num_col_dims);
  std::vector<int64_t> w_shape = {in_mat_dims[1], out_num_classes};
  w->Resize(w_shape);

  // initialize input&output data
  std::default_random_engine rand_eng;
  std::uniform_real_distribution<float> rand_dist(-5.0f, 5.0f);
  for (int i = 0; i < input->dims().production(); i++) {
    float rand_value = half2float(float2half(rand_dist(rand_eng)));
    input->mutable_data<float>()[i] = rand_value;
  }
  for (int i = 0; i < w->dims().production(); i++) {
    float rand_value = half2float(float2half(rand_dist(rand_eng) * 0.1f));
    w->mutable_data<float>()[i] = rand_value;
  }

  // create fc op
  cpp::OpDesc fc_op_desc;
  fc_op_desc.SetType("fc");
  fc_op_desc.SetInput("Input", {input_var_name});
  fc_op_desc.SetInput("W", {w_var_name});
  fc_op_desc.SetOutput("Out", {out_var_name});
  fc_op_desc.SetAttr("in_num_col_dims", static_cast<int>(in_num_col_dims));
  if (has_bias) {
    bias->Resize({out_num_classes});
    for (int i = 0; i < bias->dims().production(); i++) {
      float rand_value = half2float(float2half(rand_dist(rand_eng) * 0.01f));
      bias->mutable_data<float>()[i] = rand_value;
    }
    fc_op_desc.SetInput("Bias", {bias_var_name});
  }

  auto fc_op = std::make_shared<operators::FcOpLite>(fc_op_desc.Type());
  fc_op->SetValidPlaces({Place{TARGET(kHost), PRECISION(kFloat)},
                         Place{TARGET(kARM), PRECISION(kFloat)}});
  CHECK(fc_op->Attach(fc_op_desc, &scope));
  CHECK(fc_op->CheckShape());
  CHECK(fc_op->InferShape());

  // convert fc op and build IR graph
  ge::TensorDesc input_desc(
      ge::Shape(input->dims().Vectorize()), ge::FORMAT_NCHW, ge::DT_FLOAT);
  auto input_node = std::make_shared<ge::op::Data>(input_var_name);
  input_node->update_input_desc_x(input_desc);
  node_map_type inputs_map;
  inputs_map[input_var_name] = input_node;
  auto outputs_map =
      supported_lists.at(fc_op->op_info()->Type())(fc_op, inputs_map);
  CHECK_GT(outputs_map.size(), 0);

  // compile IR graph to om model
  std::vector<ge::Operator> graph_inputs{*inputs_map[input_var_name]};
  std::vector<ge::Operator> graph_outputs{*outputs_map[out_var_name]};
  std::string model_name(UniqueName("test_fc") + ".om");
  CHECK(npu::BuildNPUClient(graph_inputs, graph_outputs, model_name));

  // create graph op
  cpp::OpDesc graph_op_desc;
  graph_op_desc.SetType("graph_op");
  graph_op_desc.SetInput("Inputs", {input_var_name});
  graph_op_desc.SetOutput("Outputs", {out_var_name});
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
  fc_ref(fc_op);

  // compare results
  auto* out_data = out->mutable_data<float>();
  auto* out_ref_data = out_ref->mutable_data<float>();
  for (int i = 0; i < out->dims().production(); i++) {
    EXPECT_NEAR(out_data[i], out_ref_data[i], 1e-2);
  }

  // model release
  npu::OpList::Global().clear();
  npu::DeviceInfo::Global().Clear();
}

TEST(NPUBridges, fc) {
#if 0
  for (auto bs : {1}) {
    for (auto ic : {2, 4}) {
      for (auto ih : {5, 7}) {
        for (auto iw : {8, 9}) {
          for (auto in_num_col_dims : {1 /*, 2, 3*/}) {
            for (auto out_num_classes : {1, 3, 8}) {
              for (bool has_bias : {true/*, false*/}) {
                test_fc(
                    bs, ic, ih, iw, in_num_col_dims, out_num_classes, has_bias);
              }
            }
          }
        }
      }
    }
  }
#else
  test_fc(1, 1280, 1, 1, 1, 1000, true);
#endif
}

}  // namespace bridge
}  // namespace npu
}  // namespace lite
}  // namespace paddle

USE_LITE_OP(fc);
USE_NPU_BRIDGE(fc);

USE_LITE_OP(graph_op);
USE_LITE_KERNEL(graph_op, kNPU, kFloat, kNCHW, def);
