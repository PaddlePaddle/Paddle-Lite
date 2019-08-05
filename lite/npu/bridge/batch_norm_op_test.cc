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

#include "lite/operators/batch_norm_op.h"
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

template <typename dtype>
void batch_norm_ref(const std::shared_ptr<operators::BatchNormOp> op) {
  Scope* scope = op->scope();
  const OpInfo* op_info = op->op_info();
  auto x = scope->FindVar(op_info->Input("X").front())->GetMutable<Tensor>();
  auto y = scope->FindVar(op_info->Output("Y").front())->GetMutable<Tensor>();
  auto bias =
      scope->FindVar(op_info->Input("Bias").front())->GetMutable<Tensor>();
  auto scale =
      scope->FindVar(op_info->Input("Scale").front())->GetMutable<Tensor>();
  auto mean =
      scope->FindVar(op_info->Input("Mean").front())->GetMutable<Tensor>();
  auto variance =
      scope->FindVar(op_info->Input("Variance").front())->GetMutable<Tensor>();

  auto x_data = x->data<dtype>();
  auto y_data = y->mutable_data<dtype>();
  auto scale_data = scale->mutable_data<dtype>();
  auto bias_data = bias->mutable_data<dtype>();
  auto mean_data = mean->mutable_data<dtype>();
  auto variance_data = variance->mutable_data<dtype>();
  DDim x_dims = x->dims();

  float epsilon = op_info->GetAttr<float>("epsilon");
  float momentum = op_info->GetAttr<float>("momentum");
  auto data_layout = op_info->GetAttr<std::string>("data_layout");

  bool global_stats = op_info->GetAttr<bool>("use_global_stats");
  if (global_stats) {
    int64_t outer_size = 0;
    int64_t channel_size = 0;
    int64_t inner_size = 0;
    if (data_layout == "NCHW") {
      outer_size = x_dims[0];
      channel_size = x_dims[1];
      inner_size = x_dims.Slice(2, x_dims.size()).production();
    } else {
      LOG(FATAL) << "Unknown storage order: " << data_layout;
    }
    auto x_ptr = x_data;
    auto y_ptr = y_data;
    for (int o = 0; o < outer_size; o++) {
      for (int c = 0; c < channel_size; c++) {
        for (int i = 0; i < inner_size; i++) {
          dtype norm_x =
              (*x_ptr - mean_data[c]) / std::sqrt(variance_data[c] + epsilon);
          *y_ptr = norm_x * scale_data[c] + bias_data[c];
          x_ptr++;
          y_ptr++;
        }
      }
    }
  }
}

void test_batch_norm(int bs, int ic, int ih, int iw) {
  const auto& bridges = lite::npu::bridge::Factory::Instance();
  const auto& supported_lists = bridges.AllFunctions();
  CHECK(bridges.HasType("batch_norm"));

  std::string x_var_name = "x";
  std::string out_var_name = "out";
  std::string out_ref_var_name = "out_ref";
  std::string scale_var_name = "scale";
  std::string bias_var_name = "bias";
  std::string mean_var_name = "mean";
  std::string variance_var_name = "variance";

  // prepare input&output variables
  Scope scope;
  auto* x = scope.Var(x_var_name)->GetMutable<Tensor>();
  auto* scale = scope.Var(scale_var_name)->GetMutable<Tensor>();
  auto* bias = scope.Var(bias_var_name)->GetMutable<Tensor>();
  auto* mean = scope.Var(mean_var_name)->GetMutable<Tensor>();
  auto* variance = scope.Var(variance_var_name)->GetMutable<Tensor>();
  auto* out = scope.Var(out_var_name)->GetMutable<Tensor>();
  auto* out_ref = scope.Var(out_ref_var_name)->GetMutable<Tensor>();
  x->Resize({bs, ic, ih, iw});
  scale->Resize({ic});
  bias->Resize({ic});
  mean->Resize({ic});
  variance->Resize({ic});

  // initialize input&output data
  std::default_random_engine rand_eng;
  std::uniform_real_distribution<float> rand_dist(-5.0f, 5.0f);
  for (int i = 0; i < x->numel(); i++) {
    float rand_value = half2float(float2half(rand_dist(rand_eng)));
    x->mutable_data<float>()[i] = rand_value;
  }
  for (int i = 0; i < scale->numel(); i++) {
    float rand_value = half2float(float2half(rand_dist(rand_eng)));
    scale->mutable_data<float>()[i] = rand_value;
  }
  for (int i = 0; i < bias->numel(); i++) {
    float rand_value = half2float(float2half(rand_dist(rand_eng)));
    bias->mutable_data<float>()[i] = rand_value;
  }
  for (int i = 0; i < mean->numel(); i++) {
    float rand_value = half2float(float2half(rand_dist(rand_eng)));
    mean->mutable_data<float>()[i] = rand_value;
  }
  for (int i = 0; i < variance->numel(); i++) {
    float rand_value = half2float(float2half(rand_dist(rand_eng)));
    variance->mutable_data<float>()[i] = rand_value;
  }

  // create op
  cpp::OpDesc batch_norm_op_desc;
  batch_norm_op_desc.SetType("batch_norm");
  batch_norm_op_desc.SetInput("X", {x_var_name});
  batch_norm_op_desc.SetInput("Scale", {scale_var_name});
  batch_norm_op_desc.SetInput("Bias", {bias_var_name});
  batch_norm_op_desc.SetInput("Mean", {mean_var_name});
  batch_norm_op_desc.SetInput("Variance", {variance_var_name});
  batch_norm_op_desc.SetOutput("Y", {out_var_name});
  batch_norm_op_desc.SetAttr("is_test", static_cast<int>(1));
  batch_norm_op_desc.SetAttr("use_global_stats", true);
  batch_norm_op_desc.SetAttr("epsilon", 1e-5f);
  batch_norm_op_desc.SetAttr("momentum", 0.9f);
  batch_norm_op_desc.SetAttr("data_layout", std::string("NCHW"));

  auto batch_norm_op =
      std::make_shared<operators::BatchNormOp>(batch_norm_op_desc.Type());
  batch_norm_op->SetValidPlaces({Place{TARGET(kHost), PRECISION(kFloat)},
                                 Place{TARGET(kARM), PRECISION(kFloat)}});
  CHECK(batch_norm_op->Attach(batch_norm_op_desc, &scope));
  CHECK(batch_norm_op->CheckShape());
  CHECK(batch_norm_op->InferShape());

  // convert op and build IR graph
  ge::TensorDesc x_desc(
      ge::Shape(x->dims().Vectorize()), ge::FORMAT_NCHW, ge::DT_FLOAT);
  auto x_node = std::make_shared<ge::op::Data>(x_var_name);
  x_node->update_input_desc_x(x_desc);
  node_map_type inputs_map;
  inputs_map[x_var_name] = x_node;
  auto outputs_map = supported_lists.at(batch_norm_op->op_info()->Type())(
      batch_norm_op, inputs_map);
  CHECK_GT(outputs_map.size(), 0);

  // compile IR graph to om model
  std::vector<ge::Operator> graph_inputs{*inputs_map[x_var_name]};
  std::vector<ge::Operator> graph_outputs{*outputs_map[out_var_name]};
  std::string model_name(UniqueName("test_batch_norm") + ".om");
  CHECK(npu::BuildNPUClient(graph_inputs, graph_outputs, model_name));

  // create graph op
  cpp::OpDesc graph_op_desc;
  graph_op_desc.SetType("graph_op");
  graph_op_desc.SetInput("Inputs", {x_var_name});
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
  batch_norm_ref<float>(batch_norm_op);

  // compare results
  auto* out_data = out->mutable_data<float>();
  auto* out_ref_data = out_ref->mutable_data<float>();
  for (int i = 0; i < out->numel(); i++) {
    EXPECT_NEAR(out_data[i], out_ref_data[i], 1e-5);
  }

  // release model resources
  npu::OpList::Global().clear();
  npu::DeviceInfo::Global().Clear();
}

TEST(NPUBridges, batch_norm) {
  for (auto bs : {3}) {
    for (auto ic : {7}) {
      for (auto ih : {2}) {
        for (auto iw : {4}) {
          test_batch_norm(bs, ic, ih, iw);
        }
      }
    }
  }
}

}  // namespace bridge
}  // namespace npu
}  // namespace lite
}  // namespace paddle

USE_LITE_OP(batch_norm);
USE_NPU_BRIDGE(batch_norm);

USE_LITE_OP(graph_op);
USE_LITE_KERNEL(graph_op, kNPU, kFloat, kNCHW, def);
