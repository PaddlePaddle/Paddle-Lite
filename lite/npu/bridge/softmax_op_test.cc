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

#include "lite/operators/softmax_op.h"
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
void softmax_ref(const std::shared_ptr<operators::SoftmaxOp> op) {
  Scope* scope = op->scope();
  const OpInfo* op_info = op->op_info();
  auto x = scope->FindVar(op_info->Input("X").front())->GetMutable<Tensor>();
  auto out =
      scope->FindVar(op_info->Output("Out").front())->GetMutable<Tensor>();
  auto x_data = x->data<dtype>();
  auto out_data = out->mutable_data<dtype>();
  DDim x_dims = x->dims();

  auto x_rank = x_dims.size();
  int axis = op_info->GetAttr<int>("axis");
  if (axis < 0) {
    axis += x_rank;
  }
  int axis_size = x_dims[axis];
  int outer_num = x_dims.Slice(0, axis).production();
  int inner_num = x_dims.Slice(axis + 1, x_rank).production();
  int compute_size = outer_num * inner_num;
  for (int i = 0; i < compute_size; i++) {
    int idx_inner = i % inner_num;
    int idx_outer = (i / inner_num) * axis_size;
    int start = idx_outer * inner_num + idx_inner;
    int offset;

    offset = start;
    dtype max_data = std::numeric_limits<dtype>::lowest();
    for (int j = 0; j < axis_size; j++) {
      max_data = x_data[offset] > max_data ? x_data[offset] : max_data;
      offset += inner_num;
    }

    offset = start;
    dtype sum_data = (dtype)0;
    for (int j = 0; j < axis_size; j++) {
      out_data[offset] = exp(x_data[offset] - max_data);
      sum_data += out_data[offset];
      offset += inner_num;
    }

    offset = start;
    for (int j = 0; j < axis_size; j++) {
      out_data[offset] /= sum_data;
      offset += inner_num;
    }
  }
}

void test_softmax(int bs, int ic, int ih, int iw) {
  const auto& bridges = lite::npu::bridge::Factory::Instance();
  const auto& supported_lists = bridges.AllFunctions();
  CHECK(bridges.HasType("softmax"));

  std::string x_var_name = "x";
  std::string out_var_name = "out";
  std::string out_ref_var_name = "out_ref";

  // prepare input&output variables
  Scope scope;
  auto* x = scope.Var(x_var_name)->GetMutable<Tensor>();
  auto* out = scope.Var(out_var_name)->GetMutable<Tensor>();
  auto* out_ref = scope.Var(out_ref_var_name)->GetMutable<Tensor>();
  x->Resize({bs, ic, ih, iw});

  // initialize input&output data
  std::default_random_engine rand_eng;
  std::uniform_real_distribution<float> rand_dist(-5.0f, 5.0f);
  for (int i = 0; i < x->numel(); i++) {
    float fp32_value = rand_dist(rand_eng);
    float fp16_value = half2float(float2half(fp32_value));
    x->mutable_data<float>()[i] = fp16_value;
  }

  // create op
  cpp::OpDesc softmax_op_desc;
  softmax_op_desc.SetType("softmax");
  softmax_op_desc.SetInput("X", {x_var_name});
  softmax_op_desc.SetOutput("Out", {out_var_name});
  softmax_op_desc.SetAttr("axis", -1);

  std::shared_ptr<operators::SoftmaxOp> softmax_op =
      std::make_shared<operators::SoftmaxOp>("softmax");
  softmax_op->SetValidPlaces({Place{TARGET(kHost), PRECISION(kFloat)},
                              Place{TARGET(kARM), PRECISION(kFloat)}});
  softmax_op->Attach(softmax_op_desc, &scope);
  softmax_op->CheckShape();
  softmax_op->InferShape();

  // convert op and build IR graph
  ge::TensorDesc x_desc(
      ge::Shape(x->dims().Vectorize()), ge::FORMAT_NCHW, ge::DT_FLOAT);
  std::shared_ptr<ge::op::Data> x_node =
      std::make_shared<ge::op::Data>(x_var_name);
  x_node->update_input_desc_x(x_desc);
  node_map_type inputs_map;
  inputs_map[x_var_name] = x_node;
  auto outputs_map =
      supported_lists.at(softmax_op->op_info()->Type())(softmax_op, inputs_map);
  CHECK_GT(outputs_map.size(), 0);

  // compile IR graph to om model
  std::vector<ge::Operator> graph_inputs{*inputs_map[x_var_name]};
  std::vector<ge::Operator> graph_outputs{*outputs_map[out_var_name]};
  std::string model_name(UniqueName("test_softmax") + ".om");
  CHECK(npu::BuildNPUClient(graph_inputs, graph_outputs, model_name));

  // create graph op
  cpp::OpDesc graph_op_desc;
  graph_op_desc.SetType("graph_op");
  graph_op_desc.SetInput("Inputs", {x_var_name});
  graph_op_desc.SetOutput("Outputs", {out_var_name});
  graph_op_desc.SetAttr("model_name", model_name);

  std::shared_ptr<operators::GraphOpLite> graph_op =
      std::make_shared<operators::GraphOpLite>("graph_op");
  graph_op->SetValidPlaces({Place{TARGET(kNPU), PRECISION(kFloat)}});
  graph_op->Attach(graph_op_desc, &scope);
  graph_op->CheckShape();
  graph_op->InferShape();

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
  softmax_ref<float>(softmax_op);

  // compare results
  auto* out_data = out->mutable_data<float>();
  auto* out_ref_data = out_ref->mutable_data<float>();
  for (int i = 0; i < out->numel(); i++) {
    EXPECT_NEAR(out_data[i], out_ref_data[i], 1e-2);
  }
}

TEST(NPUBridges, softmax) {
  for (auto bs : {3}) {
    for (auto ic : {7}) {
      for (auto ih : {2}) {
        for (auto iw : {4}) {
          test_softmax(bs, ic, ih, iw);
        }
      }
    }
  }
}

}  // namespace bridge
}  // namespace npu
}  // namespace lite
}  // namespace paddle

USE_LITE_OP(softmax);
USE_NPU_BRIDGE(softmax);

USE_LITE_OP(graph_op);
USE_LITE_KERNEL(graph_op, kNPU, kFloat, kNCHW, def);
