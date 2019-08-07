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

void test_elementwise_add(int bs, int ic, int ih, int iw) {
  const auto& bridges = lite::npu::bridge::Factory::Instance();
  const auto& supported_lists = bridges.AllFunctions();
  CHECK(bridges.HasType("elementwise_add"));

  std::string x_var_name = "x";
  std::string y_var_name = "y";
  std::string out_var_name = "out";
  std::string out_ref_var_name = "out_ref";

  // prepare input&output variables
  Scope scope;
  auto* x = scope.Var(x_var_name)->GetMutable<Tensor>();
  auto* y = scope.Var(y_var_name)->GetMutable<Tensor>();
  auto* out = scope.Var(out_var_name)->GetMutable<Tensor>();
  auto* out_ref = scope.Var(out_ref_var_name)->GetMutable<Tensor>();
  x->Resize({bs, ic, ih, iw});
  y->Resize({bs, ic, ih, iw});

  // initialize input&output data
  std::default_random_engine rand_eng;
  std::uniform_real_distribution<float> rand_dist(-5.0f, 5.0f);
  for (int i = 0; i < x->numel(); i++) {
    float rand_value = half2float(float2half(rand_dist(rand_eng)));
    x->mutable_data<float>()[i] = rand_value;
  }
  for (int i = 0; i < y->numel(); i++) {
    float rand_value = half2float(float2half(rand_dist(rand_eng)));
    y->mutable_data<float>()[i] = rand_value;
  }

  // create op
  cpp::OpDesc elementwise_add_op_desc;
  elementwise_add_op_desc.SetType("elementwise_add");
  elementwise_add_op_desc.SetInput("X", {x_var_name});
  elementwise_add_op_desc.SetInput("Y", {y_var_name});
  elementwise_add_op_desc.SetOutput("Out", {out_var_name});
  elementwise_add_op_desc.SetAttr("axis", -1);

  auto elementwise_add_op =
      std::make_shared<operators::ElementwiseOp>("elementwise_add");
  elementwise_add_op->SetValidPlaces({Place{TARGET(kHost), PRECISION(kFloat)},
                                      Place{TARGET(kARM), PRECISION(kFloat)}});
  CHECK(elementwise_add_op->Attach(elementwise_add_op_desc, &scope));
  CHECK(elementwise_add_op->CheckShape());
  CHECK(elementwise_add_op->InferShape());

  // convert op and build IR graph
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
  auto outputs_map = supported_lists.at(elementwise_add_op->op_info()->Type())(
      elementwise_add_op, inputs_map);
  CHECK_GT(outputs_map.size(), 0);

  // compile IR graph to om model
  std::vector<ge::Operator> graph_inputs{*inputs_map[x_var_name],
                                         *inputs_map[y_var_name]};
  std::vector<ge::Operator> graph_outputs{*outputs_map[out_var_name]};
  std::string model_name(UniqueName("test_elementwise_add") + ".om");
  CHECK(npu::BuildNPUClient(graph_inputs, graph_outputs, model_name));

  // create graph op
  cpp::OpDesc graph_op_desc;
  graph_op_desc.SetType("graph_op");
  graph_op_desc.SetInput("Inputs", {x_var_name, y_var_name});
  graph_op_desc.SetOutput("Outputs", {out_var_name});
  graph_op_desc.SetAttr("model_name", model_name);

  auto graph_op = std::make_shared<operators::GraphOpLite>("graph_op");
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
  elementwise_add_ref<float>(elementwise_add_op);

  // compare results
  auto* out_data = out->mutable_data<float>();
  auto* out_ref_data = out_ref->mutable_data<float>();
  for (int i = 0; i < out->numel(); i++) {
    EXPECT_NEAR(out_data[i], out_ref_data[i], 1e-5);
  }

  // model release
  npu::OpList::Global().clear();
  npu::DeviceInfo::Global().Clear();
}

TEST(NPUBridges, elementwise_add) {
  for (auto bs : {3}) {
    for (auto ic : {7}) {
      for (auto ih : {2}) {
        for (auto iw : {4}) {
          test_elementwise_add(bs, ic, ih, iw);
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

USE_LITE_OP(graph_op);
USE_LITE_KERNEL(graph_op, kNPU, kFloat, kNCHW, def);
