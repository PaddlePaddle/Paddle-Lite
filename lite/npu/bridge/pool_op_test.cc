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

#include "lite/operators/pool_op.h"
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

void pool_ref(const std::shared_ptr<operators::PoolOpLite> op) {
  Scope* scope = op->scope();
  const OpInfo* op_info = op->op_info();
  auto x = scope->FindVar(op_info->Input("X").front())->GetMutable<Tensor>();
  auto out =
      scope->FindVar(op_info->Output("Out").front())->GetMutable<Tensor>();
  auto& in_dims = x->dims();
  auto& out_dims = out->dims();

  const float* src_ptr = x->data<const float>();
  float* dst_ptr = out->mutable_data<float>();

  std::vector<int> ksize = op_info->GetAttr<std::vector<int>>("ksize");
  std::vector<int> strides = op_info->GetAttr<std::vector<int>>("strides");
  std::vector<int> paddings = op_info->GetAttr<std::vector<int>>("paddings");

  std::string pooling_type = op_info->GetAttr<std::string>("pooling_type");
  bool global_pooling = op_info->GetAttr<bool>("global_pooling");

  int in_n = in_dims[0];
  int in_c = in_dims[1];
  int in_h = in_dims[2];
  int in_w = in_dims[3];
  int size_in_n = in_c * in_h * in_w;
  int size_in_c = in_h * in_w;

  int out_h = out_dims[2];
  int out_w = out_dims[3];
  int size_out_n = in_c * out_h * out_w;
  int size_out_c = out_h * out_w;

  int window_h = ksize[0];
  int window_w = ksize[1];
  int stride_h = strides[0];
  int stride_w = strides[1];
  int pad_h = paddings[0];
  int pad_w = paddings[1];

  if (global_pooling == true) {
    for (int n = 0; n < in_n; ++n) {
      for (int c = 0; c < in_c; ++c) {
        const float* src = src_ptr + n * size_in_n + c * size_in_c;
        float res = src[0];
        if (pooling_type == "max") {
          for (int i = 1; i < size_in_c; ++i) {
            float cur_val = src[i];
            res = cur_val > res ? cur_val : res;
          }
        } else if (pooling_type == "avg") {
          for (int i = 1; i < size_in_c; ++i) {
            float cur_val = src[i];
            res += cur_val;
          }
          res /= size_in_c;
        }
        dst_ptr[n * size_out_n + c] = res;
      }
    }
  } else {
    for (int n = 0; n < in_n; ++n) {
      for (int c = 0; c < in_c; ++c) {
        for (int h = 0; h < out_h; ++h) {
          int sh = h * stride_h;
          int eh = sh + window_h;
          sh = (sh - pad_h) < 0 ? 0 : sh - pad_h;
          eh = (eh - pad_h) > in_h ? in_h : eh - pad_h;
          for (int w = 0; w < out_w; ++w) {
            int sw = w * stride_w;
            int ew = sw + window_w;
            sw = (sw - pad_w) < 0 ? 0 : sw - pad_w;
            ew = (ew - pad_w) > in_w ? in_w : ew - pad_w;
            int pooling_size = (ew - sw) * (eh - sh);
            if (pooling_size == 0) continue;
            float res = 0.f;
            for (int kh = sh; kh < eh; ++kh) {
              for (int kw = sw; kw < ew; ++kw) {
                int src_idx = n * size_in_n + c * size_in_c + kh * in_w + kw;
                if (kh == sh && kw == sw) {
                  res = src_ptr[src_idx];
                } else {
                  if (pooling_type == "max") {
                    res = res >= src_ptr[src_idx] ? res : src_ptr[src_idx];
                  }
                  if (pooling_type == "avg") {
                    res += src_ptr[src_idx];
                  }
                }
              }
            }
            if (pooling_type == "avg") {
              res /= pooling_size;
            }
            dst_ptr[n * size_out_n + c * size_out_c + h * out_w + w] = res;
          }
        }
      }
    }
  }
}

void test_pool(int bs, int ic, int ih, int iw) {
  const auto& bridges = lite::npu::bridge::Factory::Instance();
  const auto& supported_lists = bridges.AllFunctions();
  CHECK(bridges.HasType("pool2d"));

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
  cpp::OpDesc pool_op_desc;
  pool_op_desc.SetType("pool2d");
  pool_op_desc.SetInput("X", {x_var_name});
  pool_op_desc.SetOutput("Out", {out_var_name});
  pool_op_desc.SetAttr("pooling_type", std::string("max"));
  pool_op_desc.SetAttr("ksize", std::vector<int>({3, 3}));
  pool_op_desc.SetAttr("global_pooling", false);
  pool_op_desc.SetAttr("strides", std::vector<int>({1, 1}));
  pool_op_desc.SetAttr("paddings", std::vector<int>({1, 1}));

  std::shared_ptr<operators::PoolOpLite> pool_op =
      std::make_shared<operators::PoolOpLite>("pool2d");
  pool_op->SetValidPlaces({Place{TARGET(kX86), PRECISION(kFloat)},
                           Place{TARGET(kARM), PRECISION(kFloat)}});
  pool_op->Attach(pool_op_desc, &scope);
  pool_op->CheckShape();
  pool_op->InferShape();

  // convert op and build IR graph
  ge::TensorDesc x_desc(
      ge::Shape(x->dims().Vectorize()), ge::FORMAT_NCHW, ge::DT_FLOAT);
  std::shared_ptr<ge::op::Data> x_node =
      std::make_shared<ge::op::Data>(x_var_name);
  x_node->update_input_desc_x(x_desc);
  node_map_type inputs_map;
  inputs_map[x_var_name] = x_node;
  auto outputs_map =
      supported_lists.at(pool_op->op_info()->Type())(pool_op, inputs_map);
  CHECK_GT(outputs_map.size(), 0);

  // compile IR graph to om model
  std::vector<ge::Operator> graph_inputs{*inputs_map[x_var_name]};
  std::vector<ge::Operator> graph_outputs{*outputs_map[out_var_name]};
  std::string model_name(UniqueName("test_pool") + ".om");
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
  pool_ref(pool_op);

  // compare results
  auto* out_data = out->mutable_data<float>();
  auto* out_ref_data = out_ref->mutable_data<float>();
  for (int i = 0; i < out->numel(); i++) {
    EXPECT_NEAR(out_data[i], out_ref_data[i], 1e-5);
  }
}

TEST(NPUBridges, pool) {
  for (auto bs : {3}) {
    for (auto ic : {7}) {
      for (auto ih : {2}) {
        for (auto iw : {4}) {
          test_pool(bs, ic, ih, iw);
        }
      }
    }
  }
}

}  // namespace bridge
}  // namespace npu
}  // namespace lite
}  // namespace paddle

USE_LITE_OP(pool2d);
USE_NPU_BRIDGE(pool2d);

USE_LITE_OP(graph_op);
USE_LITE_KERNEL(graph_op, kNPU, kFloat, kNCHW, def);
