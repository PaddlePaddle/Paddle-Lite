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

#include "lite/operators/conv_op.h"
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

void conv_ref(const std::shared_ptr<operators::ConvOpLite> op) {
  Scope* scope = op->scope();
  const OpInfo* op_info = op->op_info();
  auto input =
      scope->FindVar(op_info->Input("Input").front())->GetMutable<Tensor>();
  auto filter =
      scope->FindVar(op_info->Input("Filter").front())->GetMutable<Tensor>();
  auto output =
      scope->FindVar(op_info->Output("Output").front())->GetMutable<Tensor>();
  std::vector<int32_t> strides =
      op_info->GetAttr<std::vector<int32_t>>("strides");
  std::vector<int32_t> paddings =
      op_info->GetAttr<std::vector<int32_t>>("paddings");
  int32_t groups = op_info->GetAttr<int32_t>("groups");
  std::vector<int32_t> dilations =
      op_info->GetAttr<std::vector<int32_t>>("dilations");
  bool fuse_relu = op_info->GetAttr<bool>("fuse_relu");
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
  auto input_dims = input->dims();
  auto filter_dims = filter->dims();
  auto output_dims = output->dims();
  auto input_data = input->mutable_data<float>();
  auto filter_data = filter->mutable_data<float>();
  auto output_data = output->mutable_data<float>();
  int kernel_w = filter_dims[3];
  int kernel_h = filter_dims[2];
  int stride_w = strides[0];
  int stride_h = strides[1];
  int dila_w = dilations[0];
  int dila_h = dilations[1];
  int pad_w = paddings[0];
  int pad_h = paddings[1];
  int batch_size = input_dims[0];
  int in_ch_size = input_dims[1];
  int in_h = input_dims[2];
  int in_w = input_dims[3];
  int out_ch_size = output_dims[1];
  int out_h = output_dims[2];
  int out_w = output_dims[3];
  int out_c_group = out_ch_size / groups;
  int in_c_group = in_ch_size / groups;
  for (int n = 0; n < batch_size; ++n) {
    for (int g = 0; g < groups; ++g) {
      for (int oc = 0; oc < out_c_group; ++oc) {
        for (int oh = 0; oh < out_h; ++oh) {
          for (int ow = 0; ow < out_w; ++ow) {
            int out_idx = n * groups * out_c_group * out_h * out_w +
                          g * out_c_group * out_h * out_w + oc * out_h * out_w +
                          oh * out_w + ow;
            float out_value =
                bias_data != nullptr ? (bias_data[g * out_c_group + oc]) : 0;
            // + out_value *= beta;
            for (int ic = 0; ic < in_c_group; ++ic) {
              for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                  int iw = ow * stride_w - pad_w + kw * (dila_w);
                  int ih = oh * stride_h - pad_h + kh * (dila_h);
                  if (iw < 0 || iw >= in_w) continue;
                  if (ih < 0 || ih >= in_h) continue;
                  int in_idx = n * in_ch_size * in_h * in_w +
                               g * in_c_group * in_h * in_w + ic * in_h * in_w +
                               ih * in_w + iw;
                  int filter_idx =
                      g * out_c_group * in_c_group * kernel_h * kernel_w +
                      oc * in_c_group * kernel_h * kernel_w +
                      ic * kernel_h * kernel_w + kh * kernel_w + kw;
                  out_value += input_data[in_idx] * filter_data[filter_idx];
                }
              }
            }
            if (fuse_relu) {
              out_value = out_value > 0 ? out_value : 0;
            }
            output_data[out_idx] = out_value;
          }
        }
      }
    }
  }
}

void test_conv(int bs,
               int ic,
               int oc,
               int ih,
               int iw,
               bool has_bias,
               bool fuse_relu,
               bool depthwise,
               int dilation,
               int stride,
               int padding,
               int ks) {
  const auto& bridges = lite::npu::bridge::Factory::Instance();
  const auto& supported_lists = bridges.AllFunctions();
  CHECK(bridges.HasType("conv2d"));
  CHECK(bridges.HasType("depthwise_conv2d"));

  // prepare input&output variables
  Scope scope;
  std::string input_var_name("input");
  std::string filter_var_name("filter");
  std::string bias_var_name("bias");
  std::string output_var_name("output");
  std::string output_ref_var_name("output_ref");
  auto* input = scope.Var(input_var_name)->GetMutable<Tensor>();
  auto* filter = scope.Var(filter_var_name)->GetMutable<Tensor>();
  auto* bias = scope.Var(bias_var_name)->GetMutable<Tensor>();
  auto* output = scope.Var(output_var_name)->GetMutable<Tensor>();
  auto* output_ref = scope.Var(output_ref_var_name)->GetMutable<Tensor>();

  // get group size and input&filter shape
  int groups = 1;
  depthwise &= ks != 1;
  if (depthwise) {  // depthwise convolution ?
    groups = oc = ic;
  }
  std::vector<int64_t> input_shape = {bs, ic, ih, iw};
  std::vector<int64_t> filter_shape = {oc, ic / groups, ks, ks};
  input->Resize(input_shape);
  filter->Resize(filter_shape);

  // initialize input&output data
  std::default_random_engine rand_eng;
  std::uniform_real_distribution<float> rand_dist(-5.0f, 5.0f);
  for (int i = 0; i < input->dims().production(); i++) {
    float rand_value = half2float(float2half(rand_dist(rand_eng)));
    input->mutable_data<float>()[i] = rand_value;
  }
  for (int i = 0; i < filter->dims().production(); i++) {
    float rand_value = half2float(float2half(rand_dist(rand_eng) * 0.1f));
    filter->mutable_data<float>()[i] = rand_value;
  }

  // create conv2d op
  cpp::OpDesc conv_op_desc;
  conv_op_desc.SetType(depthwise ? "depthwise_conv2d" : "conv2d");
  conv_op_desc.SetInput("Input", {input_var_name});
  conv_op_desc.SetInput("Filter", {filter_var_name});
  conv_op_desc.SetOutput("Output", {output_var_name});
  conv_op_desc.SetAttr("dilations", std::vector<int32_t>({dilation, dilation}));
  conv_op_desc.SetAttr("strides", std::vector<int32_t>({stride, stride}));
  conv_op_desc.SetAttr("paddings", std::vector<int32_t>({padding, padding}));
  conv_op_desc.SetAttr("groups", groups);
  conv_op_desc.SetAttr("fuse_relu", static_cast<bool>(fuse_relu));
  if (has_bias) {
    bias->Resize({bs, oc, 1, 1});
    for (int i = 0; i < bias->dims().production(); i++) {
      float rand_value = half2float(float2half(rand_dist(rand_eng) * 0.01f));
      bias->mutable_data<float>()[i] = rand_value;
    }
    conv_op_desc.SetInput("Bias", {bias_var_name});
  }

  auto conv_op = std::make_shared<operators::ConvOpLite>(conv_op_desc.Type());
  conv_op->SetValidPlaces({Place{TARGET(kHost), PRECISION(kFloat)},
                           Place{TARGET(kARM), PRECISION(kFloat)}});
  CHECK(conv_op->Attach(conv_op_desc, &scope));
  CHECK(conv_op->CheckShape());
  CHECK(conv_op->InferShape());

  // convert conv2d op and build IR graph
  ge::TensorDesc input_desc(
      ge::Shape(input->dims().Vectorize()), ge::FORMAT_NCHW, ge::DT_FLOAT);
  auto input_node = std::make_shared<ge::op::Data>(input_var_name);
  input_node->update_input_desc_x(input_desc);
  node_map_type inputs_map;
  inputs_map[input_var_name] = input_node;
  auto outputs_map =
      supported_lists.at(conv_op->op_info()->Type())(conv_op, inputs_map);
  CHECK_GT(outputs_map.size(), 0);

  // compile IR graph to om model
  std::vector<ge::Operator> graph_inputs{*inputs_map[input_var_name]};
  std::vector<ge::Operator> graph_outputs{*outputs_map[output_var_name]};
  std::string model_name(UniqueName("test_conv2d") + ".om");
  CHECK(npu::BuildNPUClient(graph_inputs, graph_outputs, model_name));

  // create graph op
  cpp::OpDesc graph_op_desc;
  graph_op_desc.SetType("graph_op");
  graph_op_desc.SetInput("Inputs", {input_var_name});
  graph_op_desc.SetOutput("Outputs", {output_var_name});
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
  output_ref->CopyDataFrom(*output);

  // execute reference implementation and save to output tensor('out')
  conv_ref(conv_op);

  // compare results
  auto* output_data = output->mutable_data<float>();
  auto* output_ref_data = output_ref->mutable_data<float>();
  for (int i = 0; i < output->dims().production(); i++) {
    EXPECT_NEAR(output_data[i], output_ref_data[i], 1e-1);
  }

  // model release
  npu::OpList::Global().clear();
  npu::DeviceInfo::Global().Clear();
}

TEST(NPUBridges, conv) {
#if 0
  for (auto bs : {1}) {
    for (auto ic : {3, 6}) {
      for (auto oc : {6, 9}) {
        for (auto ih : {14, 28}) {
          for (auto iw : {14, 28}) {
            for (auto has_bias : {false /*, true*/}) {
              for (auto fuse_relu : {false /*, true*/}) {
                for (auto depthwise : {/* false,*/ true}) {
                  for (auto dilation : {1}) {
                    for (auto stride : {1, 2}) {
                      for (auto padding : {0, 1 /*, 2*/}) {
                        for (auto ks : {1, 3 /* , 5*/}) {
                          LOG(INFO)
                              << "bs: " << bs << " ic: " << ic << " oc: " << oc
                              << " ih: " << ih << " iw: " << iw
                              << " has_bias: " << has_bias
                              << " fuse_relu: " << fuse_relu
                              << " depthwise: " << depthwise
                              << " dilation: " << dilation
                              << " stride: " << stride
                              << " padding: " << padding << " ks: " << ks;
                          test_conv(bs,
                                    ic,
                                    oc,
                                    ih,
                                    iw,
                                    has_bias,
                                    fuse_relu,
                                    depthwise,
                                    dilation,
                                    stride,
                                    padding,
                                    ks);
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
#else
  test_conv(1, 8, 6, 14, 14, false, false, true, 1, 1, 1, 3);
#endif
}

}  // namespace bridge
}  // namespace npu
}  // namespace lite
}  // namespace paddle

USE_LITE_OP(conv2d);
USE_NPU_BRIDGE(conv2d);

USE_LITE_OP(depthwise_conv2d);
USE_NPU_BRIDGE(depthwise_conv2d);

USE_LITE_OP(graph_op);
USE_LITE_KERNEL(graph_op, kNPU, kFloat, kNCHW, def);
