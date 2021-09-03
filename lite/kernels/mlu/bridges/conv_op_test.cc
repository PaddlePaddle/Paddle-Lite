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

#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"
#include "lite/core/subgraph/subgraph_bridge_registry.h"
#include "lite/kernels/mlu/bridges/test_helper.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace mlu {

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
  auto input_dims = input->dims();
  auto filter_dims = filter->dims();
  auto output_dims = output->dims();
  auto input_data = input->mutable_data<float>();
  auto filter_data = filter->mutable_data<float>();
  auto output_data = output->mutable_data<float>();
  int kernel_w = filter_dims[3];
  int kernel_h = filter_dims[2];
  int stride_w = strides[1];
  int stride_h = strides[0];
  int dila_w = dilations[1];
  int dila_h = dilations[0];
  int pad_w = paddings[2];
  int pad_h = paddings[0];
  int batch_size = input_dims[0];
  int in_ch_size = input_dims[1];
  int in_h = input_dims[2];
  int in_w = input_dims[3];
  int out_ch_size = output_dims[1];
  int out_h = output_dims[2];
  int out_w = output_dims[3];
  int out_c_group = out_ch_size / groups;
  int in_c_group = in_ch_size / groups;
  Tensor* bias = nullptr;
  float* bias_data = nullptr;
  bool is_channel_bias = false;
  if (op_info->HasInput("Bias")) {
    auto bias_var_names = op_info->Input("Bias");
    if (bias_var_names.size() > 0) {
      auto bias_var_name = bias_var_names.front();
      bias = scope->FindVar(bias_var_name)->GetMutable<lite::Tensor>();
      auto bias_dims = bias->dims();
      is_channel_bias = bias_dims.production() == out_ch_size;
      bias_data = bias->mutable_data<float>();
    }
  }
  for (int n = 0; n < batch_size; ++n) {
    for (int g = 0; g < groups; ++g) {
      for (int oc = 0; oc < out_c_group; ++oc) {
        for (int oh = 0; oh < out_h; ++oh) {
          for (int ow = 0; ow < out_w; ++ow) {
            int out_idx = n * groups * out_c_group * out_h * out_w +
                          g * out_c_group * out_h * out_w + oc * out_h * out_w +
                          oh * out_w + ow;
            float out_value =
                bias_data != nullptr
                    ? (is_channel_bias ? bias_data[g * out_c_group + oc]
                                       : bias_data[out_idx])
                    : 0;
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
               bool is_channel_bias,
               bool fuse_relu,
               bool depthwise,
               int dilation,
               int stride,
               int padding,
               int kernel) {
  // prepare input&output variables
  Scope scope;
  std::string input_var_name("input");
  std::string filter_var_name("filter");
  std::string filter_int_var_name("filter_int");
  std::string bias_var_name("bias");
  std::string output_var_name("output");
  std::string output_ref_var_name("output_ref");
  auto* input = scope.Var(input_var_name)->GetMutable<Tensor>();
  auto* filter = scope.Var(filter_var_name)->GetMutable<Tensor>();
  auto* filter_int = scope.Var(filter_int_var_name)->GetMutable<Tensor>();
  auto* bias = scope.Var(bias_var_name)->GetMutable<Tensor>();
  auto* output = scope.Var(output_var_name)->GetMutable<Tensor>();
  auto* output_ref = scope.Var(output_ref_var_name)->GetMutable<Tensor>();

  // get group size and input&filter shape
  int groups = 1;
  if (depthwise) {  // depthwise convolution ?
    groups = oc = ic;
  }
  std::vector<int64_t> input_shape = {bs, ic, ih, iw};
  std::vector<int64_t> filter_shape = {oc, ic / groups, kernel, kernel};
  std::vector<int64_t> output_shape({bs, oc});
  for (size_t i = 0; i < 2; i++) {
    const int dkernel = dilation * (kernel - 1) + 1;
    int output_size = (input_shape[i + 2] + 2 * padding - dkernel) / stride + 1;
    output_shape.push_back(output_size);
  }
  input->Resize(input_shape);
  filter->Resize(filter_shape);
  filter_int->Resize(filter_shape);
  // initialize input&output data
  FillTensor<int8_t, int8_t>(filter_int, -4, 4);
  float filter_scale = 1. / 16;
  float input_scale = 1. / 8;

  Tensor input_int;
  input_int.Resize(input_shape);
  FillTensor<int8_t, int8_t>(&input_int, -127, 127);
  for (size_t i = 0; i < input->data_size(); i++) {
    input->mutable_data<float>()[i] = input_int.data<int8_t>()[i] * input_scale;
  }
  for (size_t i = 0; i < filter->data_size(); i++) {
    filter->mutable_data<float>()[i] =
        filter_int->data<int8_t>()[i] * filter_scale;
  }

  // initialize op desc
  cpp::OpDesc opdesc;
  opdesc.SetType(depthwise ? "depthwise_conv2d" : "conv2d");
  opdesc.SetInput("Input", {input_var_name});
  opdesc.SetInput("Filter", {filter_var_name});
  opdesc.SetOutput("Output", {output_var_name});
  opdesc.SetAttr("dilations", std::vector<int32_t>({dilation, dilation}));
  opdesc.SetAttr("strides", std::vector<int32_t>({stride, stride}));
  opdesc.SetAttr("paddings",
                 std::vector<int32_t>({padding, padding, padding, padding}));
  opdesc.SetAttr("groups", groups);
  opdesc.SetAttr("fuse_relu", static_cast<bool>(fuse_relu));
  if (has_bias) {
    if (is_channel_bias) {
      bias->Resize({oc});
    } else {
      bias->Resize({output_shape});
    }
    FillTensor<float>(bias);
    opdesc.SetInput("Bias", {bias_var_name});
  }

  auto op_cpu = CreateOp<operators::ConvOpLite>(opdesc, &scope);
  // execute reference implementation and save to output tensor('out')
  conv_ref(op_cpu);
  output_ref->CopyDataFrom(*output);

  // initialize op desc
  cpp::OpDesc opdesc_mlu;
  opdesc_mlu.SetType(depthwise ? "depthwise_conv2d" : "conv2d");
  opdesc_mlu.SetInput("Input", {input_var_name});
  opdesc_mlu.SetInput("Filter", {filter_int_var_name});
  opdesc_mlu.SetOutput("Output", {output_var_name});
  opdesc_mlu.SetAttr("dilations", std::vector<int32_t>({dilation, dilation}));
  opdesc_mlu.SetAttr("strides", std::vector<int32_t>({stride, stride}));
  opdesc_mlu.SetAttr(
      "paddings", std::vector<int32_t>({padding, padding, padding, padding}));
  opdesc_mlu.SetAttr("groups", groups);
  opdesc_mlu.SetAttr("fuse_relu", static_cast<bool>(fuse_relu));

  OpInfo op_info(opdesc_mlu);
  op_info.SetInputScale(filter_int_var_name,
                        std::vector<float>(oc, filter_scale));
  op_info.SetInputScale(input_var_name, {input_scale});

  if (has_bias) {
    if (is_channel_bias) {
      bias->Resize({oc});
    } else {
      bias->Resize({output_shape});
    }
    FillTensor<float>(bias);
    op_info.SetInput("Bias", {bias_var_name});
  }

  for (int i = 0; i < bs; i++) {
    for (int j = 0; j < ic; j++) {
      for (int k = 0; k < ih * iw; k++) {
        input->mutable_data<float>()[i * ic * ih * iw + k * ic + j] =
            input_int.data<int8_t>()[i * ic * ih * iw + j * ih * iw + k] *
            input_scale;
      }
    }
  }

  // create and convert op to MLU model, then run it on MLU
  auto op = CreateOp<operators::ConvOpLite>(op_info, &scope);
  LaunchOp(op, {input_var_name}, {output_var_name});
  // compare results
  auto* output_data = output->mutable_data<float>();
  auto* output_ref_data = output_ref->mutable_data<float>();
  Tensor output_trans;
  output_trans.Resize({output_shape});
  transpose(output_data,
            output_trans.mutable_data<float>(),
            {static_cast<int>(output_shape[0]),
             static_cast<int>(output_shape[2]),
             static_cast<int>(output_shape[3]),
             static_cast<int>(output_shape[1])},
            {0, 3, 1, 2});
  output_data = output_trans.mutable_data<float>();
  for (int i = 0; i < output->dims().production(); i++) {
    VLOG(5) << i;
    EXPECT_NEAR(output_data[i], output_ref_data[i], 1e-5);
  }
}

TEST(MLUBridges, conv) {
#if 1
  for (auto bs : {1}) {
    for (auto ic : {3}) {
      for (auto oc : {32}) {
        for (auto ih : {13}) {
          for (auto iw : {13}) {
            for (auto has_bias : {false}) {
              for (auto is_channel_bias : {true}) {
                for (auto fuse_relu : {false}) {
                  for (auto depthwise : {false}) {
                    for (auto dilation : {1}) {
                      for (auto stride : {1}) {
                        for (auto kernel : {3}) {
                          // std::vector<int> paddings = {kernel / 2};
                          std::vector<int> paddings = {0};
                          if (kernel / 2 != 0) {
                            // paddings.push_back(0);
                          }
                          for (auto padding : paddings) {
                            VLOG(3) << "bs: " << bs << " ic: " << ic
                                    << " oc: " << oc << " ih: " << ih
                                    << " iw: " << iw
                                    << " has_bias: " << has_bias
                                    << " is_channel_bias: " << is_channel_bias
                                    << " fuse_relu: " << fuse_relu
                                    << " depthwise: " << depthwise
                                    << " dilation: " << dilation
                                    << " stride: " << stride
                                    << " padding: " << padding
                                    << " kernel: " << kernel;
                            test_conv(bs,
                                      ic,
                                      oc,
                                      ih,
                                      iw,
                                      has_bias,
                                      is_channel_bias,
                                      fuse_relu,
                                      depthwise,
                                      dilation,
                                      stride,
                                      padding,
                                      kernel);
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
  }
#else
  test_conv(1, 3, 6, 14, 14, false, false, false, true, 2, 1, 1, 3);
  test_conv(1, 3, 6, 14, 14, false, false, false, true, 2, 1, 0, 3);
  test_conv(1, 3, 6, 14, 14, false, false, false, true, 2, 1, 2, 5);
  test_conv(1, 3, 6, 14, 14, false, false, false, true, 2, 1, 0, 5);
#endif
}

TEST(MLUBridges, depthwise_conv2d) {
  test_conv(1, 8, 8, 14, 14, false, false, false, true, 1, 1, 2, 3);
}

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

USE_SUBGRAPH_BRIDGE(conv2d, kMLU)
USE_SUBGRAPH_BRIDGE(depthwise_conv2d, kMLU)
