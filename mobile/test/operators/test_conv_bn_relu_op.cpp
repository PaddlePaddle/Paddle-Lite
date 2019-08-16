/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "../test_helper.h"
#include "../test_include.h"
#include "operators/fusion_conv_bn_relu_op.h"

namespace paddle_mobile {

// Reference convolution from Caffe for checking results.
// accumulate through explicit loops over input, output, and filters.
template <typename Itype, typename Otype, int Kernel, int Pad, int Stride>
int TestConvBnReluOp(int in_channels, int in_height, int in_width,
                     int out_channels, int groups, std::string opname) {
  int kernel_h = Kernel;
  int kernel_w = Kernel;
  int pad_h = Pad;
  int pad_w = Pad;
  int stride_h = Stride;
  int stride_w = Stride;
  int dilation_h = 1;
  int dilation_w = 1;

  int batch_size = 1;
  int input_c = in_channels;
  int input_h = in_height;
  int input_w = in_width;
  int output_c = out_channels;
  framework::DDim input_shape =
      framework::make_ddim({batch_size, input_c, input_h, input_w});
  framework::DDim filter_shape =
      framework::make_ddim({output_c, input_c / groups, kernel_h, kernel_w});
  framework::DDim shape = framework::make_ddim({output_c});

  VariableNameMap inputs;
  VariableNameMap outputs;
  auto scope = std::make_shared<framework::Scope>();
  inputs["Input"] = std::vector<std::string>({"input"});
  inputs["Filter"] = std::vector<std::string>({"filter"});
  outputs["Out"] = std::vector<std::string>({"output"});
  inputs["Mean"] = std::vector<std::string>({"input_mean"});
  inputs["Variance"] = std::vector<std::string>({"input_variance"});
  inputs["Scale"] = std::vector<std::string>({"input_scale"});
  inputs["Bias"] = std::vector<std::string>({"input_bias"});
  auto input_var = scope.get()->Var("input");
  auto input = input_var->template GetMutable<framework::LoDTensor>();
  SetupTensor<Itype>(input, input_shape, -20.0, 20.0);

  auto filter_var = scope.get()->Var("filter");
  auto filter = filter_var->template GetMutable<framework::LoDTensor>();
  SetupTensor<Itype>(filter, filter_shape, -20, 20);

  auto input_mean_var = scope.get()->Var("input_mean");
  auto input_mean = input_mean_var->template GetMutable<framework::LoDTensor>();
  SetupTensor<float>(input_mean, shape, -10.0, 10.0);
  auto vari_var = scope.get()->Var("input_variance");
  auto vari = vari_var->template GetMutable<framework::LoDTensor>();
  SetupTensor<float>(vari, shape, -10.0, 10.0);
  auto scale_var = scope.get()->Var("input_scale");
  auto scale = scale_var->template GetMutable<framework::LoDTensor>();
  SetupTensor<float>(scale, shape, -10.0, 10.0);
  auto input_bias_var = scope.get()->Var("input_bias");
  auto input_bias = input_bias_var->template GetMutable<framework::LoDTensor>();
  SetupTensor<float>(input_bias, shape, -10.0, 10.0);

  auto output_var = scope.get()->Var("output");
  framework::AttributeMap attrs;
  attrs["strides"].Set<vector<int>>(std::vector<int>({stride_h, stride_w}));
  attrs["paddings"].Set<vector<int>>(std::vector<int>({pad_h, pad_w}));
  attrs["dilations"].Set<vector<int>>(
      std::vector<int>({dilation_h, dilation_w}));
  attrs["groups"].Set<int>(groups);
  attrs["epsilon"].Set<float>(1e-6);
  attrs["momentum"].Set<float>(0.f);
  auto *op = new operators::FusionConvBNReluOp<CPU, float>(
      "fusion_conv_bn_relu", inputs, outputs, attrs, scope.get());
  op->InferShape();
  op->Init();
  for (int i = 0; i < 10; ++i) {
    op->Run();
  }
  auto time1 = time();
  for (int i = 0; i < 10; ++i) {
    op->Run();
  }
  auto time2 = time();
  std::ofstream out_file("./out_conv.txt", std::ios::app);
  out_file << opname << " cost :" << time_diff(time1, time2) / 10.0 << "ms"
           << std::endl;
  out_file.close();

  delete op;
  return 0;
}

}  // namespace paddle_mobile

int main(int argc, char *argv[]) {
  // kernel = 3, pad = 1, stride = 2
  paddle_mobile::TestConvBnReluOp<float, float, 3, 1, 2>(3, 48, 48, 16, 1,
                                                         "conv_bn_relu");
  // kernel = 1, pad = 0, stride = 1
  paddle_mobile::TestConvBnReluOp<float, float, 1, 0, 1>(16, 24, 24, 8, 1,
                                                         "depthwise_seperable");
  // kernel = 1, pad = 0, stride = 1
  paddle_mobile::TestConvBnReluOp<float, float, 1, 0, 1>(8, 24, 24, 24, 1,
                                                         "MBConv_3x3_conv1");
  // kernel = 1, pad = 0, stride = 1
  paddle_mobile::TestConvBnReluOp<float, float, 1, 0, 1>(24, 24, 24, 8, 1,
                                                         "MBConv_3x3_pw1");
  // kernel = 1, pad = 0, stride = 1
  paddle_mobile::TestConvBnReluOp<float, float, 1, 0, 1>(8, 24, 24, 24, 1,
                                                         "MBConv_3x3_conv2");
  // kernel = 1, pad = 0, stride = 1
  paddle_mobile::TestConvBnReluOp<float, float, 1, 0, 1>(24, 24, 24, 8, 1,
                                                         "MBConv_3x3_pw2");
  // kernel = 1, pad = 0, stride = 1
  paddle_mobile::TestConvBnReluOp<float, float, 1, 0, 1>(8, 24, 24, 24, 1,
                                                         "MBConv_3x3_conv3");
  // kernel = 1, pad = 0, stride = 1
  paddle_mobile::TestConvBnReluOp<float, float, 1, 0, 1>(24, 12, 12, 16, 1,
                                                         "MBConv_3x3_pw3");
  // kernel = 1, pad = 0, stride = 1
  paddle_mobile::TestConvBnReluOp<float, float, 1, 0, 1>(
      16, 12, 12, 48, 1, "MBConv_5x5_stage1_conv1");
  // kernel = 1, pad = 0, stride = 1
  paddle_mobile::TestConvBnReluOp<float, float, 1, 0, 1>(
      48, 12, 12, 16, 1, "MBConv_5x5_stage1_pw1");
  // kernel = 1, pad = 0, stride = 1
  paddle_mobile::TestConvBnReluOp<float, float, 1, 0, 1>(
      16, 12, 12, 48, 1, "MBConv_5x5_stage1_conv2");
  // kernel = 1, pad = 0, stride = 1
  paddle_mobile::TestConvBnReluOp<float, float, 1, 0, 1>(
      48, 12, 12, 16, 1, "MBConv_5x5_stage1_pw2");
  // kernel = 1, pad = 0, stride = 1
  paddle_mobile::TestConvBnReluOp<float, float, 1, 0, 1>(
      16, 12, 12, 48, 1, "MBConv_5x5_stage1_conv3");
  // kernel = 1, pad = 0, stride = 1
  paddle_mobile::TestConvBnReluOp<float, float, 1, 0, 1>(
      48, 6, 6, 32, 1, "MBConv_5x5_stage1_pw3");
  // kernel = 1, pad = 0, stride = 1
  paddle_mobile::TestConvBnReluOp<float, float, 1, 0, 1>(
      32, 6, 6, 192, 1, "MBConv_5x5_stage2_conv1");
  // kernel = 1, pad = 0, stride = 1
  paddle_mobile::TestConvBnReluOp<float, float, 1, 0, 1>(
      192, 6, 6, 32, 1, "MBConv_5x5_stage2_pw1");
  // kernel = 1, pad = 0, stride = 1
  paddle_mobile::TestConvBnReluOp<float, float, 1, 0, 1>(
      32, 6, 6, 192, 1, "MBConv_5x5_stage2_conv2");
  // kernel = 1, pad = 0, stride = 1
  paddle_mobile::TestConvBnReluOp<float, float, 1, 0, 1>(
      192, 6, 6, 32, 1, "MBConv_5x5_stage2_pw2");
  // kernel = 1, pad = 0, stride = 1
  paddle_mobile::TestConvBnReluOp<float, float, 1, 0, 1>(
      32, 6, 6, 192, 1, "MBConv_5x5_stage2_conv3");
  // kernel = 1, pad = 0, stride = 1
  paddle_mobile::TestConvBnReluOp<float, float, 1, 0, 1>(
      192, 6, 6, 64, 1, "MBConv_5x5_stage2_pw3");

  return 0;
}
