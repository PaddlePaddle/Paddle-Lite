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
#include "operators/fusion_dwconv_bn_relu_op.h"

namespace paddle_mobile {

template <typename Itype, typename Otype, int Kernel, int Pad, int Stride>
int TestDWConvAddBnReluOp(int in_channels, int in_height, int in_width,
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
  inputs["Mean"] = std::vector<std::string>({"mean"});
  inputs["Variance"] = std::vector<std::string>({"variance"});
  inputs["Scale"] = std::vector<std::string>({"scale"});
  inputs["Bias"] = std::vector<std::string>({"bias"});
  outputs["Out"] = std::vector<std::string>({"output"});

  auto input_var = scope.get()->Var("input");
  auto input = input_var->template GetMutable<framework::LoDTensor>();
  SetupTensor<Itype>(input, input_shape, -20.0, 20.0);

  auto filter_var = scope.get()->Var("filter");
  auto filter = filter_var->template GetMutable<framework::LoDTensor>();
  SetupTensor<Itype>(filter, filter_shape, -20, 20);

  auto mean_var = scope.get()->Var("mean");
  auto mean = mean_var->template GetMutable<framework::LoDTensor>();
  SetupTensor<float>(mean, shape, -10.0, 10.0);

  auto vari_var = scope.get()->Var("variance");
  auto vari = vari_var->template GetMutable<framework::LoDTensor>();
  SetupTensor<float>(vari, shape, -10.0, 10.0);

  auto scale_var = scope.get()->Var("scale");
  auto scale = scale_var->template GetMutable<framework::LoDTensor>();
  SetupTensor<float>(scale, shape, -10.0, 10.0);

  auto bias_var = scope.get()->Var("bias");
  auto bias = bias_var->template GetMutable<framework::LoDTensor>();
  SetupTensor<float>(bias, shape, -10.0, 10.0);

  auto output_var = scope.get()->Var("output");
  framework::AttributeMap attrs;
  attrs["strides"].Set<vector<int>>(std::vector<int>({stride_h, stride_w}));
  attrs["paddings"].Set<vector<int>>(std::vector<int>({pad_h, pad_w}));
  attrs["dilations"].Set<vector<int>>(
      std::vector<int>({dilation_h, dilation_w}));
  attrs["groups"].Set<int>(groups);
  attrs["epsilon"].Set<float>(1e-6);
  attrs["momentum"].Set<float>(0.f);

  auto *op = new operators::FusionDWConvBNReluOp<CPU, float>(
      "fusion_dwconv_bn_relu", inputs, outputs, attrs, scope.get());
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
  std::ofstream out_file("./out_dwconv.txt", std::ios::app);
  out_file << opname << " cost :" << time_diff(time1, time2) / 10.0 << "ms"
           << std::endl;
  out_file.close();

  delete op;
  return 0;
}

}  // namespace paddle_mobile

int main(int argc, char *argv[]) {
  // kernel = 3, pad = 1, stride = 1
  paddle_mobile::TestDWConvAddBnReluOp<float, float, 3, 1, 1>(
      16, 24, 24, 16, 16, "depthwise_seperable");
  // kernel = 3, pad = 1, stride = 1
  paddle_mobile::TestDWConvAddBnReluOp<float, float, 3, 1, 1>(
      24, 24, 24, 24, 24, "MBConv_3x3_dw1");
  // kernel = 3, pad = 1, stride = 1
  paddle_mobile::TestDWConvAddBnReluOp<float, float, 3, 1, 1>(
      24, 24, 24, 24, 24, "MBConv_3x3_dw2");
  // kernel = 3, pad = 1, stride = 2
  paddle_mobile::TestDWConvAddBnReluOp<float, float, 3, 1, 2>(
      24, 24, 24, 24, 24, "MBConv_3x3_dw3");
  // kernel = 5, pad = 2, stride = 1
  paddle_mobile::TestDWConvAddBnReluOp<float, float, 5, 2, 1>(
      48, 12, 12, 48, 48, "MBConv_5x5_stage1_dw1");
  // kernel = 5, pad = 2, stride = 1
  paddle_mobile::TestDWConvAddBnReluOp<float, float, 5, 2, 1>(
      48, 12, 12, 48, 48, "MBConv_5x5_stage1_dw2");
  // kernel = 5, pad = 2, stride = 2
  paddle_mobile::TestDWConvAddBnReluOp<float, float, 5, 2, 2>(
      48, 12, 12, 48, 48, "MBConv_5x5_stage1_dw3");
  // kernel = 5, pad = 2, stride = 1
  paddle_mobile::TestDWConvAddBnReluOp<float, float, 5, 2, 1>(
      192, 6, 6, 192, 192, "MBConv_5x5_stage2_dw1");
  // kernel = 5, pad = 2, stride = 1
  paddle_mobile::TestDWConvAddBnReluOp<float, float, 5, 2, 1>(
      192, 6, 6, 192, 192, "MBConv_5x5_stage2_dw2");
  // kernel = 5, pad = 2, stride = 1
  paddle_mobile::TestDWConvAddBnReluOp<float, float, 5, 2, 1>(
      192, 6, 6, 192, 192, "MBConv_5x5_stage2_dw3");

  return 0;
}
