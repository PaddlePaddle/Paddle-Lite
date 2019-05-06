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

#ifdef PADDLE_MOBILE_CL
#include <iostream>
#include "../test_helper.h"
#include "../test_include.h"
#include "common/common.h"
#include "framework/cl/cl_helper.h"
#include "framework/cl/cl_image.h"
#include "operators/conv_op.h"
#include "operators/kernel/cl/cl-kernel-func/conv_func.h"

namespace paddle_mobile {

template <typename Itype, typename Otype, int Kernel, int Pad, int Stride>
int TestConvOp(int in_channels, int in_height, int in_width, int out_channels,
               int groups) {
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

  //    std::cerr << " init " << std::endl;
  VariableNameMap inputs;
  VariableNameMap outputs;
  auto scope = std::make_shared<framework::Scope>();
  inputs["Input"] = std::vector<std::string>({"input"});
  inputs["Filter"] = std::vector<std::string>({"filter"});
  outputs["Output"] = std::vector<std::string>({"output"});
  cl_context context = scope->GetCLScpoe()->Context();
  cl_command_queue command_queue = scope->GetCLScpoe()->CommandQueue();

  //    std::cerr << " input " << std::endl;
  auto input_var = scope.get()->Var("input");
  auto input = input_var->template GetMutable<framework::CLImage>();
  const int in_numel = framework::product(input_shape);
  float *in_data = new float[in_numel];
  for (int i = 0; i < in_numel; ++i) {
    in_data[i] = (i % 36 / 6) + 1;
  }
  input->SetTensorData(in_data, input_shape);
  input->InitNormalCLImage(context, command_queue);
  DLOG << "input image \n" << *input;

  //    std::cerr << " filter " << std::endl;
  auto filter_var = scope.get()->Var("filter");
  auto filter = filter_var->template GetMutable<framework::CLImage>();
  const int filter_numel = product(filter_shape);
  float *filter_data = new float[filter_numel];
  for (int i = 0; i < filter_numel; ++i) {
    filter_data[i] = i % 9;
  }
  filter->SetTensorData(filter_data, filter_shape);

  //    std::cerr << " attrs " << std::endl;
  framework::AttributeMap attrs;
  attrs["strides"].Set<vector<int>>(std::vector<int>({stride_h, stride_w}));
  attrs["paddings"].Set<vector<int>>(std::vector<int>({pad_h, pad_w}));
  attrs["dilations"].Set<vector<int>>(
      std::vector<int>({dilation_h, dilation_w}));
  attrs["groups"].Set<int>(groups);

  std::cerr << " output " << std::endl;
  auto output_var = scope.get()->Var("output");
  auto output = output_var->template GetMutable<framework::CLImage>();

  auto *op = new operators::ConvOp<GPU_CL, float>("conv2d", inputs, outputs,
                                                  attrs, scope.get());

  op->InferShape();

  framework::DDim ddim = output->dims();

  DLOG << "output dims = " << ddim;
  output->InitEmptyImage(context, command_queue, ddim);

  //    std::cerr << " op->init " << std::endl;
  op->Init();

  auto time1 = time();
  op->Run();
  auto time2 = time();
  std::cerr << "time cost : " << time_diff(time1, time2) << std::endl;

  delete op;
  return 0;
}

}  // namespace paddle_mobile

int TestAll(const int in_channels, const int in_height, const int in_width,
            const int out_channels, const int groups) {
  std::cerr << "in_channels=" << in_channels << ", in_height=" << in_height
            << ", in_width=" << in_width << ", out_channels=" << out_channels
            << ", groups=" << groups << std::endl;
  std::cerr << "float, kernel=3, pad=1, stride=1" << std::endl;
  paddle_mobile::TestConvOp<float, float, 3, 1, 1>(
      in_channels, in_height, in_width, out_channels, groups);

  return 0;
}

int main() {
  TestAll(4, 6, 6, 4, 1);
  //  TestAll(6, 32, 32, 24, 1);
  //    TestAll(12, 32, 32, 24, 1);
  //    TestAll(24, 32, 32, 24, 1);
  //    TestAll(36, 32, 32, 24, 1);
  //    TestAll(48, 32, 32, 24, 1);
  //    TestAll(60, 32, 32, 24, 1);
  //    TestAll(72, 32, 32, 24, 1);
  //    TestAll(116, 32, 32, 24, 1);
  //    TestAll(232, 32, 32, 24, 1);
  //    TestAll(464, 32, 32, 24, 1);
  //
  //    TestAll(6, 64, 64, 24, 1);
  //    TestAll(12, 64, 64, 24, 1);
  //    TestAll(24, 64, 64, 24, 1);
  //    TestAll(36, 64, 64, 24, 1);
  //    TestAll(48, 64, 64, 24, 1);
  //    TestAll(60, 64, 64, 24, 1);
  //    TestAll(72, 64, 64, 24, 1);
  //    TestAll(116, 64, 64, 24, 1);
  //    TestAll(232, 64, 64, 24, 1);
  //    TestAll(464, 64, 64, 24, 1);
  //
  //    TestAll(6, 128, 128, 24, 1);
  //    TestAll(12, 128, 128, 24, 1);
  //    TestAll(24, 128, 128, 24, 1);
  //    TestAll(36, 128, 128, 24, 1);
  //    TestAll(48, 128, 128, 24, 1);
  //    TestAll(60, 128, 128, 24, 1);
  //    TestAll(72, 128, 128, 24, 1);
  //    TestAll(116, 128, 128, 24, 1);
  //    TestAll(232, 128, 128, 24, 1);
  //    TestAll(464, 128, 128, 24, 1);
  //
  //
  //    TestAll(6, 32, 32, 6, 1);
  //    TestAll(12, 32, 32, 12, 1);
  //    TestAll(24, 32, 32, 24, 1);
  //    TestAll(36, 32, 32, 36, 1);
  //    TestAll(48, 32, 32, 48, 1);
  //    TestAll(60, 32, 32, 60, 1);
  //    TestAll(72, 32, 32, 72, 1);
  //    TestAll(116, 32, 32, 116, 1);
  //    TestAll(232, 32, 32, 232, 1);
  //    TestAll(464, 32, 32, 464, 1);
  //
  //    TestAll(6, 64, 64, 6, 1);
  //    TestAll(12, 64, 64, 12, 1);
  //    TestAll(24, 64, 64, 24, 1);
  //    TestAll(36, 64, 64, 36, 1);
  //    TestAll(48, 64, 64, 48, 1);
  //    TestAll(60, 64, 64, 60, 1);
  //    TestAll(72, 64, 64, 72, 1);
  //    TestAll(116, 64, 64, 116, 1);
  //    TestAll(232, 64, 64, 232, 1);
  //    TestAll(464, 64, 64, 464, 1);
  //
  //    TestAll(6, 128, 128, 6, 1);
  //    TestAll(12, 128, 128, 12, 1);
  //    TestAll(24, 128, 128, 24, 1);
  //    TestAll(36, 128, 128, 36, 1);
  //    TestAll(48, 128, 128, 48, 1);
  //    TestAll(60, 128, 128, 60, 1);
  //    TestAll(72, 128, 128, 72, 1);
  //    TestAll(116, 128, 128, 116, 1);
  //    TestAll(232, 128, 128, 232, 1);
  //    TestAll(464, 128, 128, 464, 1);
  return 0;
}
#endif
