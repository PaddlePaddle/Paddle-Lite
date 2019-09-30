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

#include <iostream>
#include "../test_include.h"
#include "operators/math/pooling.h"
#include "operators/pool_op.h"

namespace paddle_mobile {

namespace math = operators::math;

template <int PoolType, int Kernel, int Pad, int Stride>
int TestPoolOp(int in_channels, int in_height, int in_width) {
  int kernel_h = Kernel;
  int kernel_w = Kernel;
  int pad_h = Pad;
  int pad_w = Pad;
  int stride_h = Stride;
  int stride_w = Stride;
  std::string pooling_type = (PoolType == 0 ? "max" : "avg");

  int batch_size = 1;
  int input_c = in_channels;
  int input_h = in_height;
  int input_w = in_width;

  framework::DDim input_shape =
      framework::make_ddim({batch_size, input_c, input_h, input_w});

  VariableNameMap inputs;
  VariableNameMap outputs;
  auto scope = std::make_shared<framework::Scope>();
  inputs["X"] = std::vector<std::string>({"input"});
  outputs["Out"] = std::vector<std::string>({"output"});

  auto input_var = scope.get()->Var("input");
  auto input = input_var->template GetMutable<framework::LoDTensor>();
  SetupTensor<float>(input, input_shape, -127, 127);

  //  for (int i = 0; i < input->numel(); ++i) {
  //    DLOG << "input[" << i << "] = " << input->data<float>()[i];
  //  }

  auto output_var = scope.get()->Var("output");
  framework::AttributeMap attrs;
  attrs["pooling_type"].Set<std::string>(pooling_type);
  attrs["ksize"].Set<vector<int>>(std::vector<int>({kernel_h, kernel_w}));
  attrs["strides"].Set<vector<int>>(std::vector<int>({stride_h, stride_w}));
  attrs["paddings"].Set<vector<int>>(std::vector<int>({pad_h, pad_w}));
  attrs["ceil_mode"].Set<bool>(true);
  //  attrs["ceil_mode"].Set<bool>(false);
  attrs["global_pooling"].Set<bool>(false);
  attrs["exclusive"].Set<bool>(true);

  auto *op = new operators::PoolOp<CPU, float>("pool2d", inputs, outputs, attrs,
                                               scope.get());
  op->InferShape();
  op->Init();
  op->Run();

  auto output = output_var->template Get<framework::LoDTensor>();
  framework::Tensor output_cmp;
  output_cmp.mutable_data<float>(output->dims());

  if (pooling_type == "avg") {
    math::Pooling<AVG>()(*input, std::vector<int>{kernel_h, kernel_w},
                         std::vector<int>{stride_h, stride_w},
                         std::vector<int>{pad_h, pad_w}, &output_cmp);
  } else {
    math::Pooling<MAX>()(*input, std::vector<int>{kernel_h, kernel_w},
                         std::vector<int>{stride_h, stride_w},
                         std::vector<int>{pad_h, pad_w}, &output_cmp);
  }

  // compare results
  const float *output_data = output->data<float>();
  float *output_cmp_data = output_cmp.data<float>();
  for (int i = 0; i < output->numel(); ++i) {
    float gap = output_data[i] - output_cmp_data[i];
    //    PADDLE_MOBILE_ENFORCE(output_data[i] == output_cmp_data[i],
    //                          "output[%d] = %d, output_cmp[%d] = %d", i,
    //                          output_data[i], i, output_cmp_data[i]);
    if (gap > 1e-5 && std::abs(gap / (output_data[i] + 1e-5)) > 1e-3) {
      LOG(kLOG_INFO) << "output_data[" << i << "] = " << output_data[i]
                     << ", output_cmp_data[" << i
                     << "] = " << output_cmp_data[i];
      exit(1);
    }
  }
  delete op;
  return 0;
}
}  // namespace paddle_mobile

int Test(const int in_channels, const int in_height, const int in_width) {
  LOG(paddle_mobile::kLOG_INFO)
      << "float, pooling_type=max, kernel=3, pad=0, stride=1";
  paddle_mobile::TestPoolOp<0, 3, 0, 1>(in_channels, in_height, in_width);
  LOG(paddle_mobile::kLOG_INFO)
      << "float, pooling_type=max, kernel=3, pad=1, stride=1";
  paddle_mobile::TestPoolOp<0, 3, 1, 1>(in_channels, in_height, in_width);
  LOG(paddle_mobile::kLOG_INFO)
      << "float, pooling_type=max, kernel=3, pad=2, stride=1";
  paddle_mobile::TestPoolOp<0, 3, 2, 1>(in_channels, in_height, in_width);
  LOG(paddle_mobile::kLOG_INFO)
      << "float, pooling_type=max, kernel=3, pad=5, stride=1";
  paddle_mobile::TestPoolOp<0, 3, 5, 1>(in_channels, in_height, in_width);

  LOG(paddle_mobile::kLOG_INFO)
      << "float, pooling_type=avg, kernel=3, pad=0, stride=1";
  paddle_mobile::TestPoolOp<1, 3, 0, 1>(in_channels, in_height, in_width);
  LOG(paddle_mobile::kLOG_INFO)
      << "float, pooling_type=avg, kernel=3, pad=1, stride=1";
  paddle_mobile::TestPoolOp<1, 3, 1, 1>(in_channels, in_height, in_width);
  LOG(paddle_mobile::kLOG_INFO)
      << "float, pooling_type=avg, kernel=3, pad=2, stride=1";
  paddle_mobile::TestPoolOp<1, 3, 2, 1>(in_channels, in_height, in_width);
  LOG(paddle_mobile::kLOG_INFO)
      << "float, pooling_type=avg, kernel=3, pad=5, stride=1";
  paddle_mobile::TestPoolOp<1, 3, 5, 1>(in_channels, in_height, in_width);

  LOG(paddle_mobile::kLOG_INFO)
      << "float, pooling_type=max, kernel=3, pad=0, stride=2";
  paddle_mobile::TestPoolOp<0, 3, 0, 2>(in_channels, in_height, in_width);
  LOG(paddle_mobile::kLOG_INFO)
      << "float, pooling_type=max, kernel=3, pad=1, stride=2";
  paddle_mobile::TestPoolOp<0, 3, 1, 2>(in_channels, in_height, in_width);
  LOG(paddle_mobile::kLOG_INFO)
      << "float, pooling_type=max, kernel=3, pad=2, stride=2";
  paddle_mobile::TestPoolOp<0, 3, 2, 2>(in_channels, in_height, in_width);
  LOG(paddle_mobile::kLOG_INFO)
      << "float, pooling_type=max, kernel=3, pad=5, stride=2";
  paddle_mobile::TestPoolOp<0, 3, 5, 2>(in_channels, in_height, in_width);

  LOG(paddle_mobile::kLOG_INFO)
      << "float, pooling_type=avg, kernel=3, pad=0, stride=2";
  paddle_mobile::TestPoolOp<1, 3, 0, 2>(in_channels, in_height, in_width);
  LOG(paddle_mobile::kLOG_INFO)
      << "float, pooling_type=avg, kernel=3, pad=1, stride=2";
  paddle_mobile::TestPoolOp<1, 3, 1, 2>(in_channels, in_height, in_width);
  LOG(paddle_mobile::kLOG_INFO)
      << "float, pooling_type=avg, kernel=3, pad=2, stride=2";
  paddle_mobile::TestPoolOp<1, 3, 2, 2>(in_channels, in_height, in_width);
  LOG(paddle_mobile::kLOG_INFO)
      << "float, pooling_type=avg, kernel=3, pad=5, stride=2";
  paddle_mobile::TestPoolOp<1, 3, 5, 2>(in_channels, in_height, in_width);

  LOG(paddle_mobile::kLOG_INFO)
      << "float, pooling_type=max, kernel=2, pad=0, stride=1";
  paddle_mobile::TestPoolOp<0, 2, 0, 1>(in_channels, in_height, in_width);
  LOG(paddle_mobile::kLOG_INFO)
      << "float, pooling_type=max, kernel=2, pad=1, stride=1";
  paddle_mobile::TestPoolOp<0, 2, 1, 1>(in_channels, in_height, in_width);
  LOG(paddle_mobile::kLOG_INFO)
      << "float, pooling_type=max, kernel=2, pad=2, stride=1";
  paddle_mobile::TestPoolOp<0, 2, 2, 1>(in_channels, in_height, in_width);
  LOG(paddle_mobile::kLOG_INFO)
      << "float, pooling_type=max, kernel=2, pad=5, stride=1";
  paddle_mobile::TestPoolOp<0, 2, 5, 1>(in_channels, in_height, in_width);

  LOG(paddle_mobile::kLOG_INFO)
      << "float, pooling_type=avg, kernel=2, pad=0, stride=1";
  paddle_mobile::TestPoolOp<1, 2, 0, 1>(in_channels, in_height, in_width);
  LOG(paddle_mobile::kLOG_INFO)
      << "float, pooling_type=avg, kernel=2, pad=1, stride=1";
  paddle_mobile::TestPoolOp<1, 2, 1, 1>(in_channels, in_height, in_width);
  LOG(paddle_mobile::kLOG_INFO)
      << "float, pooling_type=avg, kernel=2, pad=2, stride=1";
  paddle_mobile::TestPoolOp<1, 2, 2, 1>(in_channels, in_height, in_width);
  LOG(paddle_mobile::kLOG_INFO)
      << "float, pooling_type=avg, kernel=2, pad=5, stride=1";
  paddle_mobile::TestPoolOp<1, 2, 5, 1>(in_channels, in_height, in_width);

  LOG(paddle_mobile::kLOG_INFO)
      << "float, pooling_type=max, kernel=2, pad=0, stride=2";
  paddle_mobile::TestPoolOp<0, 2, 0, 2>(in_channels, in_height, in_width);
  LOG(paddle_mobile::kLOG_INFO)
      << "float, pooling_type=max, kernel=2, pad=1, stride=2";
  paddle_mobile::TestPoolOp<0, 2, 1, 2>(in_channels, in_height, in_width);
  LOG(paddle_mobile::kLOG_INFO)
      << "float, pooling_type=max, kernel=2, pad=2, stride=2";
  paddle_mobile::TestPoolOp<0, 2, 2, 2>(in_channels, in_height, in_width);
  LOG(paddle_mobile::kLOG_INFO)
      << "float, pooling_type=max, kernel=2, pad=5, stride=2";
  paddle_mobile::TestPoolOp<0, 2, 5, 2>(in_channels, in_height, in_width);

  LOG(paddle_mobile::kLOG_INFO)
      << "float, pooling_type=avg, kernel=2, pad=0, stride=2";
  paddle_mobile::TestPoolOp<1, 2, 0, 2>(in_channels, in_height, in_width);
  LOG(paddle_mobile::kLOG_INFO)
      << "float, pooling_type=avg, kernel=2, pad=1, stride=2";
  paddle_mobile::TestPoolOp<1, 2, 1, 2>(in_channels, in_height, in_width);
  LOG(paddle_mobile::kLOG_INFO)
      << "float, pooling_type=avg, kernel=2, pad=2, stride=2";
  paddle_mobile::TestPoolOp<1, 2, 2, 2>(in_channels, in_height, in_width);
  LOG(paddle_mobile::kLOG_INFO)
      << "float, pooling_type=avg, kernel=2, pad=5, stride=2";
  paddle_mobile::TestPoolOp<1, 2, 5, 2>(in_channels, in_height, in_width);
}

int main(int argc, char *argv[]) {
  //  if (argc < 4) {
  //    LOG(paddle_mobile::kLOG_INFO)
  //        << "Usage:\n"
  //        << "  ./test-pool-op in_channels in_height in_width \n"
  //        << "  params:\n"
  //        << "   -in_channels: int, input image's channels\n"
  //        << "   -in_height: int, input image's height\n"
  //        << "   -in_width: int, input image's width\n";
  //    return 1;
  //  }
  //  int in_channels = atoi(argv[1]);
  //  int in_height = atoi(argv[2]);
  //  int in_width = atoi(argv[3]);
  Test(1, 10, 10);
  Test(1, 50, 50);
  Test(32, 10, 10);
  Test(32, 50, 50);
}
