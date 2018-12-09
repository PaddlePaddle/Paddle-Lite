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

static int PoolOutputSize(int input_size, int filter_size, int padding,
                          int stride, bool ceil_mode) {
  int output_size;
  if (!ceil_mode) {
    output_size = (input_size - filter_size + 2 * padding) / stride + 1;
  } else {
    output_size =
        (input_size - filter_size + 2 * padding + stride - 1) / stride + 1;
  }
  return output_size;
}

template <typename T, int CeilMode, int PoolType, int Kernel, int Pad,
          int Stride>
int TestPoolOp(int in_channels, int in_height, int in_width) {
  int kernel_h = Kernel;
  int kernel_w = Kernel;
  int pad_h = Pad;
  int pad_w = Pad;
  int stride_h = Stride;
  int stride_w = Stride;
  bool ceil_mode = CeilMode != 0;
  std::string pooling_type = (PoolType == 0 ? "max" : "avg");

  int batch_size = 1;
  int input_c = in_channels;
  int input_h = in_height;
  int input_w = in_width;

  framework::DDim input_shape =
      framework::make_ddim({batch_size, input_c, input_h, input_w});

  std::vector<int64_t> output_shape_v({batch_size, input_c});
  output_shape_v.push_back(
      PoolOutputSize(input_h, kernel_h, pad_h, stride_h, ceil_mode));
  output_shape_v.push_back(
      PoolOutputSize(input_w, kernel_w, pad_w, stride_w, ceil_mode));

  framework::DDim output_shape = framework::make_ddim(output_shape_v);

  VariableNameMap inputs;
  VariableNameMap outputs;
  auto scope = std::make_shared<framework::Scope>();
  inputs["X"] = std::vector<std::string>({"input"});
  outputs["Out"] = std::vector<std::string>({"output"});

  auto input_var = scope.get()->Var("input");
  auto input = input_var->template GetMutable<framework::LoDTensor>();
  SetupTensor<T>(input, input_shape, -127, 127);

  auto output_var = scope.get()->Var("output");
  framework::AttributeMap attrs;
  attrs["pooling_type"].SetString(pooling_type);
  attrs["ksize"].Set<vector<int>>(std::vector<int>({kernel_h, kernel_w}));
  attrs["strides"].Set<vector<int>>(std::vector<int>({stride_h, stride_w}));
  attrs["paddings"].Set<vector<int>>(std::vector<int>({pad_h, pad_w}));
  attrs["ceil_mode"].Set<bool>(false);
  attrs["global_pooling"].Set<bool>(false);

  auto *op = new operators::PoolOp<CPU, float>("pool2d", inputs, outputs, attrs,
                                               scope);
  op->InferShape();
  op->Init();
  op->Run();

  framework::Tensor output_cmp;
  output_cmp.mutable_data<T>(output_shape);

  if (pooling_type == "avg") {
    math::Pooling<Avg>()(*input, std::vector<int>{kernel_h, kernel_w},
                         std::vector<int>{stride_h, stride_w},
                         std::vector<int>{pad_h, pad_w}, &output_cmp);
  } else {
    math::Pooling<Max>()(*input, std::vector<int>{kernel_h, kernel_w},
                         std::vector<int>{stride_h, stride_w},
                         std::vector<int>{pad_h, pad_w}, &output_cmp);
  }

  // compare results
  auto output = output_var->template Get<framework::LoDTensor>();
  const T *output_data = output->data<T>();
  T *output_cmp_data = output_cmp.data<T>();
  for (int i = 0; i < output->numel(); ++i) {
    PADDLE_MOBILE_ENFORCE(output_data[i] == output_cmp_data[i],
                          "output[%d] = %d, output_cmp[%d] = %d", i,
                          output_data[i], i, output_cmp_data[i]);
  }
  delete op;
  return 0;
}
}  // namespace paddle_mobile

int main(int argc, char *argv[]) {
  if (argc < 4) {
    LOG(paddle_mobile::kLOG_INFO)
        << "Usage:\n"
        << "  ./test-pool-op in_channels in_height in_width \n"
        << "  params:\n"
        << "   -in_channels: int, input image's channels\n"
        << "   -in_height: int, input image's height\n"
        << "   -in_width: int, input image's width\n";
    return 1;
  }
  int in_channels = atoi(argv[1]);
  int in_height = atoi(argv[2]);
  int in_width = atoi(argv[3]);
  // kernel = 3, pad = 1, stride = 1
  LOG(paddle_mobile::kLOG_INFO)
      << "float, ceil_mode=false, pooling_type=max, kernel=3, pad=1, stride=1";
  paddle_mobile::TestPoolOp<float, 0, 0, 3, 1, 1>(in_channels, in_height,
                                                  in_width);
  // kernel = 3, pad = 0, stride = 2
  LOG(paddle_mobile::kLOG_INFO)
      << "float, ceil_mode=false, pooling_type=max, kernel=3, pad=0, stride=2";
  paddle_mobile::TestPoolOp<float, 0, 0, 3, 0, 2>(in_channels, in_height,
                                                  in_width);
  // kernel = 5, pad = 0, stride = 1
  LOG(paddle_mobile::kLOG_INFO)
      << "float, ceil_mode=false, pooling_type=avg, kernel=5, pad=0, stride=1";
  paddle_mobile::TestPoolOp<float, 0, 1, 5, 0, 1>(in_channels, in_height,
                                                  in_width);
  // kernel = 5, pad = 0, stride = 2
  LOG(paddle_mobile::kLOG_INFO)
      << "float, ceil_mode=false, pooling_type=avg, kernel=5, pad=0, stride=1";
  paddle_mobile::TestPoolOp<float, 0, 1, 5, 0, 2>(in_channels, in_height,
                                                  in_width);
  // kernel = 7, pad = 0, stride = 1
  LOG(paddle_mobile::kLOG_INFO)
      << "float, ceil_mode=false, pooling_type=avg, kernel=7, pad=0, stride=1";
  paddle_mobile::TestPoolOp<float, 0, 1, 7, 0, 1>(in_channels, in_height,
                                                  in_width);
  // kernel = 7, pad = 0, stride = 4
  LOG(paddle_mobile::kLOG_INFO)
      << "float, ceil_mode=false, pooling_type=avg, kernel=7, pad=0, stride=4";
  paddle_mobile::TestPoolOp<float, 0, 1, 7, 0, 4>(in_channels, in_height,
                                                  in_width);
}
