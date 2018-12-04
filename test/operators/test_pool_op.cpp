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
#include "operators/kernel/central-arm-func/pool_arm_func.h"
#include "operators/pool_op.h"

namespace paddle_mobile {
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

template <typename T>
static void PoolAvgPad0(std::vector<int> ksize, std::vector<int> strides,
                        const framework::Tensor *input,
                        framework::Tensor *out) {
  const int32_t batch_size = input->dims()[0];
  const int32_t input_c = input->dims()[1];
  const int32_t input_h = input->dims()[2];
  const int32_t input_w = input->dims()[3];
  const int32_t out_c = out->dims()[1];
  const int32_t out_h = out->dims()[2];
  const int32_t out_w = out->dims()[3];
  const int32_t kernel_h = ksize[0];
  const int32_t kernel_w = ksize[1];
  const int32_t stride_h = strides[0];
  const int32_t stride_w = strides[1];
  const int32_t inputdata_channel_stride = input_h * input_w;
  const int32_t input_batch_stride = input_c * inputdata_channel_stride;
  const int32_t outputdata_channel_stride = out_h * out_w;
  const int32_t output_batch_stride = out_c * outputdata_channel_stride;
  T *out_data = out->mutable_data<T>();
  const T *input_data = input->data<T>();
  const T **rows = new const T *[kernel_h];
  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < out_c; ++j) {
      const T *img_in = input_data + j * inputdata_channel_stride;
      T *img_out = out_data + j * outputdata_channel_stride;
      for (int k = 0; k < out_h; ++k) {
        for (int m = 0; m < kernel_h; ++m) {
          rows[m] = img_in + (stride_h * k + m) * input_w;
        }
        int32_t left = out_w;
        while (left > 0) {
          float tmp = 0;
          for (int m = 0; m < kernel_h; ++m) {
            for (int l = 0; l < kernel_w; ++l) {
              tmp += rows[m][l];
            }
          }
          if (typeid(T) == typeid(int8_t)) {
            tmp = tmp / (kernel_h * kernel_w);
            if (tmp < -127) {
              *img_out = -127;
            } else if (tmp > 127) {
              *img_out = 127;
            } else {
              *img_out = static_cast<T>(std::round(tmp));
            }
          } else {
            *img_out = static_cast<T>(tmp / (kernel_h * kernel_w));
          }
          for (int m = 0; m < kernel_h; ++m) {
            rows[m] += stride_w;
          }
          img_out++;
          left--;
        }
      }
    }
    input_data += input_batch_stride;
    out_data += output_batch_stride;
  }
  delete[] rows;
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
  if (pooling_type == "avg" && pad_h == 0 && pad_h == pad_w) {
    PoolAvgPad0<T>(std::vector<int>{kernel_h, kernel_w},
                   std::vector<int>{stride_h, stride_w}, input, &output_cmp);
  } else {
    if (typeid(T) == typeid(int8_t)) {
      operators::PoolBasic<int8_t, int32_t>(
          pooling_type, std::vector<int>{kernel_h, kernel_w},
          std::vector<int>{stride_h, stride_w}, std::vector<int>{pad_h, pad_w},
          input, &output_cmp);
    } else {
      operators::PoolBasic<float, float>(
          pooling_type, std::vector<int>{kernel_h, kernel_w},
          std::vector<int>{stride_h, stride_w}, std::vector<int>{pad_h, pad_w},
          input, &output_cmp);
    }
  }

  // compare results
  int eq = 0;
  int neq = 0;
  auto output = output_var->template Get<framework::LoDTensor>();
  const T *output_data = output->data<T>();
  T *output_cmp_data = output_cmp.data<T>();
  for (int i = 0; i < output->numel(); ++i) {
    PADDLE_MOBILE_ENFORCE(output_data[i] == output_cmp_data[i],
                          "The execution of test_pool_op is failed!");
    if (output_data[i] == output_cmp_data[i]) {
      ++eq;
    } else {
      ++neq;
    }
  }
  std::cout << "eq = " << eq << ", neq = " << neq << std::endl;
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
#if __ARM_NEON
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
#endif
  // kernel = 3, pad = 0, stride = 1
  LOG(paddle_mobile::kLOG_INFO)
      << "int8_t, ceil_mode=false, pooling_type=max, kernel=3, pad=0, stride=1";
  paddle_mobile::TestPoolOp<int8_t, 0, 0, 3, 0, 1>(in_channels, in_height,
                                                   in_width);
  // kernel = 3, pad = 1, stride = 1
  LOG(paddle_mobile::kLOG_INFO)
      << "int8_t, ceil_mode=false, pooling_type=max, kernel=3, pad=1, stride=1";
  paddle_mobile::TestPoolOp<int8_t, 0, 0, 3, 1, 1>(in_channels, in_height,
                                                   in_width);
  // kernel = 3, pad = 2, stride = 1
  LOG(paddle_mobile::kLOG_INFO)
      << "int8_t, ceil_mode=false, pooling_type=max, kernel=3, pad=2, stride=1";
  paddle_mobile::TestPoolOp<int8_t, 0, 0, 3, 2, 1>(in_channels, in_height,
                                                   in_width);
  // kernel = 3, pad = 0, stride = 2
  LOG(paddle_mobile::kLOG_INFO)
      << "int8_t, ceil_mode=false, pooling_type=max, kernel=3, pad=0, stride=2";
  paddle_mobile::TestPoolOp<int8_t, 0, 0, 3, 0, 2>(in_channels, in_height,
                                                   in_width);
  // kernel = 3, pad = 1, stride = 2
  LOG(paddle_mobile::kLOG_INFO)
      << "int8_t, ceil_mode=false, pooling_type=max, kernel=3, pad=1, stride=2";
  paddle_mobile::TestPoolOp<int8_t, 0, 0, 3, 1, 2>(in_channels, in_height,
                                                   in_width);
  // kernel = 3, pad = 0, stride = 2
  LOG(paddle_mobile::kLOG_INFO)
      << "int8_t, ceil_mode=false, pooling_type=max, kernel=3, pad=2, stride=2";
  paddle_mobile::TestPoolOp<int8_t, 0, 0, 3, 2, 2>(in_channels, in_height,
                                                   in_width);
  // kernel = 3, pad = 3, stride = 3
  LOG(paddle_mobile::kLOG_INFO)
      << "int8_t, ceil_mode=false, pooling_type=max, kernel=3, pad=3, stride=3";
  paddle_mobile::TestPoolOp<int8_t, 0, 0, 3, 3, 3>(in_channels, in_height,
                                                   in_width);
  // kernel = 7, pad = 0, stride = 1
  LOG(paddle_mobile::kLOG_INFO)
      << "int8_t, ceil_mode=false, pooling_type=avg, kernel=7, pad=0, stride=1";
  paddle_mobile::TestPoolOp<int8_t, 0, 1, 7, 0, 1>(in_channels, in_height,
                                                   in_width);
  // kernel = 7, pad = 0, stride = 2
  LOG(paddle_mobile::kLOG_INFO)
      << "int8_t, ceil_mode=false, pooling_type=avg, kernel=7, pad=0, stride=2";
  paddle_mobile::TestPoolOp<int8_t, 0, 1, 7, 0, 2>(in_channels, in_height,
                                                   in_width);
  // kernel = 7, pad = 0, stride = 3
  LOG(paddle_mobile::kLOG_INFO)
      << "int8_t, ceil_mode=false, pooling_type=avg, kernel=7, pad=0, stride=3";
  paddle_mobile::TestPoolOp<int8_t, 0, 1, 7, 0, 3>(in_channels, in_height,
                                                   in_width);
  // kernel = 3, pad = 0, stride = 1
  LOG(paddle_mobile::kLOG_INFO)
      << "int8_t, ceil_mode=false, pooling_type=avg, kernel=3, pad=0, stride=1";
  paddle_mobile::TestPoolOp<int8_t, 0, 1, 3, 0, 1>(in_channels, in_height,
                                                   in_width);
  // kernel = 3, pad = 0, stride = 3
  LOG(paddle_mobile::kLOG_INFO)
      << "int8_t, ceil_mode=false, pooling_type=avg, kernel=3, pad=0, stride=3";
  paddle_mobile::TestPoolOp<int8_t, 0, 1, 3, 0, 3>(in_channels, in_height,
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
  // kernel = 5, pad = 0, stride = 1
  LOG(paddle_mobile::kLOG_INFO)
      << "float, ceil_mode=false, pooling_type=avg, kernel=5, pad=0, stride=1";
  paddle_mobile::TestPoolOp<float, 0, 1, 5, 0, 1>(in_channels, in_height,
                                                  in_width);
}
