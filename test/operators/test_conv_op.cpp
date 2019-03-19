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
#include "../test_helper.h"
#include "../test_include.h"
#include "operators/conv_op.h"

namespace paddle_mobile {

// Reference convolution from Caffe for checking results.
// accumulate through explicit loops over input, output, and filters.
template <typename Itype, typename Otype>
void conv2d(const framework::Tensor *input, const framework::Tensor *filter,
            const framework::AttributeMap &attrs, framework::Tensor *output) {
  framework::AttrReader attr_reader(attrs);
  std::vector<int> paddings = attr_reader.Get<std::vector<int>>("paddings");
  std::vector<int> strides = attr_reader.Get<std::vector<int>>("strides");
  std::vector<int> dilations = attr_reader.Get<std::vector<int>>("dilations");
  int groups = attr_reader.Get<int>("groups");
  int kernel_h = filter->dims()[2];
  int kernel_w = filter->dims()[3];
  int pad_h = paddings[0];
  int pad_w = paddings[1];
  int stride_h = strides[0];
  int stride_w = strides[1];
  int dilation_h = dilations[0];
  int dilation_w = dilations[1];
  auto in_shape = input->dims();
  auto out_shape = output->dims();

  const bool has_depth = 0;
  int kernel_d, pad_d, stride_d, dilation_d;
  if (has_depth) {
    kernel_d = kernel_h;
    stride_d = stride_h;
    pad_d = pad_h;
    dilation_d = dilation_h;
  } else {
    kernel_d = stride_d = dilation_d = 1;
    pad_d = 0;
  }
  // Groups
  int o_g = out_shape[1] / groups;
  int k_g = in_shape[1] / groups;
  int o_head, k_head;
  // Convolution
  vector<int> weight_offset(4 + has_depth);
  vector<int> in_offset(4 + has_depth);
  vector<int> out_offset(4 + has_depth);
  auto offset = [](const framework::Tensor *input, const vector<int> &indics) {
    framework::DDim shape = input->dims();
    size_t count = 0;
    for (int i = 0; i < indics.size(); ++i) {
      count *= shape[i];
      count += indics[i];
    }
    return count;
  };

  const Itype *in_data = input->data<Itype>();
  const Itype *w_data = filter->data<Itype>();
  Otype *out_data = output->mutable_data<Otype>();
  memset(out_data, 0, output->numel() * sizeof(Otype));
  for (int n = 0; n < out_shape[0]; n++) {
    for (int g = 0; g < groups; g++) {
      o_head = o_g * g;
      k_head = k_g * g;
      for (int o = 0; o < o_g; o++) {
        for (int k = 0; k < k_g; k++) {
          for (int z = 0; z < (has_depth ? out_shape[2] : 1); z++) {
            for (int y = 0; y < out_shape[2 + has_depth]; y++) {
              for (int x = 0; x < out_shape[3 + has_depth]; x++) {
                for (int r = 0; r < kernel_d; r++) {
                  for (int p = 0; p < kernel_h; p++) {
                    for (int q = 0; q < kernel_w; q++) {
                      int in_z = z * stride_d - pad_d + r * dilation_d;
                      int in_y = y * stride_h - pad_h + p * dilation_h;
                      int in_x = x * stride_w - pad_w + q * dilation_w;
                      if (in_z >= 0 && in_z < (has_depth ? in_shape[2] : 1) &&
                          in_y >= 0 && in_y < in_shape[2 + has_depth] &&
                          in_x >= 0 && in_x < in_shape[3 + has_depth]) {
                        weight_offset[0] = o + o_head;
                        weight_offset[1] = k;
                        if (has_depth) {
                          weight_offset[2] = r;
                        }
                        weight_offset[2 + has_depth] = p;
                        weight_offset[3 + has_depth] = q;
                        in_offset[0] = n;
                        in_offset[1] = k + k_head;
                        if (has_depth) {
                          in_offset[2] = in_z;
                        }
                        in_offset[2 + has_depth] = in_y;
                        in_offset[3 + has_depth] = in_x;
                        out_offset[0] = n;
                        out_offset[1] = o + o_head;
                        if (has_depth) {
                          out_offset[2] = z;
                        }
                        out_offset[2 + has_depth] = y;
                        out_offset[3 + has_depth] = x;

                        out_data[offset(output, out_offset)] +=
                            in_data[offset(input, in_offset)] *
                            w_data[offset(filter, weight_offset)];
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

  VariableNameMap inputs;
  VariableNameMap outputs;
  auto scope = std::make_shared<framework::Scope>();
  inputs["Input"] = std::vector<std::string>({"input"});
  inputs["Filter"] = std::vector<std::string>({"filter"});
  outputs["Output"] = std::vector<std::string>({"output"});

  auto input_var = scope.get()->Var("input");
  auto input = input_var->template GetMutable<framework::LoDTensor>();
  SetupTensor<Itype>(input, input_shape, -20.0, 20.0);

  auto filter_var = scope.get()->Var("filter");
  auto filter = filter_var->template GetMutable<framework::LoDTensor>();
  SetupTensor<Itype>(filter, filter_shape, -20, 20);

  //  for (int i = 0; i < input->numel(); ++i) {
  //    DLOG << "input[" << i << "] = " << float(input->data<Itype>()[i]);
  //  }
  //  for (int i = 0; i < filter->numel(); ++i) {
  //    DLOG << "filter[" << i << "] = " << float(filter->data<Itype>()[i]);
  //  }

  auto output_var = scope.get()->Var("output");
  framework::AttributeMap attrs;
  attrs["strides"].Set<vector<int>>(std::vector<int>({stride_h, stride_w}));
  attrs["paddings"].Set<vector<int>>(std::vector<int>({pad_h, pad_w}));
  attrs["dilations"].Set<vector<int>>(
      std::vector<int>({dilation_h, dilation_w}));
  attrs["groups"].Set<int>(groups);

  auto *op = new operators::ConvOp<CPU, float>("conv2d", inputs, outputs, attrs,
                                               scope.get());
  op->InferShape();
  op->Init();
  //  struct timespec ts_begin, ts_end;
  // warmup
  //  op->Run();
  //  clock_gettime(CLOCK_MONOTONIC, &ts_begin);
  //  for (int i = 0; i < 10; ++i) {
  op->Run();
  //  }
  //  clock_gettime(CLOCK_MONOTONIC, &ts_end);
  //  uint64_t elapsed = (ts_end.tv_sec - ts_begin.tv_sec) * 1e3 +
  //                     (ts_end.tv_nsec - ts_begin.tv_nsec) / 1e6;
  //  LOG(kLOG_INFO) << "elapsed: " << elapsed / 10.0 << " ms";

  // compare results
  auto *output = output_var->template Get<framework::LoDTensor>();
  framework::Tensor output_cmp;
  output_cmp.mutable_data<Otype>(output->dims());
  conv2d<Itype, Otype>(input, filter, attrs, &output_cmp);

  const Otype *output_data = output->data<Otype>();
  Otype *output_cmp_data = output_cmp.data<Otype>();
  for (int i = 0; i < output->numel(); ++i) {
    float gap = abs(output_data[i] - output_cmp_data[i]);
    //    PADDLE_MOBILE_ENFORCE(std::abs(gap / (output_data[i] + 1e-5)) < 1e-3,
    //                          "output[%d] = %d, output_cmp[%d] = %d", i,
    //                          output_data[i], i, output_cmp_data[i]);
    if (gap > 1e-2 && (gap / (abs(output_data[i]) + 1e-5) > 1e-2)) {
      std::cerr << "output_data[" << i << "] = " << output_data[i]
                << ", output_cmp_data[" << i << "] = " << output_cmp_data[i]
                << std::endl;
      exit(1);
    }
  }
  delete op;
  return 0;
}

}  // namespace paddle_mobile

int TestAll(const int in_channels, const int in_height, const int in_width,
            const int out_channels, const int groups) {
  std::cerr << "in_channels=" << in_channels << ", in_height=" << in_height
            << ", in_width=" << in_width << ", out_channels=" << out_channels
            << ", groups=" << groups << std::endl;
  std::cerr << "float, kernel=1, pad=0, stride=1" << std::endl;
  paddle_mobile::TestConvOp<float, float, 1, 0, 1>(
      in_channels, in_height, in_width, out_channels, groups);

  // kernel = 3, pad = 0, stride = 1
  std::cerr << "float, kernel=3, pad=0, stride=1" << std::endl;
  paddle_mobile::TestConvOp<float, float, 3, 0, 1>(
      in_channels, in_height, in_width, out_channels, groups);
  // kernel = 3, pad = 1, stride = 1
  std::cerr << "float, kernel=3, pad=1, stride=1" << std::endl;
  paddle_mobile::TestConvOp<float, float, 3, 1, 1>(
      in_channels, in_height, in_width, out_channels, groups);
  // kernel = 3, pad = 2, stride = 1
  std::cerr << "float, kernel=3, pad=2, stride=1" << std::endl;
  paddle_mobile::TestConvOp<float, float, 3, 2, 1>(
      in_channels, in_height, in_width, out_channels, groups);
  // kernel = 3, pad = 5, stride = 1
  std::cerr << "float, kernel=3, pad=5, stride=1" << std::endl;
  paddle_mobile::TestConvOp<float, float, 3, 5, 1>(
      in_channels, in_height, in_width, out_channels, groups);

  // kernel = 3, pad = 0, stride = 2
  std::cerr << "float, kernel=3, pad=0, stride=2" << std::endl;
  paddle_mobile::TestConvOp<float, float, 3, 0, 2>(
      in_channels, in_height, in_width, out_channels, groups);
  // kernel = 3, pad = 1, stride = 2
  std::cerr << "float, kernel=3, pad=1, stride=2" << std::endl;
  paddle_mobile::TestConvOp<float, float, 3, 1, 2>(
      in_channels, in_height, in_width, out_channels, groups);
  // kernel = 3, pad = 2, stride = 2
  std::cerr << "float, kernel=3, pad=2, stride=2" << std::endl;
  paddle_mobile::TestConvOp<float, float, 3, 2, 2>(
      in_channels, in_height, in_width, out_channels, groups);
  // kernel = 3, pad = 5, stride = 2
  std::cerr << "float, kernel=3, pad=5, stride=2" << std::endl;
  paddle_mobile::TestConvOp<float, float, 3, 5, 2>(
      in_channels, in_height, in_width, out_channels, groups);

#ifndef __aarch64__
  // kernel = 3, pad = 0, stride = 1
  std::cerr << "int8, kernel=3, pad=0, stride=1" << std::endl;
  paddle_mobile::TestConvOp<int8_t, int32_t, 3, 0, 1>(
      in_channels, in_height, in_width, out_channels, groups);
  // kernel = 3, pad = 1, stride = 1
  std::cerr << "int8, kernel=3, pad=1, stride=1" << std::endl;
  paddle_mobile::TestConvOp<int8_t, int32_t, 3, 1, 1>(
      in_channels, in_height, in_width, out_channels, groups);
  // kernel = 3, pad = 2, stride = 1
  std::cerr << "int8, kernel=3, pad=2, stride=1" << std::endl;
  paddle_mobile::TestConvOp<int8_t, int32_t, 3, 2, 1>(
      in_channels, in_height, in_width, out_channels, groups);
  // kernel = 3, pad = 5, stride = 1
  std::cerr << "int8, kernel=3, pad=5, stride=1" << std::endl;
  paddle_mobile::TestConvOp<int8_t, int32_t, 3, 5, 1>(
      in_channels, in_height, in_width, out_channels, groups);

  // kernel = 3, pad = 0, stride = 2
  std::cerr << "int8, kernel=3, pad=0, stride=2" << std::endl;
  paddle_mobile::TestConvOp<int8_t, int32_t, 3, 0, 2>(
      in_channels, in_height, in_width, out_channels, groups);
  // kernel = 3, pad = 1, stride = 2
  std::cerr << "int8, kernel=3, pad=1, stride=2" << std::endl;
  paddle_mobile::TestConvOp<int8_t, int32_t, 3, 1, 2>(
      in_channels, in_height, in_width, out_channels, groups);
  // kernel = 3, pad = 2, stride = 2
  std::cerr << "int8, kernel=3, pad=2, stride=2" << std::endl;
  paddle_mobile::TestConvOp<int8_t, int32_t, 3, 2, 2>(
      in_channels, in_height, in_width, out_channels, groups);
  // kernel = 3, pad = 5, stride = 2
  std::cerr << "int8, kernel=3, pad=5, stride=2" << std::endl;
  paddle_mobile::TestConvOp<int8_t, int32_t, 3, 5, 2>(
      in_channels, in_height, in_width, out_channels, groups);
#endif  // __aarch64__

  // kernel = 5, pad = 0, stride = 1
  std::cerr << "float, kernel=5, pad=0, stride=1" << std::endl;
  paddle_mobile::TestConvOp<float, float, 5, 0, 1>(
      in_channels, in_height, in_width, out_channels, groups);
  // kernel = 5, pad = 1, stride = 1
  std::cerr << "float, kernel=5, pad=1, stride=1" << std::endl;
  paddle_mobile::TestConvOp<float, float, 5, 1, 1>(
      in_channels, in_height, in_width, out_channels, groups);
  // kernel = 5, pad = 2, stride = 1
  std::cerr << "float, kernel=5, pad=2, stride=1" << std::endl;
  paddle_mobile::TestConvOp<float, float, 5, 2, 1>(
      in_channels, in_height, in_width, out_channels, groups);
  // kernel = 5, pad = 5, stride = 1
  std::cerr << "float, kernel=5, pad=5, stride=1" << std::endl;
  paddle_mobile::TestConvOp<float, float, 5, 5, 1>(
      in_channels, in_height, in_width, out_channels, groups);

#ifndef __aarch64__
  // kernel = 5, pad = 0, stride = 1
  std::cerr << "int8, kernel=5, pad=0, stride=1" << std::endl;
  paddle_mobile::TestConvOp<int8_t, int32_t, 5, 0, 1>(
      in_channels, in_height, in_width, out_channels, groups);
  // kernel = 5, pad = 1, stride = 1
  std::cerr << "int8, kernel=5, pad=1, stride=1" << std::endl;
  paddle_mobile::TestConvOp<int8_t, int32_t, 5, 1, 1>(
      in_channels, in_height, in_width, out_channels, groups);
  // kernel = 5, pad = 2, stride = 1
  std::cerr << "int8, kernel=5, pad=2, stride=1" << std::endl;
  paddle_mobile::TestConvOp<int8_t, int32_t, 5, 2, 1>(
      in_channels, in_height, in_width, out_channels, groups);
  // kernel = 5, pad = 5, stride = 1
  std::cerr << "int8, kernel=5, pad=5, stride=1" << std::endl;
  paddle_mobile::TestConvOp<int8_t, int32_t, 5, 5, 1>(
      in_channels, in_height, in_width, out_channels, groups);
#endif  // __aarch64__

  return 0;
}

int main() {
  TestAll(16, 10, 10, 16, 16);
  TestAll(1, 5, 5, 1, 1);
  TestAll(1, 5, 5, 10, 1);
  TestAll(10, 5, 5, 10, 10);

  TestAll(5, 33, 33, 5, 1);
  TestAll(5, 33, 33, 13, 1);
  TestAll(13, 33, 33, 13, 13);

  TestAll(5, 33, 13, 5, 1);
  TestAll(5, 33, 13, 13, 1);
  TestAll(13, 33, 13, 13, 13);
  return 0;
}
