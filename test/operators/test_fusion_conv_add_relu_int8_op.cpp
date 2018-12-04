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

#ifdef FUSION_CONVADDRELU_INT8_OP

#include <iostream>
#include <limits>
#include "../test_helper.h"
#include "../test_include.h"
#include "operators/fusion_conv_add_relu_int8_op.h"

namespace paddle_mobile {
int32_t qadd_int32(int32_t l, int32_t r) {
  int64_t res = static_cast<int64_t>(l) + static_cast<int64_t>(r);
  if (res > std::numeric_limits<int32_t>::max())
    return std::numeric_limits<int32_t>::max();
  else if (res < std::numeric_limits<int32_t>::min())
    return std::numeric_limits<int32_t>::min();
  else
    return static_cast<int32_t>(res);
}

// round to zero
float round2zero(float v) {
  float res;
  if (v > 0)
    res = std::floor(v);
  else if (v < 0)
    res = std::ceil(v);
  return res;
}

int8_t qscale_int32(int32_t v, float scale) {
  float res = static_cast<float>(v) * scale;
  res = round2zero(res);
  if (res > 127)
    return static_cast<int8_t>(127);
  else if (res < -127)
    return static_cast<int8_t>(-127);
  else
    return static_cast<int8_t>(res);
}

// Reference convolution from Caffe for checking results.
// accumulate through explicit loops over input, output, and filters.
template <typename T>
void conv2d(const framework::Tensor *input, const framework::Tensor *filter,
            const framework::Tensor *bias, const framework::AttributeMap &attrs,
            framework::Tensor *output, float scale) {
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

  const T *in_data = input->data<T>();
  const T *w_data = filter->data<T>();
  framework::Tensor output_32;
  int32_t *out_data_32 = output_32.mutable_data<int32_t>(out_shape);
  memset(out_data_32, 0, output_32.numel() * sizeof(int32_t));
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

                        out_data_32[offset(output, out_offset)] +=
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

  T *out_data = output->mutable_data<T>();
  int32_t n = out_shape[0];
  int32_t c = out_shape[1];
  int32_t h = out_shape[2];
  int32_t w = out_shape[3];
  const int32_t *bias_data = bias->data<int32_t>();
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < c; ++j) {
      int32_t bias_v = bias_data[j];
      for (int k = 0; k < h; ++k) {
        for (int l = 0; l < w; ++l) {
          int32_t tmp = out_data_32[i * c * h * w + j * h * w + k * w + l];
          tmp = qadd_int32(tmp, bias_v);
          tmp = std::max(0, tmp);
          out_data[i * c * h * w + j * h * w + k * w + l] =
              qscale_int32(tmp, scale);
        }
      }
    }
  }
}

template <typename T, int Kernel, int Pad, int Stride>
int TestConvOp(int in_channels, int in_height, int in_width, int out_channels) {
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
      framework::make_ddim({output_c, input_c, kernel_h, kernel_w});

  int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;
  int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
  int output_h = (input_h + 2 * pad_h - kernel_extent_h) / stride_h + 1;
  int output_w = (input_w + 2 * pad_w - kernel_extent_w) / stride_w + 1;
  framework::DDim output_shape = framework::make_ddim(
      std::vector<int>({batch_size, output_c, output_h, output_w}));

  framework::DDim bias_shape = framework::make_ddim({output_c});

  VariableNameMap inputs;
  VariableNameMap outputs;
  auto scope = std::make_shared<framework::Scope>();
  inputs["Input"] = std::vector<std::string>({"input"});
  inputs["Filter"] = std::vector<std::string>({"filter"});
  inputs["Scale"] = std::vector<std::string>({"scale"});
  inputs["Y"] = std::vector<std::string>({"bias"});
  outputs["Out"] = std::vector<std::string>({"output"});

  auto input_var = scope.get()->Var("input");
  auto input = input_var->template GetMutable<framework::LoDTensor>();
  SetupTensor<T>(input, input_shape, -127, 127);

  auto filter_var = scope.get()->Var("filter");
  auto filter = filter_var->template GetMutable<framework::LoDTensor>();
  SetupTensor<T>(filter, filter_shape, -127, 127);

  auto scale_var = scope.get()->Var("scale");
  auto scale = scale_var->template GetMutable<framework::LoDTensor>();
  scale->Resize(framework::make_ddim({1}));
  float scale_v = 0.000828f;
  scale->mutable_data<float>()[0] = scale_v;

  auto bias_var = scope.get()->Var("bias");
  auto bias = bias_var->template GetMutable<framework::LoDTensor>();
  SetupTensor<int32_t>(bias, bias_shape, -127, 127);

  auto output_var = scope.get()->Var("output");
  framework::AttributeMap attrs;
  attrs["strides"].Set<vector<int>>(std::vector<int>({stride_h, stride_w}));
  attrs["paddings"].Set<vector<int>>(std::vector<int>({pad_h, pad_w}));
  attrs["dilations"].Set<vector<int>>(
      std::vector<int>({dilation_h, dilation_w}));
  attrs["groups"].Set<int>(1);
  attrs["axis"].Set<int>(0);

  auto *op = new operators::FusionConvAddReluInt8Op<CPU, T>(
      "fusion_conv_add_relu_int8", inputs, outputs, attrs, scope);
  op->InferShape();
  op->Init();
  op->Run();

  framework::Tensor output_cmp;
  output_cmp.mutable_data<T>(output_shape);
  conv2d<T>(input, filter, bias, attrs, &output_cmp, scale_v);

  // compare results
  int eq = 0;
  int neq = 0;
  auto output = output_var->template Get<framework::LoDTensor>();
  const T *output_data = output->data<T>();
  T *output_cmp_data = output_cmp.data<T>();
  for (int i = 0; i < output->numel(); ++i) {
    PADDLE_MOBILE_ENFORCE(
        output_data[i] == output_cmp_data[i],
        "The execution of test_fusion_conv_add_relu_int8_op is failed!");
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
  if (argc < 5) {
    LOG(paddle_mobile::kLOG_INFO)
        << "Usage:\n"
        << "  ./test-conv-add-relu-int8-op in_channels in_height in_width "
           "out_channels\n"
        << "  params:\n"
        << "   -in_channels: int, input image's channels\n"
        << "   -in_height: int, input image's height\n"
        << "   -in_width: int, input image's width\n"
        << "   -out_channels: int, conv output channels\n";
    return 1;
  }
  int in_channels = atoi(argv[1]);
  int in_height = atoi(argv[2]);
  int in_width = atoi(argv[3]);
  int out_channels = atoi(argv[4]);
  // kernel = 3, pad = 1, stride = 1
  LOG(paddle_mobile::kLOG_INFO) << "int8_t, kernel=3, pad=1, stride=1";
  paddle_mobile::TestConvOp<int8_t, 3, 1, 1>(in_channels, in_height, in_width,
                                             out_channels);
  // kernel = 7, pad = 0, stride = 2
  LOG(paddle_mobile::kLOG_INFO) << "int8, kernel=7, pad=0, stride=2";
  paddle_mobile::TestConvOp<int8_t, 7, 0, 2>(in_channels, in_height, in_width,
                                             out_channels);
  // kernel = 7, pad = 1, stride = 2
  LOG(paddle_mobile::kLOG_INFO) << "int8, kernel=7, pad=1, stride=2";
  paddle_mobile::TestConvOp<int8_t, 7, 1, 2>(in_channels, in_height, in_width,
                                             out_channels);
  // kernel = 7, pad = 3, stride = 2
  LOG(paddle_mobile::kLOG_INFO) << "int8, kernel=7, pad=3, stride=2";
  paddle_mobile::TestConvOp<int8_t, 7, 3, 2>(in_channels, in_height, in_width,
                                             out_channels);
  // kernel = 7, pad = 0, stride = 1
  LOG(paddle_mobile::kLOG_INFO) << "int8, kernel=7, pad=0, stride=1";
  paddle_mobile::TestConvOp<int8_t, 7, 0, 1>(in_channels, in_height, in_width,
                                             out_channels);
  // kernel = 7, pad = 1, stride = 1
  LOG(paddle_mobile::kLOG_INFO) << "int8, kernel=7, pad=1, stride=1";
  paddle_mobile::TestConvOp<int8_t, 7, 1, 1>(in_channels, in_height, in_width,
                                             out_channels);
  // kernel = 7, pad = 3, stride = 1
  LOG(paddle_mobile::kLOG_INFO) << "int8, kernel=7, pad=3, stride=1";
  paddle_mobile::TestConvOp<int8_t, 7, 3, 1>(in_channels, in_height, in_width,
                                             out_channels);
  // kernel = 7, pad = 5, stride = 3
  LOG(paddle_mobile::kLOG_INFO) << "int8, kernel=7, pad=5, stride=3";
  paddle_mobile::TestConvOp<int8_t, 7, 5, 3>(in_channels, in_height, in_width,
                                             out_channels);
  // kernel = 7, pad = 3, stride = 4
  LOG(paddle_mobile::kLOG_INFO) << "int8, kernel=7, pad=3, stride=4";
  paddle_mobile::TestConvOp<int8_t, 7, 3, 4>(in_channels, in_height, in_width,
                                             out_channels);
  // kernel = 3, pad = 0, stride = 1
  LOG(paddle_mobile::kLOG_INFO) << "int8, kernel=3, pad=0, stride=1";
  paddle_mobile::TestConvOp<int8_t, 3, 0, 1>(in_channels, in_height, in_width,
                                             out_channels);
  // kernel = 3, pad = 1, stride = 1
  LOG(paddle_mobile::kLOG_INFO) << "int8, kernel=3, pad=1, stride=1";
  paddle_mobile::TestConvOp<int8_t, 3, 1, 1>(in_channels, in_height, in_width,
                                             out_channels);

  // kernel = 5, pad = 0, stride = 1
  LOG(paddle_mobile::kLOG_INFO) << "int8, kernel=5, pad=0, stride=1";
  paddle_mobile::TestConvOp<int8_t, 5, 0, 1>(in_channels, in_height, in_width,
                                             out_channels);

  // kernel = 5, pad = 2, stride = 1
  LOG(paddle_mobile::kLOG_INFO) << "int8, kernel=5, pad=2, stride=1";
  paddle_mobile::TestConvOp<int8_t, 5, 2, 1>(in_channels, in_height, in_width,
                                             out_channels);
}

#endif
