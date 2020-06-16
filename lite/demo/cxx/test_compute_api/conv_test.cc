// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <iostream>
#include <memory>
#include <vector>
#include "compute_api.h"    // NOLINT
#include "compute_param.h"  // NOLINT
#include "compute_utils.h"  // NOLINT
#include "paddle_api.h"     // NOLINT
#include "utils.h"          // NOLINT

using namespace paddle::lite_api;  // NOLINT

static int basic_test = 1;
static int batch = 1;
static int in_channel = 32;
static int in_height = 112;
static int in_width = 112;
static int out_channel = 32;
static int group = 1;
static int kernel_h = 3;
static int kernel_w = 3;
static int pad_h0 = 1;
static int pad_h1 = 1;
static int pad_w0 = 1;
static int pad_w1 = 1;
static int stride_h = 1;
static int stride_w = 1;
static int dila_h = 1;
static int dila_w = 1;
static int flag_act = 0;
static int flag_bias = 1;
static float leaky_relu_alpha = 2.f;
static int warmup = 0;
static int repeats = 1;
static int check_result = 1;
static int power_mode = 3;
static int threads = 1;

template <typename Dtype1, typename Dtype2>
static void conv_basic(const Dtype1* din,
                       Dtype2* dout,
                       int num,
                       int chout,
                       int hout,
                       int wout,
                       int chin,
                       int hin,
                       int win,
                       const Dtype1* weights,
                       const Dtype2* bias,
                       int group,
                       int kernel_w,
                       int kernel_h,
                       int stride_w,
                       int stride_h,
                       int dila_w,
                       int dila_h,
                       int pad_w,
                       int pad_h,
                       bool flag_bias,
                       int act_type,
                       float six = 6.f,
                       float scale = 1.f) {
  Dtype2 beta = 0;
  auto src_data = din;
  auto dst_data_ref = dout;
  auto weights_data = weights;
  auto with_bias = flag_bias;
  auto bias_data = bias;

  int in_num = num;
  int out_channels = chout;
  int out_h = hout;
  int out_w = wout;

  int in_channel = chin;
  int in_h = hin;
  int in_w = win;
  int out_c_group = out_channels / group;
  int in_c_group = in_channel / group;

  for (int n = 0; n < in_num; ++n) {
#pragma omp parallel for collapse(4)
    for (int g = 0; g < group; ++g) {
      for (int oc = 0; oc < out_c_group; ++oc) {
        for (int oh = 0; oh < out_h; ++oh) {
          for (int ow = 0; ow < out_w; ++ow) {
            int out_idx = n * group * out_c_group * out_h * out_w +
                          g * out_c_group * out_h * out_w + oc * out_h * out_w +
                          oh * out_w + ow;
            Dtype2 bias_d = with_bias ? (bias_data[g * out_c_group + oc]) : 0;
            dst_data_ref[out_idx] = bias_d;  // + dst_data_ref[out_idx] * beta;
            for (int ic = 0; ic < in_c_group; ++ic) {
              for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                  int iw = ow * stride_w - pad_w + kw * (dila_w);
                  int ih = oh * stride_h - pad_h + kh * (dila_h);
                  if (iw < 0 || iw >= in_w) continue;
                  if (ih < 0 || ih >= in_h) continue;

                  int iidx = n * in_channel * in_h * in_w +
                             g * in_c_group * in_h * in_w + ic * in_h * in_w +
                             ih * in_w + iw;
                  int widx =
                      g * out_c_group * in_c_group * kernel_h * kernel_w +
                      oc * in_c_group * kernel_h * kernel_w +
                      ic * kernel_h * kernel_w + kh * kernel_w + kw;

                  dst_data_ref[out_idx] += src_data[iidx] * weights_data[widx];
                }
              }
            }
            if (act_type > 0) {
              // 1-relu 2-relu6 4-leakyrelu
              if (act_type == 1) {
                dst_data_ref[out_idx] = dst_data_ref[out_idx] > (Dtype2)0
                                            ? dst_data_ref[out_idx]
                                            : (Dtype2)0;
              } else if (act_type == 2) {
                dst_data_ref[out_idx] = dst_data_ref[out_idx] > (Dtype2)0
                                            ? dst_data_ref[out_idx]
                                            : (Dtype2)0;
                dst_data_ref[out_idx] = dst_data_ref[out_idx] < (Dtype2)six
                                            ? dst_data_ref[out_idx]
                                            : (Dtype2)six;
              } else if (act_type == 4) {
                dst_data_ref[out_idx] =
                    dst_data_ref[out_idx] > (Dtype2)0
                        ? dst_data_ref[out_idx]
                        : (Dtype2)(dst_data_ref[out_idx] * scale);
              } else {
                printf("this act type: %d does not support \n", act_type);
              }
            }
          }
        }
      }
    }
  }
}

shape_t compute_out_dim(const shape_t& dim_in, const ConvParam& param) {
  shape_t dim_out = dim_in;
  auto paddings = *param.paddings;
  auto dilations = *param.dilations;
  auto filter_shape = param.filter->shape();
  dim_out[1] = filter_shape[0];
  auto kernel_h = filter_shape[2];
  auto kernel_w = filter_shape[3];
  auto h = dim_in[2];
  auto w = dim_in[3];
  int dila_h = dilations[0];
  int dila_w = dilations[1];
  int pad_top = paddings[0];
  int pad_bottom = paddings[1];
  int pad_left = paddings[2];
  int pad_right = paddings[3];
  int stride_h = param.strides[0];
  int stride_w = param.strides[1];
  auto kernel_exten = dila_h * (kernel_h - 1) + 1;
  auto hout = (h + pad_top + pad_bottom - kernel_exten) / stride_h + 1;
  kernel_exten = dila_w * (kernel_w - 1) + 1;
  auto wout = (w + pad_left + pad_right - kernel_exten) / stride_w + 1;
  dim_out[2] = hout;
  dim_out[3] = wout;
  return dim_out;
}

void test_conv_fp32(const std::vector<shape_t>& input_dims,
                    const shape_t& weight_dim,
                    int group,
                    const std::vector<int>& strides,
                    const std::vector<int>& pads,
                    const std::vector<int>& dilas,
                    bool flag_bias,
                    int flag_act,
                    const int thread_num,
                    const int power_mode,
                    const float leakey_relu_scale) {
  ComputeEngine<TARGET(kARM)>::env_init(static_cast<PowerMode>(power_mode),
                                        thread_num);
  ConvParam param;
  param.x = new Tensor;
  param.x->set_precision(PRECISION(kFloat));
  param.filter = new Tensor;
  param.filter->Resize(weight_dim);
  param.filter->set_precision(PRECISION(kFloat));
  if (flag_bias) {
    param.bias = new Tensor;
    param.bias->Resize({weight_dim[0]});
    param.bias->set_precision(PRECISION(kFloat));
  }
  param.strides = strides;
  param.paddings = std::make_shared<std::vector<int>>(pads);
  param.dilations = std::make_shared<std::vector<int>>(dilas);
  param.groups = group;
  const float six = 6.f;
  if (flag_act > 0) {
    ActivationParam act_param;
    act_param.has_active = true;
    act_param.active_type =
        static_cast<ActivationType>(flag_act);  // 1-relu, 2-relu6, 4-leakyrelu
    if (flag_act == 1) {
      // param.fuse_relu = true;
    } else if (flag_act == 2) {
      act_param.Relu_clipped_coef = six;
    } else if (flag_act == 4) {
      act_param.Leaky_relu_alpha = leakey_relu_scale;
    }
    param.activation_param = act_param;
  }

  param.output = new Tensor;
  param.output->set_precision(PRECISION(kFloat));

  fill_tensor_rand(*param.filter, -1.f, 1.f);
  //  fill_tensor_const(*param.filter, 1.f);
  if (flag_bias) {
    fill_tensor_rand(*param.bias, -1.f, 1.f);
    //    fill_tensor_const(*param.bias, 1.f);
  }
  auto wptr = param.filter->data<float>();
  auto bias_ptr = flag_bias ? param.bias->data<float>() : nullptr;

  ComputeEngine<TARGET(kARM)> conv;
  conv.CreateOperator("conv2d");
  for (auto& dim_in : input_dims) {
    param.x->Resize(dim_in);
    shape_t out_tmp_dims = compute_out_dim(dim_in, param);
    if (out_tmp_dims[2] < 1 || out_tmp_dims[3] < 1) {
      continue;
    }
    param.output->Resize(out_tmp_dims);
    break;
  }
  conv.SetParam(&param);

  for (auto& dim_in : input_dims) {
    if (weight_dim[1] * group != dim_in[1]) {
      "input channel must equal to weights channel\n";
      exit(1);
    }
    shape_t dim_out = compute_out_dim(dim_in, param);
    if (dim_out[2] < 1 || dim_out[3] < 1) {
      continue;
    }
    param.x->Resize(dim_in);
    param.output->Resize(dim_out);

    fill_tensor_rand(*param.x, -1.f, 1.f);
    // fill_tensor_const(*param.x, 1.f);
    auto din = param.x->data<float>();

    Tensor tout_basic;
    if (check_result) {
      tout_basic.set_precision(PRECISION(kFloat));
      tout_basic.Resize(dim_out);
      fill_tensor_const(tout_basic, 0.f);
      auto dout_basic = tout_basic.mutable_data<float>();
      conv_basic<float, float>(din,
                               dout_basic,
                               dim_in[0],
                               dim_out[1],
                               dim_out[2],
                               dim_out[3],
                               dim_in[1],
                               dim_in[2],
                               dim_in[3],
                               wptr,
                               bias_ptr,
                               group,
                               weight_dim[3],
                               weight_dim[2],
                               strides[1],
                               strides[0],
                               dilas[1],
                               dilas[0],
                               pads[2],
                               pads[0],
                               flag_bias,
                               flag_act,
                               six,
                               leakey_relu_scale);
    }
    /// warm up
    for (int i = 0; i < warmup; ++i) {
      conv.Launch();
    }
    /// compute
    Timer t0;
    for (int i = 0; i < repeats; ++i) {
      t0.Start();
      conv.Launch();
      t0.Stop();
    }

    double gops = 2.0 * dim_production(*param.output) * dim_in[1] *
                  weight_dim[2] * weight_dim[3] / param.groups;
    std::cout << "conv fp32: input shape: (" << dim_in[0] << ", " << dim_in[1]
              << ", " << dim_in[2] << ", " << dim_in[3] << "), output shape: ("
              << dim_out[0] << ", " << dim_out[1] << ", " << dim_out[2] << ", "
              << dim_out[3] << "),running time, avg: " << t0.LapTimes().Avg()
              << ", min time: " << t0.LapTimes().Min()
              << ", total GOPS: " << 1e-9 * gops
              << " GOPS, avg GOPs: " << 1e-6 * gops / t0.LapTimes().Avg()
              << " GOPs, max GOPs: " << 1e-6 * gops / t0.LapTimes().Min()
              << std::endl;

    if (check_result) {
      double max_ratio = 0;
      double max_diff = 0;
      tensor_cmp_host(tout_basic, *param.output, max_ratio, max_diff);
      std::cout << "compare result, max diff: " << max_diff
                << ", max ratio: " << max_ratio << std::endl;
      if (std::abs(max_ratio) > 1e-3f) {
        if (max_diff > 5e-4f) {
          std::cout << "basic result\n";
          print_tensor(tout_basic);
          std::cout << "lite result\n";
          print_tensor(*param.output);
          Tensor tdiff;
          tdiff.Resize(tout_basic.shape());
          tdiff.set_precision(PRECISION(kFloat));
          tensor_diff(tout_basic, *param.output, tdiff);
          print_tensor(tdiff);
          std::cerr << "test fp32 conv: input: (" << dim_in[0] << ", "
                    << dim_in[1] << ", " << dim_in[2] << ", " << dim_in[3]
                    << "), output: (" << dim_out[0] << ", " << dim_out[1]
                    << ", " << dim_out[2] << ", " << dim_out[3]
                    << "), weight dim: (" << weight_dim[0] << ", "
                    << weight_dim[1] << ", " << weight_dim[2] << ", "
                    << weight_dim[3] << "), pad: " << pads[0] << ", " << pads[1]
                    << ", " << pads[2] << ", " << pads[3]
                    << ", stride: " << strides[0] << ", " << strides[1]
                    << ", dila_: " << dilas[0] << ", " << dilas[1]
                    << ", group: " << group
                    << ", bias: " << (flag_bias ? "true" : "false")
                    << ", act: " << flag_act << ", threads: " << thread_num
                    << ", power_mode: " << power_mode << " failed!!\n";
          exit(1);
        }
      }
    }
    std::cout << "test fp32 conv: input: (" << dim_in[0] << ", " << dim_in[1]
              << ", " << dim_in[2] << ", " << dim_in[3] << "), output: ("
              << dim_out[0] << ", " << dim_out[1] << ", " << dim_out[2] << ", "
              << dim_out[3] << "), weight dim: (" << weight_dim[0] << ", "
              << weight_dim[1] << ", " << weight_dim[2] << ", " << weight_dim[3]
              << "), pad: " << pads[0] << ", " << pads[1] << ", " << pads[2]
              << ", " << pads[3] << ", stride: " << strides[0] << ", "
              << strides[1] << ", dila_: " << dilas[0] << ", " << dilas[1]
              << ", group: " << group
              << ", bias: " << (flag_bias ? "true" : "false")
              << ", act: " << flag_act << ", threads: " << thread_num
              << ", power_mode: " << power_mode << " success!!\n";
  }
  param.x->ReleaseRawTensor();
  param.filter->ReleaseRawTensor();
  param.output->ReleaseRawTensor();
  if (flag_bias) {
    param.bias->ReleaseRawTensor();
  }
  delete param.x;
  delete param.filter;
  delete param.output;
  delete param.bias;
}

int main(int argc, const char** argv) {
  if (argc < 2) {
    std::cout << "usage: ./" << argv[0]
              << "basic_test check_result batch in_channel in_height in_width "
                 "out_channel group kernel_h pad_h0 stride_h dila_h flag_act "
                 "flag_bias warmup repeats threads power_mode."
              << std::endl;
    return 0;
  }
  if (argc >= 2) {
    basic_test = atoi(argv[1]);
  }
  if (argc >= 3) {
    check_result = atoi(argv[2]);
  }
  if (argc >= 4) {
    batch = atoi(argv[3]);
  }
  if (argc >= 5) {
    in_channel = atoi(argv[4]);
  }
  if (argc >= 6) {
    in_height = atoi(argv[5]);
  }
  if (argc >= 7) {
    in_width = atoi(argv[6]);
  }
  if (argc >= 8) {
    out_channel = atoi(argv[7]);
  }
  if (argc >= 9) {
    group = atof(argv[8]);
  }
  if (argc >= 10) {
    if (argc >= 13) {
      kernel_h = atoi(argv[9]);
      kernel_w = kernel_h;
      pad_h0 = atoi(argv[10]);
      pad_h1 = pad_h0;
      pad_w0 = pad_h0;
      pad_w1 = pad_h0;
      stride_h = atoi(argv[11]);
      stride_w = stride_h;
      dila_h = atoi(argv[12]);
      dila_w = dila_h;
    } else {
      std::cout
          << "kernel_h padh0 stride_h dila_h must be set at the same time."
          << std::endl;
    }
  }
  if (argc >= 14) {
    flag_act = atoi(argv[13]);
  }
  if (argc >= 15) {
    flag_bias = atoi(argv[14]);
  }
  if (argc >= 16) {
    warmup = atoi(argv[15]);
  }
  if (argc >= 17) {
    repeats = atoi(argv[16]);
  }
  if (argc >= 18) {
    threads = atoi(argv[17]);
  }
  if (argc >= 19) {
    power_mode = atoi(argv[18]);
  }
  if (argc >= 20) {
    leaky_relu_alpha = atof(argv[19]);
  }
  // basic test
  if (basic_test) {
    std::cout << "RUN BASIC TEST BEGIN: " << std::endl;
    for (auto& cin : {1, 3, 8}) {
      for (auto& cout : {1, 5, 16}) {
        for (auto& g : {1, 2}) {
          for (auto& kw : {1, 2, 3}) {
            for (auto& kh : {1, 2, 3}) {
              for (auto& stride : {1, 2}) {
                for (auto& pad_left : {0, 2}) {
                  for (auto& pad_right : {0, 2}) {
                    for (auto& pad_top : {0, 2}) {
                      for (auto& pad_bottom : {0, 2}) {
                        for (auto& dila : {1, 2}) {
                          for (auto& flag_bias : {false, true}) {
                            for (auto& flag_act : {0, 1, 2, 4}) {
                              for (auto& threads : {1, 2, 4}) {
                                if (cin % g != 0 || cout % g != 0) {
                                  continue;
                                }
                                std::vector<shape_t> dims;
                                shape_t weights_dim({cout, cin / g, kh, kw});
                                for (auto& batch : {1, 2}) {
                                  for (auto& h : {1, 3, 19, 32}) {
                                    dims.push_back(shape_t({batch, cin, h, h}));
                                  }
                                }
                                // skip 3x3 depthwise conv
                                if (g == cin && cin == cout && kw == 3 &&
                                    kh == 3) {
                                  break;
                                }
                                // skip 3x3s1 direct conv
                                if (g == 1 && (cin != 1 || cout != 1) &&
                                    kw == 3 && kh == 3 && stride == 1) {
                                  break;
                                }
                                const float leakey_relu_scale = 2.22;
                                test_conv_fp32(
                                    dims,
                                    weights_dim,
                                    g,
                                    {stride, stride},
                                    {pad_top, pad_bottom, pad_left, pad_right},
                                    {dila, dila},
                                    flag_bias,
                                    flag_act,
                                    threads,
                                    3,
                                    leakey_relu_scale);
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
    }
    std::cout << "RUN BASIC TEST END: " << std::endl;
  }

  // costum test
  std::cout << "RUN CUSTOM TEST BEGIN: " << std::endl;
  std::vector<shape_t> dims;
  dims.emplace_back(shape_t({batch, in_channel, in_height, in_width}));
  shape_t weights_dim({out_channel, in_channel / group, kernel_h, kernel_w});
  test_conv_fp32(dims,
                 weights_dim,
                 group,
                 {stride_h, stride_w},
                 {pad_h0, pad_h1, pad_w0, pad_w1},
                 {dila_h, dila_w},
                 flag_bias,
                 flag_act,
                 threads,
                 3,
                 leaky_relu_alpha);
  std::cout << "RUN CUSTOM TEST END: " << std::endl;
  return 0;
}
