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
static int flag_relu = 0;
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

template <PrecisionType ptype>
void get_conv_param(const shape_t& dim_w,
                    int g,
                    const std::vector<int>& strides,
                    const std::vector<int>& pads,
                    const std::vector<int>& dila,
                    bool flag_bias,
                    bool flag_relu,
                    ConvParam* param) {
  param->x = new Tensor;
  param->x->set_precision(PRECISION(kInt8));
  param->filter = new Tensor;
  param->filter->Resize(dim_w);
  param->filter->set_precision(PRECISION(kInt8));
  if (flag_bias) {
    param->bias = new Tensor;
    param->bias->Resize({dim_w[0]});
    param->bias->set_precision(PRECISION(kFloat));
  }
  param->enable_int8 = true;
  param->strides = strides;
  param->paddings = std::make_shared<std::vector<int>>(pads);
  param->dilations = std::make_shared<std::vector<int>>(dila);
  if (flag_relu) {
    param->activation_param.has_active = true;
    param->activation_param.active_type = ActivationType::kRelu;
  }
  param->groups = g;

  param->output = new Tensor;
  param->output->set_precision(ptype);
  param->out_ptype = ptype;
}

void release_param(ConvParam* param) {
  param->x->ReleaseRawTensor();
  param->filter->ReleaseRawTensor();
  param->output->ReleaseRawTensor();
  if (param->bias) {
    param->bias->ReleaseRawTensor();
  }
  delete param->x;
  delete param->filter;
  delete param->output;
  delete param->bias;
}

void test_conv_int8(const std::vector<shape_t>& input_dims,
                    const shape_t& weight_dim,
                    int group,
                    const std::vector<int>& strides,
                    const std::vector<int>& pads,
                    const std::vector<int>& dilas,
                    bool flag_bias,
                    bool flag_relu,
                    const int thread_num,
                    const int power_mode) {
  ComputeEngine<TARGET(kARM)>::env_init(static_cast<PowerMode>(power_mode),
                                        thread_num);
  ConvParam param_int8_out;
  ConvParam param_fp32_out;

  get_conv_param<PRECISION(kInt8)>(weight_dim,
                                   group,
                                   strides,
                                   pads,
                                   dilas,
                                   flag_bias,
                                   flag_relu,
                                   &param_int8_out);

  get_conv_param<PRECISION(kFloat)>(weight_dim,
                                    group,
                                    strides,
                                    pads,
                                    dilas,
                                    flag_bias,
                                    flag_relu,
                                    &param_fp32_out);
  Tensor weight_fp32;
  Tensor bias_fp32;
  weight_fp32.Resize(weight_dim);
  fill_tensor_rand(*param_int8_out.filter, -127, 127);
  param_fp32_out.filter->CopyFromCpu<int8_t>(
      param_int8_out.filter->data<int8_t>());
  if (flag_bias) {
    auto dim_b = param_int8_out.bias->shape();
    bias_fp32.Resize(dim_b);
    fill_tensor_rand(*param_int8_out.bias, -1.f, 1.f);
    param_fp32_out.bias->CopyFromCpu<float>(param_int8_out.bias->data<float>());
    bias_fp32.CopyFromCpu<float>(param_int8_out.bias->data<float>());
  }

  std::vector<float> scale_in{1.f / 127};
  std::vector<float> scale_out{weight_dim[1] * weight_dim[2] * weight_dim[3] /
                               127.f};
  std::vector<float> scale_w(weight_dim[0], 1.f / 127);

  param_int8_out.input_scale = scale_in[0];
  param_int8_out.output_scale = scale_out[0];
  param_int8_out.weight_scale = scale_w;

  param_fp32_out.input_scale = scale_in[0];
  param_fp32_out.output_scale = scale_out[0];
  param_fp32_out.weight_scale = scale_w;

  auto wptr_fp32 = weight_fp32.mutable_data<float>();
  auto bptr_fp32 = flag_bias ? bias_fp32.data<float>() : nullptr;
  ComputeUtils::ConvWeightsInt8ToFloat(
      *param_int8_out.filter, weight_fp32, scale_w);

  ComputeEngine<TARGET(kARM)> conv_int8_int8;
  ComputeEngine<TARGET(kARM)> conv_int8_fp32;
  conv_int8_int8.CreateOperator("conv2d", PRECISION(kInt8));
  conv_int8_fp32.CreateOperator("conv2d", PRECISION(kInt8));

  /// set param and context
  for (auto& dim_in : input_dims) {
    param_int8_out.x->Resize(dim_in);
    auto out_tmp_dims = compute_out_dim(dim_in, param_int8_out);
    if (out_tmp_dims[2] < 1 || out_tmp_dims[3] < 1) {
      continue;
    }
    param_fp32_out.x->Resize(dim_in);
    param_int8_out.output->Resize(out_tmp_dims);
    param_fp32_out.output->Resize(out_tmp_dims);
    break;
  }
  conv_int8_int8.SetParam(&param_int8_out);
  conv_int8_fp32.SetParam(&param_fp32_out);

  for (auto& dim_in : input_dims) {
    if (weight_dim[1] * group != dim_in[1]) {
      "input channel must equal to weights channel\n";
      assert(0);
    }
    auto dim_out = compute_out_dim(dim_in, param_int8_out);
    if (dim_out[2] < 1 || dim_out[3] < 1) {
      continue;
    }
    param_fp32_out.output->ReleaseRawTensor();
    delete param_fp32_out.output;
    param_fp32_out.output = new Tensor;
    param_fp32_out.output->set_precision(PRECISION(kFloat));
    param_int8_out.output->ReleaseRawTensor();
    delete param_int8_out.output;
    param_int8_out.output = new Tensor;
    param_int8_out.output->set_precision(PRECISION(kInt8));

    param_int8_out.x->Resize(dim_in);
    param_int8_out.output->Resize(dim_out);
    param_fp32_out.x->Resize(dim_in);
    param_fp32_out.output->Resize(dim_out);

    Tensor tin_fp32;
    tin_fp32.Resize(dim_in);
    tin_fp32.set_precision(PRECISION(kFloat));
    Tensor tout_basic_fp32;
    Tensor tout_basic_int8;

    fill_tensor_rand(*param_int8_out.x, -127, 127);
    param_fp32_out.x->CopyFromCpu<int8_t>(param_int8_out.x->data<int8_t>());

    ComputeUtils::TensorInt8ToFloat(*param_int8_out.x, tin_fp32, scale_in[0]);

    if (check_result) {
      tout_basic_fp32.set_precision(PRECISION(kFloat));
      tout_basic_fp32.Resize(dim_out);
      tout_basic_int8.set_precision(PRECISION(kInt8));
      tout_basic_int8.Resize(dim_out);
      fill_tensor_const(tout_basic_fp32, 0.f);
      auto dout_basic_fp32 = tout_basic_fp32.mutable_data<float>();
      auto dout_basic_int8 = tout_basic_int8.mutable_data<int8_t>();
      const float* din_fp32 = tin_fp32.data<float>();
      conv_basic<float, float>(din_fp32,
                               dout_basic_fp32,
                               dim_in[0],
                               dim_out[1],
                               dim_out[2],
                               dim_out[3],
                               dim_in[1],
                               dim_in[2],
                               dim_in[3],
                               wptr_fp32,
                               bptr_fp32,
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
                               static_cast<int>(flag_relu));
      ComputeUtils::TensorFloatToInt8(
          tout_basic_fp32, tout_basic_int8, scale_out[0]);
    }

    double gops = 2.0 * dim_production(tout_basic_int8) * dim_in[1] *
                  weight_dim[2] * weight_dim[3] / group;
    /// warm up
    for (int i = 0; i < warmup; ++i) {
      conv_int8_int8.Launch();
    }
    /// compute fp32 output
    Timer t0;
    for (int i = 0; i < repeats; ++i) {
      t0.Start();
      conv_int8_fp32.Launch();
      t0.Stop();
    }
    std::cout << "int8 conv, fp32 output: output shape: (" << dim_out[0] << ", "
              << dim_out[1] << ", " << dim_out[2] << ", " << dim_out[3]
              << "), running time, avg: " << t0.LapTimes().Avg()
              << ", min time: " << t0.LapTimes().Min()
              << ", total GOPS: " << 1e-9 * gops
              << " GOPS, avg GOPs: " << 1e-6 * gops / t0.LapTimes().Avg()
              << " GOPs, max GOPs: " << 1e-6 * gops / t0.LapTimes().Min()
              << std::endl;

    /// compute int8 output
    t0.Reset();
    for (int i = 0; i < repeats; ++i) {
      t0.Start();
      conv_int8_int8.Launch();
      t0.Stop();
    }
    std::cout << "int8 conv, int8 output: output shape: (" << dim_out[0] << ", "
              << dim_out[1] << ", " << dim_out[2] << ", " << dim_out[3]
              << "), running time, avg: " << t0.LapTimes().Avg()
              << ", min time: " << t0.LapTimes().Min()
              << ", total GOPS: " << 1e-9 * gops
              << " GOPS, avg GOPs: " << 1e-6 * gops / t0.LapTimes().Avg()
              << " GOPs, max GOPs: " << 1e-6 * gops / t0.LapTimes().Min()
              << std::endl;

    /// compare result fp32 output
    if (check_result) {
      double max_ratio = 0;
      double max_diff = 0;
      tensor_cmp_host(
          tout_basic_fp32, *param_fp32_out.output, max_ratio, max_diff);
      std::cout << "FP32 compare result, max diff: " << max_diff
                << ", max ratio: " << max_ratio << std::endl;
      if (std::abs(max_ratio) > 1e-5f) {
        if (max_diff > 5e-5f) {
          std::cout << "basic result\n";
          print_tensor(tout_basic_fp32);
          std::cout << "lite result\n";
          print_tensor(*param_fp32_out.output);
          Tensor tdiff;
          tdiff.Resize(tout_basic_fp32.shape());
          tdiff.set_precision(PRECISION(kFloat));
          tensor_diff(tout_basic_fp32, *param_fp32_out.output, tdiff);
          print_tensor(tdiff);
          release_param(&param_int8_out);
          release_param(&param_fp32_out);
          std::cerr << "test int8 conv, fp32 out: input: (" << dim_in[0] << ", "
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
                    << ", relu: " << (flag_relu ? "true" : "false")
                    << ", threads: " << thread_num
                    << ", power_mode: " << power_mode << " failed!!\n";
          exit(1);
        }
      }
    }
    /// compare result int8 output
    if (check_result) {
      double max_ratio = 0;
      double max_diff = 0;
      // ! int8
      tensor_cmp_host(
          tout_basic_int8, *param_int8_out.output, max_ratio, max_diff);
      std::cout << "int8 compare result, max diff: " << max_diff
                << ", max ratio: " << max_ratio << std::endl;
      if (fabs(max_diff) > 0) {
        Tensor tdiff;
        tdiff.Resize(tout_basic_int8.shape());
        tdiff.set_precision(PRECISION(kInt8));
        tensor_diff(tout_basic_int8, *param_int8_out.output, tdiff);
        auto ptr = tdiff.data<int8_t>();
        auto ptr_basic_fp32 = tout_basic_fp32.data<float>();
        float count = 0;
        bool check = true;
        for (int i = 0; i < dim_production(tdiff); ++i) {
          if (abs(ptr[i]) > 1) {
            check = false;
            std::cerr << "basic float data: " << ptr_basic_fp32[i]
                      << ", after scale: " << ptr_basic_fp32[i] / scale_out[0]
                      << std::endl;
            break;
          }
          if (ptr[i] != 0) {
            std::cerr << "basic float data: " << ptr_basic_fp32[i]
                      << ", after scale: " << ptr_basic_fp32[i] / scale_out[0]
                      << std::endl;
            count += 1;
          }
        }
        check = check &&
                count < std::max(
                            10, static_cast<int>(0.01 * dim_production(tdiff)));
        if (!check) {
          std::cout << "int8 basic result\n";
          print_tensor(tout_basic_int8);
          std::cout << "int8 lite result\n";
          print_tensor(*param_int8_out.output);
          std::cout << "int8 diff tensor\n";
          print_tensor(tdiff);
          release_param(&param_int8_out);
          release_param(&param_fp32_out);
          std::cerr << "test int8 conv, fp32 out: input: (" << dim_in[0] << ", "
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
                    << ", relu: " << (flag_relu ? "true" : "false")
                    << ", threads: " << thread_num
                    << ", power_mode: " << power_mode << " failed!!\n";
          exit(1);
        }
      }
    }
    std::cout << "test int8 conv: input: (" << dim_in[0] << ", " << dim_in[1]
              << ", " << dim_in[2] << ", " << dim_in[3] << "), output: ("
              << dim_out[0] << ", " << dim_out[1] << ", " << dim_out[2] << ", "
              << dim_out[3] << "), weight dim: (" << weight_dim[0] << ", "
              << weight_dim[1] << ", " << weight_dim[2] << ", " << weight_dim[3]
              << "), pad: " << pads[0] << ", " << pads[1] << ", " << pads[2]
              << ", " << pads[3] << ", stride: " << strides[0] << ", "
              << strides[1] << ", dila_: " << dilas[0] << ", " << dilas[1]
              << ", bias: " << (flag_bias ? "true" : "false")
              << ", relu: " << (flag_relu ? "true" : "false")
              << ", threads: " << thread_num << ", power_mode: " << power_mode
              << " successed!!\n";
  }
  release_param(&param_int8_out);
  release_param(&param_fp32_out);
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
    flag_relu = atoi(argv[13]);
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
                                test_conv_int8(
                                    dims,
                                    weights_dim,
                                    g,
                                    {stride, stride},
                                    {pad_top, pad_bottom, pad_left, pad_right},
                                    {dila, dila},
                                    flag_bias,
                                    flag_relu,
                                    threads,
                                    3);
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
  test_conv_int8(dims,
                 weights_dim,
                 group,
                 {stride_h, stride_w},
                 {pad_h0, pad_h1, pad_w0, pad_w1},
                 {dila_h, dila_w},
                 flag_bias,
                 flag_relu,
                 threads,
                 3);
  std::cout << "RUN CUSTOM TEST END: " << std::endl;
  return 0;
}
