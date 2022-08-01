// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef LITE_WITH_X86

#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include <vector>
#include "lite/core/context.h"
#include "lite/core/profile/timer.h"
#include "lite/kernels/x86/conv_compute.h"
#include "lite/operators/op_params.h"
#include "lite/tests/utils/naive_math_impl.h"
#include "lite/tests/utils/tensor_utils.h"

typedef paddle::lite::Tensor Tensor;
using paddle::lite::profile::Timer;
typedef paddle::lite::operators::ActivationParam ActivationParam;
typedef paddle::lite::operators::ConvParam ConvParam;

typedef struct TestConvParam {
  int ic;
  int oc;
  int kw;
  int kh;
  int iw;
  int ih;
  int ow;
  int oh;
  int stride_w;
  int stride_h;
  int dil_w;
  int dil_h;
  int pad_w;
  int pad_h;
  bool has_bias;
  int relu_type;
  float alpha;
  int group;
  int num;
} TestConvParam_t;

#define TEST_INIT_PARAM            \
  int ic = param.ic;               \
  int oc = param.oc;               \
  int kw = param.kw;               \
  int kh = param.kh;               \
  int iw = param.iw;               \
  int ih = param.ih;               \
  int ow = param.ow;               \
  int oh = param.oh;               \
  int stride_w = param.stride_w;   \
  int stride_h = param.stride_h;   \
  int dil_w = param.dil_w;         \
  int dil_h = param.dil_h;         \
  int pad_w = param.pad_w;         \
  int pad_h = param.pad_h;         \
  bool has_bias = param.has_bias;  \
  int relu_type = param.relu_type; \
  int group = param.group;         \
  int num = param.num;

bool test_conv_int8(TestConvParam_t param) {
  TEST_INIT_PARAM;
  Tensor weight, input, output, w_f32, i_f32, o_basic_f32, o_basic_s8;
  Tensor tbias;
  weight.Resize({oc, ic, kh, kw});
  input.Resize({num, ic, ih, iw});
  output.Resize({num, oc, oh, ow});
  w_f32.Resize({oc, ic, kh, kw});
  i_f32.Resize({num, ic, ih, iw});
  o_basic_f32.Resize({num, oc, oh, ow});
  o_basic_s8.Resize({num, oc, oh, ow});
  tbias.Resize({oc});
  weight.set_precision(PRECISION(kInt8));
  input.set_precision(PRECISION(kInt8));
  output.set_precision(PRECISION(kInt8));
  w_f32.set_precision(PRECISION(kFloat));
  i_f32.set_precision(PRECISION(kFloat));
  o_basic_f32.set_precision(PRECISION(kFloat));
  o_basic_s8.set_precision(PRECISION(kInt8));
  tbias.set_precision(PRECISION(kFloat));

  // int8 input
  fill_tensor_rand(weight, -5, 5);
  fill_tensor_rand(input, -5, 5);
  if (has_bias)
    fill_tensor_rand(tbias, -1.f, 1.f);
  else
    fill_tensor_rand(tbias, 0, 0);
  // Scale
  std::vector<float> Sw(oc, 1.f / 64);
  float Si = 1 / 127.f;
  float So = 1 / 127.f;
  // float input
  auto a_ptr_f32 = w_f32.mutable_data<float>();
  for (int i = 0; i < oc; i++) {
    float ssa = Sw[i];
    for (int j = 0; j < ic * kh * kw; j++) {
      int offt = i * ic * kh * kw + j;
      a_ptr_f32[offt] = (weight.data<int8_t>())[offt] * ssa;
    }
  }
  auto b_ptr_f32 = i_f32.mutable_data<float>();
  for (int i = 0; i < num * ic * ih * iw; i++)
    b_ptr_f32[i] = (input.data<int8_t>())[i] * Si;
  auto output_ptr_test = output.mutable_data<int8_t>();
  auto output_ptr_basic = o_basic_s8.mutable_data<int8_t>();
  auto output_ptr_basic_f = o_basic_f32.mutable_data<float>();

  // 1-relu 2-relu6 4-leakyrelu
  memset(output_ptr_basic_f, 0, num * oc * oh * ow * sizeof(float));
  conv_basic<float, float>(i_f32.data<float>(),
                           output_ptr_basic_f,
                           num,
                           oc,
                           oh,
                           ow,
                           ic,
                           ih,
                           iw,
                           w_f32.data<float>(),
                           tbias.data<float>(),
                           group,
                           kw,
                           kh,
                           stride_w,
                           stride_h,
                           dil_w,
                           dil_h,
                           pad_w,
                           pad_h,
                           has_bias,
                           relu_type,
                           6.f,
                           1.f);
  for (int i = 0; i < num * oc * ow * oh; i++) {
    int tmp = static_cast<int>(output_ptr_basic_f[i] / So > 0
                                   ? output_ptr_basic_f[i] / So + 0.5f
                                   : output_ptr_basic_f[i] / So - 0.5f);
    output_ptr_basic[i] =
        static_cast<int8_t>(std::min(std::max(tmp, -127), 127));
  }

  // set conv param
  paddle::lite::operators::ConvParam conv_param{};
  conv_param.x = &input;
  conv_param.filter = &weight;
  conv_param.bias = &tbias;
  std::vector<int> stridess;
  stridess.push_back(stride_h);
  stridess.push_back(stride_w);
  conv_param.strides = stridess;
  std::vector<int> dilass;
  dilass.push_back(dil_h);
  dilass.push_back(dil_w);
  conv_param.dilations = std::make_shared<std::vector<int>>(dilass);
  std::vector<int> padss;
  padss.push_back(pad_h);
  padss.push_back(pad_h);
  padss.push_back(pad_w);
  padss.push_back(pad_w);
  conv_param.paddings = std::make_shared<std::vector<int>>(padss);
  conv_param.fuse_relu = true;
  conv_param.groups = group;
  conv_param.output = &output;
  ActivationParam act_param;
  act_param.has_active = true;
  int s8_relu_type = (relu_type == 4) ? 3 : relu_type;
  act_param.active_type = (paddle::lite_api::ActivationType)
      s8_relu_type;  // 1-relu, 2-relu6, 3-leakyrelu
  act_param.Relu_clipped_coef = 6.f;
  act_param.Leaky_relu_alpha = 1.f;
  conv_param.activation_param = act_param;
  conv_param.input_scale = Si;
  conv_param.output_scale = So;
  conv_param.weight_scale = Sw;
  conv_param.enable_int8 = true;

  std::unique_ptr<paddle::lite::KernelContext> ctx1(
      new paddle::lite::KernelContext);
  paddle::lite::kernels::x86::Conv2dCompute<PRECISION(kInt8), PRECISION(kInt8)>
      conv_int8_int8;
  conv_int8_int8.SetContext(std::move(ctx1));
  conv_int8_int8.SetParam(conv_param);
  conv_int8_int8.PrepareForRun();
  conv_int8_int8.Launch();

  int max_err = 0;
  int8_t lr = 0, rr = 0;
  for (int i = 0; i < num * oc * oh * ow; i++) {
    if (abs(output_ptr_basic[i] - output_ptr_test[i]) > max_err) {
      max_err = abs(output_ptr_basic[i] - output_ptr_test[i]);
      lr = output_ptr_basic[i];
      rr = output_ptr_test[i];
    }
  }
  if (max_err > 1) {
    LOG(INFO) << "max_err = " << max_err << " (real=" << static_cast<int>(lr)
              << ", fake=" << static_cast<int>(rr) << ")"
              << "\n";
    return false;
  }
  return true;
}

TEST(TestX86LiteConvInt8, conv_s8_compute) {
  for (auto ic : {16, 32})
    for (auto oc : {16, 32})
      for (auto kw : {1, 3})
        for (auto kh : {1, 3})
          for (auto iw : {28, 56})
            for (auto ih : {28, 56})
              for (auto bias : {false, true})
                for (auto relu_type : {0, 1})
                  for (auto dil : {1})
                    for (auto stride : {1})
                      for (auto pp : {0})
                        for (auto group : {1}) {
                          if ((kw == 1 || kh == 1) && (dil != 1)) continue;
                          if ((kw == 1 || kh == 1) && (pp != 0)) continue;

                          int pw = pp;
                          int ph = pp;
                          int extend_kw = dil * (kw - 1) + 1;
                          int extend_kh = dil * (kh - 1) + 1;
                          int ow = (iw - extend_kw + 2 * pw) / stride + 1;
                          int oh = (ih - extend_kh + 2 * ph) / stride + 1;
                          auto flag = test_conv_int8({ic,
                                                      oc,
                                                      kw,
                                                      kh,
                                                      iw,
                                                      ih,
                                                      ow,
                                                      oh,
                                                      stride,
                                                      stride,
                                                      dil,
                                                      dil,
                                                      pw,
                                                      ph,
                                                      bias,
                                                      relu_type,
                                                      6.f,
                                                      group,
                                                      1});
                          if (!flag) {
                            LOG(INFO) << "ic=" << ic << ", oc=" << oc
                                      << ", iw=" << iw << ", ih=" << ih
                                      << ", ow=" << ow << ", oh=" << oh
                                      << ", kw=" << kw << ", kh=" << kh
                                      << ", group=" << group << ", dil=" << dil
                                      << ", relu_type=" << relu_type
                                      << ", stride=" << stride << ", pad=" << pp
                                      << ", bias=" << bias;
                            LOG(FATAL) << "precision check failed.";
                          }
                        }
}

#endif  // LITE_WITH_X86
