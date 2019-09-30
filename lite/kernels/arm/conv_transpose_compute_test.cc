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

#include "lite/kernels/arm/conv_transpose_compute.h"
#include <gtest/gtest.h>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

template <typename type, typename type2>
static void basic_gemm(int m,
                       int n,
                       int k,
                       const type* a,
                       const type* b,
                       const type2* bias,
                       type2* c,
                       type2 alpha,
                       type2 beta,
                       bool trans_a = false,
                       bool trans_b = false,
                       bool flag_bias = false,
                       bool flag_relu = false) {
#pragma omp parallel for
  for (int i = 0; i < m; ++i) {
    type2 bias_data = (type2)0;
    if (flag_bias) {
      bias_data = bias[i];
    }
    for (int j = 0; j < n; ++j) {
      type2 sum = static_cast<type2>(0);
      for (int l = 0; l < k; ++l) {
        type av;
        type bv;
        if (trans_a) {
          av = a[l * m + i];
        } else {
          av = a[i * k + l];
        }
        if (trans_b) {
          bv = b[j * k + l];
        } else {
          bv = b[l * n + j];
        }
        sum += av * bv;
      }
      type2 tmp = alpha * sum + beta * c[i * n + j] + bias_data;
      if (flag_relu) {
        c[i * n + j] = tmp > (type2)0 ? tmp : (type2)0;
      } else {
        c[i * n + j] = tmp;
      }
    }
  }
}

//! for float, dtype1 and type2 is float
//! for int8, dytpe1 is char, dtype2 is int
template <typename Dtype1, typename Dtype2>
bool deconv_basic(const Dtype1* din,
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
                  bool flag_relu) {
  int m = chout * kernel_w * kernel_h / group;
  int n = hin * win;
  int k = chin / group;

  if (chin != chout || group != chin) {
    CHECK_OR_FALSE(chin % group == 0);
    CHECK_OR_FALSE(chout % group == 0);
  }

  lite::Tensor workspace_tensor;
  std::vector<int64_t> wt_shape = {1, 1, 1, group * m * n};
  workspace_tensor.Resize(wt_shape);
  auto* workspace_ptr = workspace_tensor.mutable_data<Dtype2>();

  int group_size_in = win * hin * chin / group;
  int group_size_out = wout * hout * chout / group;
  int group_size_coldata = m * n;
  int group_size_weights = chin * chout * kernel_w * kernel_h / (group * group);
  bool flag_1x1s1p1 = (kernel_w == 1) && (kernel_h == 1) && (stride_h == 1) &&
                      (stride_w == 1) && (pad_w == 1) && (pad_h == 1) &&
                      (dila_w == 1) && (dila_h == 1);

  for (int i = 0; i < num; ++i) {
    const Dtype1* din_batch = din + i * chin * hin * win;
    Dtype2* dout_batch = dout + i * chout * hout * wout;

    Dtype2* col_data = workspace_ptr;
    if (flag_1x1s1p1) {
      col_data = dout_batch;
    }
    memset(col_data, 0, sizeof(Dtype2) * group_size_coldata);
    for (int g = 0; g < group; ++g) {
      const Dtype1* din_group = din_batch + g * group_size_in;
      const Dtype1* weights_group = weights + g * group_size_weights;
      Dtype2* coldata_group = col_data + g * group_size_coldata;
      basic_gemm<Dtype1, Dtype2>(m,
                                 n,
                                 k,
                                 weights_group,
                                 din_group,
                                 nullptr,
                                 coldata_group,
                                 (Dtype2)1,
                                 (Dtype2)0,
                                 true,
                                 false,
                                 false,
                                 (!flag_bias && flag_relu));
    }
    if (!flag_1x1s1p1) {
      lite::arm::math::col2im(col_data,
                              chout,
                              hout,
                              wout,
                              kernel_h,
                              kernel_w,
                              pad_h,
                              pad_w,
                              stride_h,
                              stride_w,
                              dila_h,
                              dila_w,
                              dout_batch);
    }
    if (flag_bias) {
      lite::arm::math::fill_bias_relu(
          dout_batch, bias, chout, wout * hout, flag_bias, flag_relu);
    }
  }
  return true;
}

template <typename Dtype1, typename Dtype2>
void conv2d_transpose_compute_ref(const operators::ConvParam& param) {
  const Dtype1* din = param.x->data<Dtype1>();
  Dtype2* dout = param.output->mutable_data<Dtype2>();

  int num = param.x->dims()[0];
  int chout = param.output->dims()[1];
  int hout = param.output->dims()[2];
  int wout = param.output->dims()[3];

  int chin = param.x->dims()[1];
  int hin = param.x->dims()[2];
  int win = param.x->dims()[3];

  const Dtype1* weights = param.filter->mutable_data<Dtype1>();
  Dtype2* bias = nullptr;
  if (param.bias != nullptr) {
    bias = param.bias->mutable_data<Dtype2>();
  }

  int group = param.groups;
  int kernel_h = param.filter->dims()[2];
  int kernel_w = param.filter->dims()[3];
  int stride_h = param.strides[0];
  int stride_w = param.strides[1];
  int dila_h = param.dilations[0];
  int dila_w = param.dilations[1];

  int pad_h = param.paddings[0];
  int pad_w = param.paddings[1];
  bool flag_bias = (param.bias != nullptr);
  bool flag_relu = param.fuse_relu;

  deconv_basic<float, float>(din,
                             dout,
                             num,
                             chout,
                             hout,
                             wout,
                             chin,
                             hin,
                             win,
                             weights,
                             bias,
                             group,
                             kernel_w,
                             kernel_h,
                             stride_w,
                             stride_h,
                             dila_w,
                             dila_h,
                             pad_w,
                             pad_h,
                             flag_bias,
                             flag_relu);
}

TEST(conv2d_transpose_arm, retrive_op) {
  auto op = KernelRegistry::Global().Create<TARGET(kARM), PRECISION(kFloat)>(
      "conv2d_transpose");
  ASSERT_FALSE(op.empty());
  ASSERT_TRUE(op.front());
}

TEST(conv2d_transpose_arm, init) {
  Conv2DTransposeCompute compute;
  ASSERT_EQ(compute.precision(), PRECISION(kFloat));
  ASSERT_EQ(compute.target(), TARGET(kARM));
}

TEST(conv2d_transpose_arm, compute) {
  DeviceInfo::Init();
  for (auto n : {1, 2}) {
    for (auto ic : {1, 3 /*, 128*/}) {
      for (auto oc : {1, 3 /*, 128*/}) {
        for (auto ih : {2, 8 /*, 56 , 112, 224, 512*/}) {
          for (auto iw : {2, 8 /*, 56, 112, 224, 512*/}) {
            for (auto flag_bias : {false, true}) {
              for (auto flag_relu : {false, true}) {
                for (auto dilation : {1, 2}) {
                  for (auto stride : {1, 2}) {
                    for (auto padding : {0, 1, 2}) {
                      for (auto ks : {2, 3, 5}) {
                        for (auto group : {1, 2}) {
                          // obtain shape
                          if (ic % group != 0 || oc % group != 0) {
                            group = 1;
                          }
                          std::vector<int64_t> input_shape = {n, ic, ih, iw};
                          std::vector<int64_t> filter_shape = {
                              oc / group, ic, ks, ks};
                          int oh = (ih - 1) * stride - 2 * padding +
                                   dilation * (ks - 1) + 1;
                          int ow = (iw - 1) * stride - 2 * padding +
                                   dilation * (ks - 1) + 1;
                          if (oh < 1 || ow < 1) {
                            break;
                          }
                          std::vector<int64_t> output_shape = {n, oc, oh, ow};
                          std::vector<int64_t> bias_shape = {1, oc, 1, 1};

                          // define and resize tensor
                          Tensor input;
                          Tensor filter;
                          Tensor filter_copy;
                          Tensor bias;
                          Tensor output;
                          Tensor output_ref;
                          input.Resize(input_shape);
                          filter.Resize(filter_shape);
                          filter_copy.Resize(filter_shape);
                          output.Resize(output_shape);
                          output_ref.Resize(output_shape);
                          auto* input_data = input.mutable_data<float>();
                          auto* filter_data = filter.mutable_data<float>();
                          auto* filter_copy_data =
                              filter_copy.mutable_data<float>();
                          auto* output_data = output.mutable_data<float>();

                          // initialize tensor
                          for (int i = 0; i < input.dims().production(); i++) {
                            float sign = i % 3 == 0 ? -1.0f : 1.0f;
                            input_data[i] = sign * static_cast<float>(i % 128);
                          }
                          for (int i = 0; i < filter.dims().production(); i++) {
                            filter_data[i] =
                                i /
                                static_cast<float>(filter.dims().production());
                            filter_copy_data[i] =
                                i / static_cast<float>(
                                        filter_copy.dims().production());
                          }
                          if (flag_bias) {
                            bias.Resize(bias_shape);
                            auto* bias_data = bias.mutable_data<float>();
                            for (int i = 0; i < bias.dims().production(); i++) {
                              bias_data[i] = static_cast<float>(i);
                            }
                          }

                          // prepare kernel params and run
                          std::unique_ptr<KernelContext> ctx(new KernelContext);
                          ctx->As<ARMContext>();
                          Conv2DTransposeCompute conv2d_transpose;
                          conv2d_transpose.SetContext(std::move(ctx));
                          operators::ConvParam param;
                          param.x = &input;
                          param.filter = &filter;
                          param.output = &output;
                          param.bias = nullptr;
                          if (flag_bias) {
                            bias.Resize(bias_shape);
                            auto* bias_data = bias.mutable_data<float>();
                            for (int i = 0; i < bias.dims().production(); i++) {
                              bias_data[i] = static_cast<float>(i);
                            }
                            param.bias = &bias;
                          }
                          param.fuse_relu = flag_relu;
                          param.paddings = std::vector<int>({padding, padding});
                          param.strides = std::vector<int>({stride, stride});
                          param.dilations =
                              std::vector<int>({dilation, dilation});
                          param.groups = group;
                          conv2d_transpose.SetParam(param);
                          conv2d_transpose.Launch();

                          // invoking ref implementation and compare results
                          param.filter = &filter_copy;
                          param.output = &output_ref;
                          conv2d_transpose_compute_ref<float, float>(param);
                          auto* output_ref_data =
                              output_ref.mutable_data<float>();
                          for (int i = 0; i < output.dims().production(); i++) {
                            EXPECT_NEAR(
                                output_data[i], output_ref_data[i], 1e-3);
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

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
USE_LITE_KERNEL(conv2d_transpose, kARM, kFloat, kNCHW, def);
