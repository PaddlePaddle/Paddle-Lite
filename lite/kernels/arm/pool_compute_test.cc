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

#include <gtest/gtest.h>

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "lite/backends/arm/math/funcs.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/arm/pool_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

int PoolOutputSize(int input_size,
                   int filter_size,
                   int pad_left,
                   int pad_right,
                   int stride,
                   bool ceil_mode) {
  int output_size;
  if (!ceil_mode) {
    output_size =
        (input_size - filter_size + pad_left + pad_right) / stride + 1;
  } else {
    output_size =
        (input_size - filter_size + pad_left + pad_right + stride - 1) /
            stride +
        1;
  }
  return output_size;
}

std::vector<int64_t> compute_output_shape(operators::PoolParam* param_) {
  const auto x_dims = param_->x->dims();
  std::vector<int>& ksize = param_->ksize;
  auto paddings = *param_->paddings;
  if (param_->global_pooling) {
    ksize.resize(static_cast<size_t>(x_dims.size()) - 2);
    for (size_t i = 0; i < ksize.size(); ++i) {
      paddings[2 * i] = 0;
      paddings[2 * i + 1] = 0;
      ksize[i] = static_cast<int>(x_dims[i + 2]);
    }
  }

  std::vector<int64_t> output_shape({x_dims[0], x_dims[1]});
  if (param_->adaptive) {
    output_shape.insert(
        output_shape.end(), param_->ksize.begin(), param_->ksize.end());
  } else {
    for (size_t i = 0; i < param_->ksize.size(); ++i) {
      output_shape.push_back(PoolOutputSize(x_dims[i + 2],
                                            param_->ksize[i],
                                            paddings[2 * i],
                                            paddings[2 * i + 1],
                                            param_->strides[i],
                                            param_->ceil_mode));
    }
  }
  return output_shape;
}

void pool_compute_ref(const operators::PoolParam& param) {
  auto& in_dims = param.x->dims();
  auto& out_dims = param.output->dims();

  const float* din = param.x->data<const float>();
  float* dout = param.output->mutable_data<float>();

  std::vector<int> ksize = param.ksize;
  std::vector<int> strides = param.strides;
  std::vector<int> paddings = *param.paddings;

  std::string pooling_type = param.pooling_type;
  bool global_pooling = param.global_pooling;
  bool exclusive = param.exclusive;
  bool adaptive = param.adaptive;
  bool ceil_mode = param.ceil_mode;
  bool use_quantizer = param.use_quantizer;
  std::string data_format = param.data_format;

  int num = in_dims[0];
  int chin = in_dims[1];
  int hin = in_dims[2];
  int win = in_dims[3];

  int chout = out_dims[1];
  int hout = out_dims[2];
  int wout = out_dims[3];

  // no need to pad input tensor, border is zero pad inside this function
  memset(dout, 0, num * chout * hout * wout * sizeof(float));
  int kernel_h = ksize[0];
  int kernel_w = ksize[1];
  int stride_h = strides[0];
  int stride_w = strides[1];
  int pad_h = paddings[0];
  int pad_w = paddings[2];
  int size_channel_in = win * hin;
  int size_channel_out = wout * hout;
  if (global_pooling) {
    if (pooling_type == "max") {  // Pooling_max
      for (int n = 0; n < num; ++n) {
        float* dout_batch = dout + n * chout * size_channel_out;
        const float* din_batch = din + n * chin * size_channel_in;
#pragma omp parallel for
        for (int c = 0; c < chout; ++c) {
          const float* din_ch = din_batch + c * size_channel_in;  // in address
          float tmp1 = din_ch[0];
          for (int i = 0; i < size_channel_in; ++i) {
            float tmp2 = din_ch[i];
            tmp1 = tmp1 > tmp2 ? tmp1 : tmp2;
          }
          dout_batch[c] = tmp1;
        }
      }
    } else if (pooling_type == "avg") {
      // Pooling_average_include_padding
      // Pooling_average_exclude_padding
      for (int n = 0; n < num; ++n) {
        float* dout_batch = dout + n * chout * size_channel_out;
        const float* din_batch = din + n * chin * size_channel_in;
#pragma omp parallel for
        for (int c = 0; c < chout; ++c) {
          const float* din_ch = din_batch + c * size_channel_in;  // in address
          float sum = 0.f;
          for (int i = 0; i < size_channel_in; ++i) {
            sum += din_ch[i];
          }
          dout_batch[c] = sum / size_channel_in;
        }
      }
    } else {
      LOG(FATAL) << "unsupported pooling type: " << pooling_type;
    }
  } else {
    for (int ind_n = 0; ind_n < num; ++ind_n) {
#pragma omp parallel for
      for (int ind_c = 0; ind_c < chin; ++ind_c) {
        for (int ind_h = 0; ind_h < hout; ++ind_h) {
          int sh = ind_h * stride_h;
          int eh = sh + kernel_h;
          sh = (sh - pad_h) < 0 ? 0 : sh - pad_h;
          eh = (eh - pad_h) > hin ? hin : eh - pad_h;
          for (int ind_w = 0; ind_w < wout; ++ind_w) {
            int sw = ind_w * stride_w;
            int ew = sw + kernel_w;
            sw = (sw - pad_w) < 0 ? 0 : sw - pad_w;
            ew = (ew - pad_w) > win ? win : ew - pad_w;
            float result = static_cast<float>(0);
            int dst_ind = (ind_n * chout + ind_c) * size_channel_out +
                          ind_h * wout + ind_w;
            for (int kh = sh; kh < eh; ++kh) {
              for (int kw = sw; kw < ew; ++kw) {
                int src_ind =
                    (ind_n * chin + ind_c) * size_channel_in + kh * win + kw;
                if (kh == sh && kw == sw) {
                  result = din[src_ind];
                } else {
                  if (pooling_type == "max") {
                    result = result >= din[src_ind] ? result : din[src_ind];
                  } else if (pooling_type == "avg") {
                    result += din[src_ind];
                  }
                }
              }
            }
            if (pooling_type == "avg") {
              if (exclusive) {
                int div = (ew - sw) * (eh - sh);
                div = div > 0 ? div : 1;
                result /= div;
              } else {
                int bh = kernel_h;
                int bw = kernel_w;
                if (ew == win) {
                  bw = (sw + kernel_w) >= (win + paddings[3])
                           ? (win + paddings[3])
                           : (sw + kernel_w);
                  bw -= sw;
                  if ((sw - pad_w) < 0 &&
                      (sw + kernel_w) > (win + paddings[3])) {
                    bw += pad_w;
                  }
                }
                if (eh == hin) {
                  bh = (sh + kernel_h) >= (hin + paddings[1])
                           ? (hin + paddings[1])
                           : (sh + kernel_h);
                  bh -= sh;
                  if ((sh - pad_h) < 0 &&
                      (sh + kernel_h) > (hin + paddings[1])) {
                    bh += pad_h;
                  }
                }
                result /= bh * bw;
              }
            }
            dout[dst_ind] = result;
          }
        }
      }
    }
  }
}

TEST(pool_arm, init) {
  PoolCompute pool;
  ASSERT_EQ(pool.precision(), PRECISION(kFloat));
  ASSERT_EQ(pool.target(), TARGET(kARM));
}

TEST(pool_arm, compute) {
  PoolCompute pool;
  operators::PoolParam param;

  lite::Tensor x;
  lite::Tensor output;
  lite::Tensor output_ref;
#if 0
  // speedup for ci
  for (auto pooling_type : {"max", "avg"}) {
    for (auto ceil_mode : {true, false}) {
      for (auto global_pooling : {true, false}) {
        for (auto exclusive : {true, false}) {
          for (auto ksize : {2, 3}) {
            for (auto stride : {1, 2}) {
              for (auto pad_left : {0, 1}) {
                for (auto pad_right : {0, 1}) {
                  for (auto pad_top : {0, 1}) {
                    for (auto pad_bottom : {0, 1}) {
                      for (auto n : {1, 2}) {
                        for (auto c : {1, 3}) {
#if 1
                          for (auto h : {2, 3, 4, 11}) {
                            for (auto w : {2, 3, 4, 11}) {
#else
                          for (int h = 2; h < 25; h++) {
                            for (int w = 2; w < 25; w++) {
#endif
                              VLOG(3) << "n:" << n << " c:" << c << " h:" << h
                                      << " w:" << w << " ksize:" << ksize
                                      << " stride:" << stride
                                      << " pad_left:" << pad_left
                                      << " pad_right:" << pad_right
                                      << " pad_top:" << pad_top
                                      << " pad_bottom:" << pad_bottom
                                      << " exclusive:" << exclusive
                                      << " global_pooling:" << global_pooling
                                      << " ceil_mode: " << ceil_mode
                                      << " pooling_type:" << pooling_type;

                              // init x, output
                              x.Resize(
                                  DDim(std::vector<int64_t>({n, c, h, w})));
                              auto* x_data = x.mutable_data<float>();
                              for (int i = 0; i < x.dims().production(); ++i) {
                                float sign = i % 3 == 0 ? -0.03 : 0.05f;
                                x_data[i] = sign * (i % 128);
                              }

                              // fill param
                              param.x = &x;
                              param.output = &output;
                              param.pooling_type = pooling_type;
                              if (global_pooling) {
                                param.ksize = {h, w};
                              } else {
                                param.ksize = {ksize, ksize};
                              }
                              param.global_pooling = global_pooling;
                              param.strides = {stride, stride};
                              std::vector<int> paddings = {
                                  pad_top, pad_bottom, pad_left, pad_right};
                              param.exclusive = exclusive;
                              param.paddings =
                                  std::make_shared<std::vector<int>>(paddings);
                              param.ceil_mode = ceil_mode;
                              param.adaptive = false;
                              param.use_quantizer = false;

                              const std::vector<int64_t>& output_shape =
                                  compute_output_shape(&param);
                              output.Resize(DDim(output_shape));
                              output_ref.Resize(DDim(output_shape));

                              auto* output_data = output.mutable_data<float>();
                              auto* output_ref_data =
                                  output_ref.mutable_data<float>();
                              for (int i = 0; i < output.dims().production();
                                   ++i) {
                                output_data[i] = -2;
                                output_ref_data[i] = -2;
                              }

                              // compute
                              pool.SetParam(param);
                              pool.Run();

                              // compute ref
                              param.output = &output_ref;
                              pool_compute_ref(param);

                              // compare
                              for (int i = 0; i < output.dims().production();
                                   i++) {
                                EXPECT_NEAR(
                                    output_data[i], output_ref_data[i], 1e-4);
                              }
                              VLOG(3) << "compare pass";
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
#endif
}

TEST(pool_arm, retrive_op) {
  auto pool = KernelRegistry::Global().Create("pool2d");
  ASSERT_FALSE(pool.empty());
  ASSERT_TRUE(pool.front());
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(pool2d, kARM, kFloat, kNCHW, def);
