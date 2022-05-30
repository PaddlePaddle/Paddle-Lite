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

#pragma once
#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include "lite/core/context.h"
#include "lite/core/profile/timer.h"
#include "lite/operators/op_params.h"
#include "lite/tests/utils/fill_data.h"
#include "lite/tests/utils/naive_math_impl.h"
#include "lite/tests/utils/print_info.h"

#ifdef LITE_WITH_ARM
#include "lite/kernels/arm/conv_compute.h"
#ifdef ENABLE_ARM_FP16
typedef __fp16 float16_t;
#endif
#else
#include "lite/kernels/x86/conv_compute.h"
#endif  // LITE_WITH_ARM

DEFINE_int32(power_mode,
             3,
             "power mode: "
             "0 for POWER_HIGH;"
             "1 for POWER_LOW;"
             "2 for POWER_FULL;"
             "3 for NO_BIND");
DEFINE_int32(threads, 1, "threads num");
DEFINE_int32(warmup, 0, "warmup times");
DEFINE_int32(repeats, 1, "repeats times");

#if defined(LITE_WITH_ARM)
DEFINE_bool(basic_test, true, "do all tests");
#else
DEFINE_bool(basic_test, false, "do all tests");
#endif
DEFINE_bool(check_result, true, "check the result");

DEFINE_int32(batch, 1, "batch size");
DEFINE_int32(in_channel, 32, "input channel");
DEFINE_int32(in_height, 112, "input height");
DEFINE_int32(in_width, 112, "input width");

DEFINE_int32(out_channel, 32, "output channel");
DEFINE_int32(group, 32, "group");
DEFINE_int32(kernel_h, 3, "kernel height");
DEFINE_int32(kernel_w, 3, "kernel width");
DEFINE_int32(pad_h0, 1, "pad top");
DEFINE_int32(pad_h1, 1, "pad bottom");
DEFINE_int32(pad_w0, 1, "pad left");
DEFINE_int32(pad_w1, 1, "pad right");
DEFINE_int32(stride_h, 1, "stride height");
DEFINE_int32(stride_w, 1, "stride width");
DEFINE_int32(dila_h, 1, "dilation height");
DEFINE_int32(dila_w, 1, "dilation width");

DEFINE_int32(flag_act,
             0,
             "do activation");  // 0-no act, 1-relu, 2-relu6, 4-leakyrelu
DEFINE_double(leakey_relu_alpha, 1.0, "leakey relu alpha");
DEFINE_bool(flag_bias, true, "with bias");

typedef paddle::lite::DDim DDim;
typedef paddle::lite::Tensor Tensor;
typedef paddle::lite::operators::ConvParam ConvParam;
typedef paddle::lite::operators::ActivationParam ActivationParam;

using paddle::lite::profile::Timer;

DDim compute_out_dim(const DDim& dim_in,
                     const paddle::lite::operators::ConvParam& param) {
  DDim dim_out = dim_in;
  auto paddings = *param.paddings;
  auto dilations = *param.dilations;
  dim_out[1] = param.filter->dims()[0];
  auto kernel_h = param.filter->dims()[2];
  auto kernel_w = param.filter->dims()[3];
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

void act_init(ConvParam& param,  // NOLINT
              std::vector<int> strides,
              std::vector<int> pads,
              std::vector<int> dilas,
              const int group,
              const int flag_act,
              const float six,
              const float leakey_relu_scale,
              const float scale = 6.f,
              const float offset = 3.f,
              const float threshold = 6.f) {
  param.strides = strides;
  param.paddings = std::make_shared<std::vector<int>>(pads);
  param.dilations = std::make_shared<std::vector<int>>(dilas);
  param.groups = group;
  if (flag_act > 0) {
    ActivationParam act_param;
    act_param.has_active = true;
    act_param.active_type = (paddle::lite_api::ActivationType)
        flag_act;  // 1-relu, 2-relu6, 4-leakyrelu
    if (flag_act == 1) {
      param.fuse_relu = true;
    } else if (flag_act == 2) {
      act_param.Relu_clipped_coef = six;
    } else if (flag_act == 4) {
      act_param.Leaky_relu_alpha = leakey_relu_scale;
    } else if (flag_act == 10) {
      act_param.hard_swish_scale = scale;
      act_param.hard_swish_offset = offset;
      act_param.hard_swish_threshold = threshold;
    }
    param.activation_param = act_param;
  }
}

void print_conv_success_or_fail_info(std::string op_name,
                                     bool has_success,
                                     const DDim dim_in,
                                     const DDim dim_out,
                                     const DDim weight_dim,
                                     std::vector<int> pads,
                                     std::vector<int> strides,
                                     std::vector<int> dilas,
                                     int group,
                                     bool flag_bias,
                                     int flag_act,
                                     int th,
                                     int cls) {
  if (has_success) {
    VLOG(4) << op_name << " input: " << dim_in << ", output: " << dim_out
            << ", weight dim: " << weight_dim << ", pad: " << pads[0] << ", "
            << pads[1] << ", " << pads[2] << ", " << pads[3]
            << ", stride: " << strides[0] << ", " << strides[1]
            << ", dila_: " << dilas[0] << ", " << dilas[1]
            << ", group: " << group
            << ", bias: " << (flag_bias ? "true" : "false")
            << ", act: " << flag_act << ", threads: " << th
            << ", power_mode: " << cls << " successed!!\n";
  } else {
    LOG(FATAL) << op_name << " input: " << dim_in << ", output: " << dim_out
               << ", weight dim: " << weight_dim << ", pad: " << pads[0] << ", "
               << pads[1] << ", " << pads[2] << ", " << pads[3]
               << ", stride: " << strides[0] << ", " << strides[1]
               << ", dila_: " << dilas[0] << ", " << dilas[1]
               << ", group: " << group
               << ", bias: " << (flag_bias ? "true" : "false")
               << ", act: " << flag_act << ", threads: " << th
               << ", power_mode: " << cls << " failed!!\n";
  }
}

template <typename T>
int ComputeSparseZeros(const Tensor* weights,
                       int* num_build_nonzeroes,
                       const int height,
                       const int width) {
  const T* data = weights->data<T>();
  int num_nonzeroes = 0;
  int num_nonzeroes_act = 0;
  for (int i = 0; i < height; i++) {
    int line_nonzeroes = 0;
    for (int j = 0; j < width; j++) {
      if (data[i * width + j] != static_cast<T>(0)) {
        line_nonzeroes++;
      }
    }
    if (line_nonzeroes % 4 == 0) {
      num_nonzeroes += line_nonzeroes;
    } else {
      num_nonzeroes += line_nonzeroes + 4 - (line_nonzeroes % 4);
    }
    num_nonzeroes_act += line_nonzeroes;
  }
  *num_build_nonzeroes = num_nonzeroes;
  return height * width - num_nonzeroes_act;
}

template <typename T>
int ComputeSemiSparseZeros(const Tensor* weights,
                           int* count_nonzeroes,
                           int* count_channels,
                           int* count_blocks,
                           int* flag_semi,
                           const int height,
                           const int width) {
  const T* data = weights->data<T>();
  int num_nonzeroes = 0;
  int num_nonzero_blocks2 = 0;
  int num_nonzero_blocks4 = 0;
  int align4 = height & (-4);
  int align2 = height & (-2);
  for (size_t oc = 0; oc < align4; oc += 4) {
    for (size_t ic = 0; ic < width; ic++) {
      const size_t row0_nonzero =
          static_cast<size_t>(data[oc * width + ic] != static_cast<T>(0));
      const size_t row1_nonzero =
          static_cast<size_t>(data[(oc + 1) * width + ic] != static_cast<T>(0));
      const size_t row2_nonzero =
          static_cast<size_t>(data[(oc + 2) * width + ic] != static_cast<T>(0));
      const size_t row3_nonzero =
          static_cast<size_t>(data[(oc + 3) * width + ic] != static_cast<T>(0));
      num_nonzeroes +=
          row0_nonzero + row1_nonzero + row2_nonzero + row3_nonzero;
      num_nonzero_blocks2 +=
          (row0_nonzero | row1_nonzero) + (row2_nonzero | row3_nonzero);
      num_nonzero_blocks4 +=
          (row0_nonzero | row1_nonzero | row2_nonzero | row3_nonzero);
    }
  }
  for (size_t oc = align4; oc < align2; oc += 2) {
    for (size_t ic = 0; ic < width; ic++) {
      const size_t row0_nonzero =
          static_cast<size_t>(data[oc * width + ic] != static_cast<T>(0));
      const size_t row1_nonzero =
          static_cast<size_t>(data[(oc + 1) * width + ic] != static_cast<T>(0));
      num_nonzeroes += row0_nonzero + row1_nonzero;
      num_nonzero_blocks2 += (row0_nonzero | row1_nonzero);
    }
  }
  const size_t num_block2_nonzeroes = num_nonzeroes;
  for (size_t oc = align2; oc < height; oc++) {
    for (size_t ic = 0; ic < width; ic++) {
      num_nonzeroes +=
          static_cast<size_t>(data[oc * width + ic] != static_cast<T>(0));
    }
  }
  *flag_semi = 0;
  *count_channels = height;
  *count_nonzeroes = num_nonzeroes;
  *count_blocks = num_nonzeroes;
  if ((num_block2_nonzeroes * 5 >= num_nonzero_blocks2 * 9) && (height > 1)) {
    // 2-channel blocks have 90%+ non-zeroes
    *count_channels = (*count_channels) / 2 + (*count_channels) % 2;
    // spmm_parameters = &xnn_params.f32.spmm2;
    *flag_semi = 1;
    // Non-zeroes which don't fit into whole 2-channel blocks, processed
    // one-by-one
    const size_t num_remaining_nonzeroes = num_nonzeroes - num_block2_nonzeroes;
    *count_nonzeroes = num_nonzero_blocks2 * 2 + num_remaining_nonzeroes;
    *count_blocks = num_nonzero_blocks2 + num_remaining_nonzeroes;
  }
  return height * width - (*count_nonzeroes);
}

template <typename T>
int ComputeSparseWeight(const Tensor* w_tensor,
                        const int M,
                        const int K,
                        const int N,
                        const int num_nonzeroes,
                        const int num_build_nonzeroes,
                        Tensor* nonzero_output_tensor,
                        Tensor* oc_nonzeros_tensor,
                        Tensor* diffs_tensor) {
  const T* weights = w_tensor->data<T>();
  T* nonzero_output = nonzero_output_tensor->mutable_data<T>();
  auto* oc_nonzeros = oc_nonzeros_tensor->mutable_data<uint32_t>();
  auto* diffs = diffs_tensor->mutable_data<int32_t>();
  std::vector<int32_t> act_diffs;
  act_diffs.resize(num_nonzeroes);
  int first_ic = 0, last_ic = 0;
  bool first_nonzero = true;
  int nonzero_index = 0, diff_index = 0;
  for (int ocb = 0; ocb < M; ocb++) {
    oc_nonzeros[ocb] = 0;
    for (int ic = 0; ic < K; ic++) {
      if (weights[ocb * K + ic] != static_cast<T>(0)) {
        nonzero_output[nonzero_index++] = weights[ocb * K + ic];
        if (first_nonzero) {
          first_ic = ic;
        } else {
          const int diff = (ic - last_ic) * sizeof(T);
          act_diffs[diff_index++] = diff * N;
        }
        first_nonzero = false;
        last_ic = ic;
        oc_nonzeros[ocb] += 1;
      }
    }
    if (oc_nonzeros[ocb] % 4 != 0) {
      int extra_zeros = 4 - (oc_nonzeros[ocb] % 4);
      for (int j = 0; j < extra_zeros; j++) {
        nonzero_output[nonzero_index++] = 0;
      }
    }
    if (ocb != 0) {
      int cur_rem = oc_nonzeros[ocb - 1] & 3;
      oc_nonzeros[ocb] =
          (cur_rem == 0)
              ? (oc_nonzeros[ocb] + oc_nonzeros[ocb - 1])
              : (oc_nonzeros[ocb] + oc_nonzeros[ocb - 1] + 4 - cur_rem);
    }
  }
  if (!first_nonzero) {
    const int diff = (first_ic - last_ic) * sizeof(T);
    act_diffs[diff_index++] = diff * N;
  }
  int left_index = 0, right_index = 0;
  for (size_t ocb = 0; ocb < M; ocb++) {
    if (ocb == 0) {
      for (int i = 0; i < oc_nonzeros[ocb]; i++) {
        diffs[right_index++] = act_diffs[left_index++];
      }
      if (oc_nonzeros[ocb] % 4 != 0) {
        size_t extra_zeros = 4 - (oc_nonzeros[ocb] % 4);
        for (int j = 0; j < extra_zeros; j++) {
          diffs[right_index++] = 0;
        }
      }
    } else {
      int cur_rem = oc_nonzeros[ocb - 1] & 3;
      int cur_num =
          (cur_rem == 0)
              ? (oc_nonzeros[ocb] - oc_nonzeros[ocb - 1])
              : (oc_nonzeros[ocb] - (oc_nonzeros[ocb - 1] + 4 - cur_rem));
      for (int i = 0; i < cur_num; i++) {
        diffs[right_index++] = act_diffs[left_index++];
      }
      if (cur_num % 4 != 0) {
        size_t extra_zeros = 4 - (cur_num % 4);
        for (int j = 0; j < extra_zeros; j++) {
          diffs[right_index++] = 0;
        }
      }
    }
  }
  int tmp_diff = 0;
  int tmp_ik = 0;
  for (size_t ocb = 0; ocb < M; ocb++) {
    if (ocb == 0) {
      for (int ik = 0; ik < oc_nonzeros[ocb]; ik++) {
        tmp_diff += diffs[tmp_ik++];
      }
    } else {
      for (int ik = 0; ik < (oc_nonzeros[ocb] - oc_nonzeros[ocb - 1]); ik++) {
        tmp_diff += diffs[tmp_ik++];
      }
    }
    if (tmp_ik != 0) {
      diffs[tmp_ik - 1] = tmp_diff;
    }
  }
  return first_ic;
}

template <typename T>
int ComputeSemiSparseWeight(const Tensor* w_tensor,
                            const int M,
                            const int K,
                            const int N,
                            const int count_nonzeroes,
                            const int count_channels,
                            const int count_blocks,
                            Tensor* nonzero_output_tensor,
                            Tensor* oc_nonzeros_tensor,
                            Tensor* diffs_tensor) {
  const T* weights = w_tensor->data<T>();
  T* nonzero_output = nonzero_output_tensor->mutable_data<T>();
  auto* oc_nonzeros = oc_nonzeros_tensor->mutable_data<uint32_t>();
  auto* diffs = diffs_tensor->mutable_data<int32_t>();
  int align2 = M & (-2);
  size_t output_channels_block_size = 2;
  size_t first_ic = 0, last_ic = 0;
  bool first_nonzero = true;
  int nonzero_index = 0, diff_index = 0;
  size_t block_index = 0, block_n = 0;
  for (size_t ocb = 0; ocb < align2; ocb += output_channels_block_size) {
    for (size_t ic = 0; ic < K; ic++) {
      bool is_nonzero_block = false;
      for (size_t oco = 0; oco < output_channels_block_size; oco++) {
        is_nonzero_block |=
            (weights[(ocb + oco) * K + ic] != static_cast<T>(0));
      }
      if (is_nonzero_block) {
        for (size_t oco = 0; oco < output_channels_block_size; oco++) {
          nonzero_output[nonzero_index++] = weights[(ocb + oco) * K + ic];
        }
        if (first_nonzero) {
          first_ic = ic;
        } else {
          const int diff = (ic - last_ic) * sizeof(T);
          diffs[diff_index++] = diff * N;
        }
        first_nonzero = false;
        last_ic = ic;
        oc_nonzeros[block_index] += 1;
        block_n++;
      }
    }
    oc_nonzeros[block_index++] = block_n;
  }
  for (size_t ocb = align2; ocb < M; ocb++) {
    for (size_t ic = 0; ic < K; ic++) {
      if (weights[ocb * K + ic] != static_cast<T>(0)) {
        nonzero_output[nonzero_index++] = weights[ocb * K + ic];
        if (first_nonzero) {
          first_ic = ic;
        } else {
          const int diff = (ic - last_ic) * sizeof(T);
          diffs[diff_index++] = diff * N;
        }
        first_nonzero = false;
        last_ic = ic;
        oc_nonzeros[block_index] += 1;
        block_n++;
      }
    }
    oc_nonzeros[block_index++] = block_n;
  }
  int tmp_diff = 0;
  int tmp_ik = 0;
  size_t block_i = 0;
  for (size_t ocb = 0; ocb < align2; ocb += output_channels_block_size) {
    if (block_i == 0) {
      for (int ik = 0; ik < oc_nonzeros[block_i]; ik++) {
        tmp_diff += diffs[tmp_ik++];
      }
    } else {
      for (int ik = 0; ik < (oc_nonzeros[block_i] - oc_nonzeros[block_i - 1]);
           ik++) {
        tmp_diff += diffs[tmp_ik++];
      }
    }
    if (tmp_ik != 0) {
      diffs[tmp_ik - 1] = tmp_diff;
    }
    block_i++;
  }
  for (size_t ocb = align2; ocb < M; ocb++) {
    if (block_i == 0) {
      for (int ik = 0; ik < oc_nonzeros[block_i]; ik++) {
        tmp_diff += diffs[tmp_ik++];
      }
    } else {
      for (int ik = 0; ik < (oc_nonzeros[block_i] - oc_nonzeros[block_i - 1]);
           ik++) {
        tmp_diff += diffs[tmp_ik++];
      }
    }
    if (tmp_ik != 0) {
      diffs[tmp_ik - 1] = tmp_diff;
    }
    block_i++;
  }
  if (!first_nonzero) {
    const int diff = (first_ic - last_ic) * sizeof(T);
    diffs[diff_index++] = diff * N;
  }
  return first_ic;
}

template <typename T>
int ComputeSparseWeight(const Tensor* w_tensor,
                        const int M,
                        const int K,
                        const int N,
                        const int num_nonzeroes,
                        Tensor* nonzero_output_tensor,
                        Tensor* oc_nonzeros_tensor,
                        Tensor* diffs_tensor) {
  const T* weights = w_tensor->data<T>();
  T* nonzero_output = nonzero_output_tensor->mutable_data<T>();
  auto* oc_nonzeros = oc_nonzeros_tensor->mutable_data<uint32_t>();
  auto* diffs = diffs_tensor->mutable_data<int32_t>();
  int first_ic = 0, last_ic = 0;
  bool first_nonzero = true;
  int nonzero_index = 0, diff_index = 0;
  for (int ocb = 0; ocb < M; ocb++) {
    oc_nonzeros[ocb] = 0;
    for (int ic = 0; ic < K; ic++) {
      if (weights[ocb * K + ic] != static_cast<T>(0)) {
        nonzero_output[nonzero_index++] = weights[ocb * K + ic];
        if (first_nonzero) {
          first_ic = ic;
        } else {
          const int diff = (ic - last_ic) * sizeof(T);
          diffs[diff_index++] = diff * N;
        }
        first_nonzero = false;
        last_ic = ic;
        oc_nonzeros[ocb] += 1;
      }
    }
    oc_nonzeros[ocb] = nonzero_index;
  }
  int tmp_diff = 0;
  int tmp_ik = 0;
  for (size_t ocb = 0; ocb < M; ocb++) {
    if (ocb == 0) {
      for (int ik = 0; ik < oc_nonzeros[ocb]; ik++) {
        tmp_diff += diffs[tmp_ik++];
      }
    } else {
      for (int ik = 0; ik < (oc_nonzeros[ocb] - oc_nonzeros[ocb - 1]); ik++) {
        tmp_diff += diffs[tmp_ik++];
      }
    }
    if (tmp_ik != 0) {
      diffs[tmp_ik - 1] = tmp_diff;
    }
  }
  if (!first_nonzero) {
    const int diff = (first_ic - last_ic) * sizeof(T);
    diffs[diff_index++] = diff * N;
  }
  return first_ic;
}
