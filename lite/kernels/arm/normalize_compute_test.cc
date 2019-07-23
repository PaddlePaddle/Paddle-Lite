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

#include "lite/kernels/arm/normalize_compute.h"
#include <gtest/gtest.h>
#include <memory>
#include <utility>
#include <vector>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

template <typename dtype>
void group_normlize(const float* in_data,
                    const float* scale,
                    const dtype* bias,
                    int n,
                    int c,
                    int h,
                    int w,
                    float eps,
                    int group,
                    dtype* out_data,
                    dtype* out_mean,
                    dtype* out_var) {
  int group_size = (c - 1) / group + 1;
  int im_size = h * w;
  for (int n_index = 0; n_index < n; ++n_index) {
    for (int g_index = 0; g_index < group; ++g_index) {
      dtype t_mean = 0;
      dtype t_var = 0;
      int real_channels = c - g_index * group_size >= group_size
                              ? group_size
                              : c - g_index * group_size;
      int compute_size = im_size * real_channels;
      for (int im_index = 0; im_index < compute_size; ++im_index) {
        t_mean += in_data[im_index];
        t_var += in_data[im_index] * in_data[im_index];
      }
      t_mean /= compute_size;
      t_var /= compute_size;
      t_var -= t_mean * t_mean;
      dtype t_var_inv = 1 / sqrt(t_var + eps);
      if (out_mean) {
        out_mean[n * group + g_index] = t_mean;
      }
      if (out_var) {
        out_var[n * group + g_index] = t_var;
      }

      int scale_bias_start_index = g_index * group_size;
      for (int c_index = 0; c_index < real_channels; ++c_index) {
        int c_start = c_index * im_size;
        for (int im_index = 0; im_index < im_size; ++im_index) {
          dtype dest_val = (in_data[c_start + im_index] - t_mean) * t_var_inv;
          if (scale) {
            dest_val *= scale[scale_bias_start_index + c_index];
          }
          if (bias) {
            dest_val += bias[scale_bias_start_index + c_index];
          }
          out_data[c_start + im_index] = dest_val;
        }
      }
      out_data += compute_size;
      in_data += compute_size;
    }
  }
}

void normalize_compute_ref(const lite::Tensor* input,
                           // lite::Tensor* output,
                           std::vector<lite::Tensor*> output,
                           operators::NormalizeParam param) {
  int p = param.p;
  bool across_spatial = param.across_spatial;
  bool has_scale = param.has_scale;
  bool has_bias = param.has_bias;
  bool channel_shared = param.channel_shared;
  float eps = param.eps;

  auto input_dims = input->dims();
  int n = input_dims[0];
  int c = input_dims[1];
  int h = input_dims[2];
  int w = input_dims[3];

  lite::Tensor th_scale;
  lite::Tensor th_bias;
  float* scale = nullptr;
  float* bias = nullptr;
  float* out_mean = nullptr;
  float* out_var = nullptr;
  if (has_scale) {
    th_scale.resize(param.scale->dims());
    th_scale = param.scale;
    scale = th_scale->mutable_data<float>();
  }
  if (has_bias) {
    th_bias.resize(param.bias->dims());
    th_bias = param.bias;
    bias = th_bias->mutable_data<float>();
  }

  const float* src_ptr = input->data<float>();
  float* dst_ptr = output[0]->mutable_data<float>();

  if (param.group > 0) {
    // group>1, do group normal
    if (output.size() > 1) {
      // out_mean = static_cast<float*>(output[1]->mutable_data());
      // out_mean = 0;
      out_mean = static_cast<float*>(output[1]->mutable_data());
    }
    if (output.size() > 2) {
      // out_var = static_cast<float*>(output[2]->mutable_data());
      // out_var = 0;
      out_var = static_cast<float*>(output[2]->mutable_data());
    }
    group_normlize<float>(src_ptr,
                          scale,
                          bias,
                          n,
                          c,
                          h,
                          w,
                          eps,
                          param.group,
                          dst_ptr,
                          out_mean,
                          out_var);
    return;
  }
  if (across_spatial) {
    int compute_size = h * w * c;
    int outer_size = n * c * h * w / compute_size;

    for (int i = 0; i < outer_size; ++i) {
      float sum = 0;

      for (int j = 0; j < compute_size; ++j) {
        if (p == 1) {
          sum += fabsf(src_ptr[j]);
        } else {
          sum += src_ptr[j] * src_ptr[j];
        }
      }

      // LOG(INFO) << "idx: " << i << ", " << "norm: " << sum;

      if (p == 1) {
        sum = 1 / (sum + eps);
      } else {
        sum = 1 / sqrtf(sum + eps);
      }

      if (has_scale) {         //! with scale
        if (channel_shared) {  // scale is shared across channel
          for (int j = 0; j < compute_size; ++j) {
            dst_ptr[j] = src_ptr[j] * sum * scale[0];
          }
        } else {
          for (int j = 0; j < compute_size; ++j) {
            int c_idx = j / (h * w);
            dst_ptr[j] = src_ptr[j] * sum * scale[c_idx];
          }
        }
      } else {  //! without scale
        for (int j = 0; j < compute_size; ++j) {
          dst_ptr[j] = src_ptr[j] * sum;
        }
      }

      src_ptr += compute_size;
      dst_ptr += compute_size;
    }
  } else {
    int channel_in_size = h * w;

    for (int i = 0; i < n; ++i) {
      const float* src_batch_ptr = src_ptr + i * c * h * w;
      float* dst_batch_ptr = dst_ptr + i * c * h * w;

      for (int j = 0; j < h; ++j) {
        for (int k = 0; k < w; ++k) {
          const float* src_pixel = src_batch_ptr + j * w + k;
          float* dst_pixel = dst_batch_ptr + j * w + k;
          float norm = 0.f;
          // LOG(INFO)<<"c:"<<c;

          for (int l = 0; l < c; ++l) {
            if (p == 1) {
              norm += fabsf(src_pixel[l * channel_in_size]);
            } else {
              norm += src_pixel[l * channel_in_size] *
                      src_pixel[l * channel_in_size];
            }
          }
          // LOG(INFO)<<"norm:"<<norm;

          if (p == 1) {
            norm = 1.f / (norm + eps);
          } else {
            norm = 1.f / sqrtf(norm + eps);
          }

          for (int l = 0; l < c; ++l) {
            if (has_scale) {
              if (channel_shared) {
                dst_pixel[l * channel_in_size] =
                    src_pixel[l * channel_in_size] * norm * scale[0];
              } else {
                dst_pixel[l * channel_in_size] =
                    src_pixel[l * channel_in_size] * norm * scale[l];
              }
            } else {
              dst_pixel[l * channel_in_size] =
                  src_pixel[l * channel_in_size] * norm;
              // LOG(INFO)<<"dst:"<<dst_pixel[l * channel_in_size];
              // LOG(INFO)<<"src:"<<src_pixel[l * channel_in_size];
              // LOG(INFO)<<"norm_dd:"<<norm;
            }
          }
        }
      }
    }
  }
}

TEST(normalize_arm, retrive_op) {
  auto normalize =
      KernelRegistry::Global().Create<TARGET(kARM), PRECISION(kFloat)>(
          "normalize");
  ASSERT_FALSE(normalize.empty());
  ASSERT_TRUE(normalize.front());
}

TEST(normalize_arm, init) {
  NormalizeCompute normalize;
  ASSERT_EQ(normalize.precision(), PRECISION(kFloat));
  ASSERT_EQ(normalize.target(), TARGET(kARM));
}

TEST(normalize_arm, compute) {}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(normalize, kARM, kFloat, kNCHW, def);
