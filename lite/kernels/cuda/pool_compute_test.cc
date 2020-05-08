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

#include "lite/kernels/cuda/pool_compute.h"
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

using Tensor = lite::Tensor;
using DDim = lite::DDim;

#define IN(n, c, h, w)                                 \
  input_data[w + h * input_w + c * input_h * input_w + \
             n * input_c * input_h * input_w]
#define OUT(n, c, h, w)                                    \
  output_data[w + h * output_w + c * output_h * output_w + \
              n * output_c * output_h * output_w]

template <typename Dtype>
void nchw2nhwc_ref(lite::Tensor* input, lite::Tensor* output) {
  auto* input_data = input->data<Dtype>();
  auto* output_data = output->mutable_data<Dtype>();

  int input_n = input->dims()[0];
  int input_c = input->dims()[1];
  int input_h = input->dims()[2];
  int input_w = input->dims()[3];
  int output_c = output->dims()[1];
  int output_h = output->dims()[2];
  int output_w = output->dims()[3];

  for (int n = 0; n < input_n; ++n) {
    for (int c = 0; c < input_c; ++c) {
      for (int h = 0; h < input_h; ++h) {
        for (int w = 0; w < input_w; ++w) {
          OUT(n, h, w, c) = IN(n, c, h, w);
        }
      }
    }
  }
}

#undef IN
#undef OUT

#define IN(n, h, w, c)                                 \
  input_data[c + w * input_c + h * input_w * input_c + \
             n * input_h * input_w * input_c]
#define OUT(n, h, w, c)                                    \
  output_data[c + w * output_c + h * output_w * output_c + \
              n * output_h * output_w * output_c]

template <typename Dtype>
void nhwc2nchw_ref(lite::Tensor* input, lite::Tensor* output) {
  auto* input_data = input->data<Dtype>();
  auto* output_data = output->mutable_data<Dtype>();

  int input_n = input->dims()[0];
  int input_h = input->dims()[1];
  int input_w = input->dims()[2];
  int input_c = input->dims()[3];
  int output_h = output->dims()[1];
  int output_w = output->dims()[2];
  int output_c = output->dims()[3];

  for (int n = 0; n < input_n; ++n) {
    for (int c = 0; c < input_c; ++c) {
      for (int h = 0; h < input_h; ++h) {
        for (int w = 0; w < input_w; ++w) {
          OUT(n, c, h, w) = IN(n, h, w, c);
        }
      }
    }
  }
}

static int PoolOutputSize(int input_size,
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

static std::vector<int64_t> compute_output_shape(operators::PoolParam* param_,
                                                 bool is_nchw) {
  int axis = 2;
  if (!is_nchw) axis = 1;
  const auto x_dims = param_->x->dims();
  std::vector<int>& ksize = param_->ksize;
  if (param_->global_pooling) {
    ksize.resize(static_cast<size_t>(x_dims.size()) - 2);
    auto paddings = *param_->paddings;
    for (size_t i = 0; i < ksize.size(); ++i) {
      paddings[2 * i] = 0;
      paddings[2 * i + 1] = 0;
      ksize[i] = static_cast<int>(x_dims[i + 2]);
    }
  }

  std::vector<int64_t> output_shape({x_dims[0]});
  if (is_nchw) output_shape.push_back(x_dims[1]);
  if (param_->adaptive) {
    output_shape.insert(
        output_shape.end(), param_->ksize.begin(), param_->ksize.end());
  } else {
    auto paddings = *param_->paddings;
    for (size_t i = 0; i < param_->ksize.size(); ++i) {
      output_shape.push_back(PoolOutputSize(x_dims[i + axis],
                                            param_->ksize[i],
                                            paddings[2 * i],
                                            paddings[2 * i + 1],
                                            param_->strides[i],
                                            param_->ceil_mode));
    }
  }
  if (!is_nchw) output_shape.push_back(x_dims[3]);
  return output_shape;
}

static void pool_compute_ref(const operators::PoolParam& param) {
  auto& in_dims = param.x->dims();
  auto& out_dims = param.output->dims();

  const float* src_ptr = param.x->data<const float>();
  float* dst_ptr = param.output->mutable_data<float>();

  std::vector<int> ksize = param.ksize;
  std::vector<int> strides = param.strides;
  std::vector<int> paddings = *param.paddings;

  std::string pooling_type = param.pooling_type;
  bool global_pooling = param.global_pooling;
  bool exclusive = param.exclusive;
  std::string data_format = param.data_format;

  int in_n = in_dims[0];
  int in_c = in_dims[1];
  int in_h = in_dims[2];
  int in_w = in_dims[3];
  int size_in_n = in_c * in_h * in_w;
  int size_in_c = in_h * in_w;

  int out_h = out_dims[2];
  int out_w = out_dims[3];
  int size_out_n = in_c * out_h * out_w;
  int size_out_c = out_h * out_w;

  int window_h = ksize[0];
  int window_w = ksize[1];
  int stride_h = strides[0];
  int stride_w = strides[1];
  int pad_h = paddings[0];
  int pad_w = paddings[2];

  if (global_pooling == true) {
    for (int n = 0; n < in_n; ++n) {
      for (int c = 0; c < in_c; ++c) {
        const float* src = src_ptr + n * size_in_n + c * size_in_c;
        float res = src[0];
        if (pooling_type == "max") {
          for (int i = 1; i < size_in_c; ++i) {
            float cur_val = src[i];
            res = cur_val > res ? cur_val : res;
          }
        } else if (pooling_type == "avg") {
          for (int i = 1; i < size_in_c; ++i) {
            float cur_val = src[i];
            res += cur_val;
          }
          res /= size_in_c;
        }
        dst_ptr[n * size_out_n + c] = res;
      }
    }
  } else {
    for (int n = 0; n < in_n; ++n) {
      for (int c = 0; c < in_c; ++c) {
        for (int h = 0; h < out_h; ++h) {
          int sh = h * stride_h;
          int eh = sh + window_h;
          sh = (sh - pad_h) < 0 ? 0 : sh - pad_h;
          eh = (eh - pad_h) > in_h ? in_h : eh - pad_h;
          for (int w = 0; w < out_w; ++w) {
            int sw = w * stride_w;
            int ew = sw + window_w;
            sw = (sw - pad_w) < 0 ? 0 : sw - pad_w;
            ew = (ew - pad_w) > in_w ? in_w : ew - pad_w;
            int pooling_size = (ew - sw) * (eh - sh);
            if (pooling_size == 0) {
              dst_ptr[n * size_out_n + c * size_out_c + h * out_w + w] = 0.f;
              continue;
            }
            float res = 0.f;
            for (int kh = sh; kh < eh; ++kh) {
              for (int kw = sw; kw < ew; ++kw) {
                int src_idx = n * size_in_n + c * size_in_c + kh * in_w + kw;
                if (kh == sh && kw == sw) {
                  res = src_ptr[src_idx];
                } else {
                  if (pooling_type == "max") {
                    res = res >= src_ptr[src_idx] ? res : src_ptr[src_idx];
                  }
                  if (pooling_type == "avg") {
                    res += src_ptr[src_idx];
                  }
                }
              }
            }
            if (pooling_type == "avg") {
              if (exclusive) {
                res /= pooling_size;
              } else {
                res /= window_h * window_w;
              }
            }
            dst_ptr[n * size_out_n + c * size_out_c + h * out_w + w] = res;
          }
        }
      }
    }
  }
}

TEST(pool_cuda, compute) {
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  auto& context = ctx->As<CUDAContext>();
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  context.SetExecStream(stream);

  PoolCompute pool;
  operators::PoolParam param;
  pool.SetContext(std::move(ctx));

  lite::Tensor x;
  lite::Tensor x_cpu;
  lite::Tensor output;
  lite::Tensor output_cpu;
  lite::Tensor output_ref;
  for (auto pooling_type : {"max", "avg"}) {
    for (auto ceil_mode : {true, false}) {
      for (auto global_pooling : {true, false}) {
        for (auto exclusive : {true, false}) {
          for (auto ksize : {2, 3}) {
            for (auto stride : {1, 2}) {
              for (auto pad : {0, 1}) {
                for (auto n : {1, 2}) {
                  for (auto c : {1, 3}) {
                    for (auto h : {3}) {
                      for (auto w : {3}) {
                        LOG(INFO) << "n:" << n << " c:" << c << " h:" << h
                                  << " w:" << w << " ksize:" << ksize
                                  << " stride:" << stride << " pad:" << pad
                                  << " exclusive:" << exclusive
                                  << " global_pooling:" << global_pooling
                                  << " ceil_mode: " << ceil_mode
                                  << " pooling_type:" << pooling_type;

                        // init x, output
                        x.Resize(DDim(std::vector<int64_t>({n, c, h, w})));
                        x_cpu.Resize(DDim(std::vector<int64_t>({n, c, h, w})));
                        auto* x_cpu_data = x_cpu.mutable_data<float>();
                        for (int i = 0; i < x_cpu.dims().production(); ++i) {
                          float sign = i % 3 == 0 ? -0.03 : 0.05f;
                          x_cpu_data[i] = sign * (i % 128);
                        }
                        x.Assign<float, DDim, TARGET(kCUDA)>(x_cpu_data,
                                                             x_cpu.dims());
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
                        std::vector<int> paddings = {pad, pad, pad, pad};
                        param.paddings =
                            std::make_shared<std::vector<int>>(paddings);
                        param.exclusive = exclusive;
                        param.ceil_mode = ceil_mode;
                        param.adaptive = false;
                        param.use_quantizer = false;

                        const std::vector<int64_t>& output_shape =
                            compute_output_shape(&param, true);
                        if (output_shape[2] * output_shape[3] == 0) continue;
                        output.Resize(DDim(output_shape));
                        output_ref.Resize(DDim(output_shape));
                        output_cpu.Resize(DDim(output_shape));
                        auto* output_data =
                            output.mutable_data<float>(TARGET(kCUDA));
                        auto* output_ref_data =
                            output_ref.mutable_data<float>();
                        auto* output_cpu_data =
                            output_cpu.mutable_data<float>();

                        // compute
                        pool.SetParam(param);
                        pool.Launch();

                        // compute ref
                        param.x = &x_cpu;
                        param.output = &output_ref;
                        pool_compute_ref(param);

                        cudaDeviceSynchronize();
                        CopySync<TARGET(kCUDA)>(output_cpu_data,
                                                output_data,
                                                sizeof(float) * output.numel(),
                                                IoDirection::DtoH);
                        // compare
                        for (int i = 0; i < output.dims().production(); i++) {
                          EXPECT_NEAR(
                              output_cpu_data[i], output_ref_data[i], 1e-4);
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

TEST(pool_cuda, nhwc) {
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  auto& context = ctx->As<CUDAContext>();
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  context.SetExecStream(stream);

  PoolComputeNHWC pool;
  operators::PoolParam param;
  pool.SetContext(std::move(ctx));

  lite::Tensor x, temp;
  lite::Tensor x_cpu;
  lite::Tensor output;
  lite::Tensor output_cpu, output_temp;
  lite::Tensor output_ref;
  for (auto pooling_type : {"max", "avg"}) {
    for (auto ceil_mode : {false}) {
      for (auto global_pooling : {true, false}) {
        for (auto exclusive : {false, true}) {
          for (auto ksize : {3}) {
            for (auto stride : {3}) {
              for (auto pad : {1}) {
                for (auto n : {1}) {
                  for (auto c : {3}) {
                    for (auto h : {8}) {
                      for (auto w : {8}) {
                        LOG(INFO) << "n:" << n << " c:" << c << " h:" << h
                                  << " w:" << w << " ksize:" << ksize
                                  << " stride:" << stride << " pad:" << pad
                                  << " exclusive:" << exclusive
                                  << " global_pooling:" << global_pooling
                                  << " ceil_mode: " << ceil_mode
                                  << " pooling_type:" << pooling_type;

                        // init x, output
                        x.Resize(DDim(std::vector<int64_t>({n, h, w, c})));
                        temp.Resize(DDim(std::vector<int64_t>({n, h, w, c})));
                        x_cpu.Resize(DDim(std::vector<int64_t>({n, c, h, w})));

                        auto* x_cpu_data = x_cpu.mutable_data<float>();
                        for (int i = 0; i < x_cpu.dims().production(); ++i) {
                          float sign = i % 3 == 0 ? -0.03 : 0.05f;
                          x_cpu_data[i] = sign * (i % 128);
                        }

                        nchw2nhwc_ref<float>(&x_cpu, &temp);
                        auto* temp_cpu_data = temp.mutable_data<float>();

                        x.Assign<float, DDim, TARGET(kCUDA)>(temp_cpu_data,
                                                             temp.dims());
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
                        std::vector<int> paddings = {pad, pad, pad, pad};
                        param.paddings =
                            std::make_shared<std::vector<int>>(paddings);
                        param.exclusive = exclusive;
                        param.ceil_mode = ceil_mode;
                        param.adaptive = false;
                        param.use_quantizer = false;

                        const std::vector<int64_t>& output_shape =
                            compute_output_shape(&param, false);
                        if (output_shape[2] * output_shape[3] == 0) continue;
                        output.Resize(DDim(output_shape));
                        output_temp.Resize(DDim(output_shape));
                        output_cpu.Resize(DDim(output_shape));

                        auto* output_data =
                            output.mutable_data<float>(TARGET(kCUDA));
                        auto* output_cpu_data =
                            output_cpu.mutable_data<float>();

                        // compute
                        pool.SetParam(param);
                        pool.Launch();

                        // compute ref
                        param.x = &x_cpu;
                        // nchw
                        const std::vector<int64_t>& output_shape_ref =
                            compute_output_shape(&param, true);

                        output_ref.Resize(DDim(output_shape_ref));
                        // auto* output_ref_data =
                        //    output_ref.mutable_data<float>();
                        param.output = &output_ref;
                        pool_compute_ref(param);
                        nchw2nhwc_ref<float>(&output_ref, &output_temp);
                        auto* output_temp_data =
                            output_temp.mutable_data<float>();

                        cudaDeviceSynchronize();
                        CopySync<TARGET(kCUDA)>(output_cpu_data,
                                                output_data,
                                                sizeof(float) * output.numel(),
                                                IoDirection::DtoH);
                        // compare
                        for (int i = 0; i < output.dims().production(); i++) {
                          EXPECT_NEAR(
                              output_cpu_data[i], output_temp_data[i], 1e-4);
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
}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
