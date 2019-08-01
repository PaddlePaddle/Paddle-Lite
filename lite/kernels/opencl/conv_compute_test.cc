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
#include <random>
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/opencl/target_wrapper.h"

namespace paddle {
namespace lite {

#define A(i, j) a[i * lda + j]
#define B(i, j) cur_b[i * ldb + j]
#define C(i, j) cur_c[i * ldc + j]

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
                       bool flag_relu) {
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
    for (int g = 0; g < group; ++g) {
      for (int oc = 0; oc < out_c_group; ++oc) {
        for (int oh = 0; oh < out_h; ++oh) {
          for (int ow = 0; ow < out_w; ++ow) {
            int out_idx = n * group * out_c_group * out_h * out_w +
                          g * out_c_group * out_h * out_w + oc * out_h * out_w +
                          oh * out_w + ow;
            Dtype2 bias_d =
                with_bias ? (bias_data[g * out_c_group + oc]) : (Dtype2)0;
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
            if (flag_relu) {
              dst_data_ref[out_idx] = dst_data_ref[out_idx] > (Dtype2)0
                                          ? dst_data_ref[out_idx]
                                          : (Dtype2)0;
            }
          }
        }
      }
    }
  }
}

template <typename T>
void gemm_batch_bias(const int batch_size,
                     const T* a,
                     const int M,
                     const int K,
                     const T* b,
                     const int K_,
                     const int N,
                     T* biases,
                     T* c) {
  EXPECT_TRUE(K_ == K && M > 0 && N > 0 && K > 0);
  for (int bidx = 0; bidx < batch_size; ++bidx) {
    const T* cur_b = b + K * N * bidx;
    T* cur_c = c + M * N * bidx;
    EXPECT_TRUE(a && cur_b && cur_c);
    const int lda = K;
    const int ldb = N;
    const int ldc = N;
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        C(m, n) = 0.0f;
        for (int k = 0; k < K; ++k) {
          C(m, n) += A(m, k) * B(k, n);
        }
      }
    }
    if (biases) {
      for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
          C(m, n) += biases[m];
        }
      }
    }
  }
}

void PrintData(std::string name, float* a, const int rows, const int cols) {
  std::cout << "==== " << name << " ====" << std::endl;
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      std::cout << " " << a[r * cols + c];
    }
    std::cout << std::endl;
  }
}

// #define PRINT_RESULT
// #define LOOP_TEST
TEST(conv2d, compute_conv2d_1x1) {
  // conv2d 1x1 note
  // kernel/filter size = 1x1, group = 1, pad = 0, stride = 1, dilation = 1
  // gemm implement
  // a: filter_d ==> <m, k> <=> <oc, ic>
  // b: x_d      ==> <k, n> <=> <ic, ih*iw>
  // c: output_d ==> <m, n> <=> <oc, ih*iw>
  std::unique_ptr<KernelContext> context(new KernelContext);
  context->As<OpenCLContext>().InitOnce();
  const int ksize = 1;
  const int stride = 1;
  const int pad = 0;
  const int group = 1;
  const int dilation = 1;

#ifdef LOOP_TEST
  for (int batch_size = 1; batch_size < 4; ++batch_size) {
    for (int oc = 1; oc < 213; oc += 71) {         // m
      for (int ih = 1; ih < 123; ih += 31) {       // ih
        for (int iw = 1; iw < 123; iw += 31) {     // iw
          for (int ic = 1; ic < 123; ic += 121) {  // k
#else
  // groups:1 stride_h:1 stride_w:1 pad_h:0 pad_w:0 kernel_h:1 kernel_h:1
  // x_dims:1 32 112 112
  // output_dims:1 64 112 112
  // filter_dims:64 32 1 1
  const int batch_size = 8;
  const int oc = 64;
  const int ih = 112;
  const int iw = 112;
  const int ic = 32;
#endif
            for (bool bias_flag : {true, false}) {
              for (bool relu_flag : {true, false}) {
                const int oh = ih;
                const int ow = iw;
                auto kernels =
                    KernelRegistry::Global().Create("conv2d",
                                                    TARGET(kOpenCL),
                                                    PRECISION(kFloat),
                                                    DATALAYOUT(kNCHW));
                ASSERT_FALSE(kernels.empty());
                auto kernel = std::move(kernels.front());

                lite::Tensor x, filter, bias, out, out_ref;
                operators::ConvParam param;
                param.x = &x;
                param.filter = &filter;
                param.bias = bias_flag ? &bias : nullptr;
                param.output = &out;
                param.strides = {stride, stride};
                param.paddings = {pad, pad};
                param.groups = group;
                param.dilations = {dilation, dilation};
                param.fuse_relu = relu_flag;

                kernel->SetParam(param);
                std::unique_ptr<KernelContext> conv_context(new KernelContext);
                context->As<OpenCLContext>().CopySharedTo(
                    &(conv_context->As<OpenCLContext>()));
                kernel->SetContext(std::move(conv_context));

                const DDim x_dim =
                    DDim(std::vector<DDim::value_type>{batch_size, ic, ih, iw});
                const DDim filter_dim =
                    DDim(std::vector<DDim::value_type>{oc, ic, ksize, ksize});
                const DDim bias_dim = DDim(std::vector<DDim::value_type>{oc});
                const DDim out_dim =
                    DDim(std::vector<DDim::value_type>{batch_size, oc, oh, ow});

                x.Resize(x_dim);
                filter.Resize(filter_dim);
                bias.Resize(bias_dim);
                out.Resize(out_dim);
                out_ref.Resize(out_dim);

                auto* x_data =
                    x.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
                auto* filter_data =
                    filter.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
                auto* bias_data =
                    bias.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));

                std::default_random_engine engine;
                std::uniform_real_distribution<float> dist(-5, 5);
                auto* mapped_x = static_cast<float*>(TargetWrapperCL::Map(
                    x_data, 0, sizeof(float) * x_dim.production()));
                for (int i = 0; i < x_dim.production(); ++i) {
                  mapped_x[i] = static_cast<int>(dist(engine));
                }
                auto* mapped_filter = static_cast<float*>(TargetWrapperCL::Map(
                    filter_data, 0, sizeof(float) * filter_dim.production()));
                for (int i = 0; i < filter_dim.production(); ++i) {
                  mapped_filter[i] = static_cast<int>(dist(engine));
                }
                auto* mapped_bias = static_cast<float*>(TargetWrapperCL::Map(
                    bias_data, 0, sizeof(float) * bias_dim.production()));
                for (int i = 0; i < bias_dim.production(); ++i) {
                  mapped_bias[i] = static_cast<int>(dist(engine));
                }

                // run opencl kernel
                kernel->Launch();

                auto* wait_list = context->As<OpenCLContext>().cl_wait_list();
                auto* out_ptr = param.output->data<float, cl::Buffer>();
                auto it = wait_list->find(out_ptr);
                if (it != wait_list->end()) {
                  VLOG(4) << "--- Find the sync event for the target cl "
                             "tensor. ---";
                  auto& event = *(it->second);
                  event.wait();
                } else {
                  LOG(FATAL) << "Could not find the sync event for the target "
                                "cl tensor.";
                }

                // run cpu ref
                auto* out_ref_data = out_ref.mutable_data<float>(TARGET(kARM));
                conv_basic(mapped_x,
                           out_ref_data,
                           batch_size,
                           oc,
                           oh,
                           ow,
                           ic,
                           ih,
                           iw,
                           mapped_filter,
                           mapped_bias,
                           group,
                           ksize,
                           ksize,
                           stride,
                           stride,
                           dilation,
                           dilation,
                           pad,
                           pad,
                           bias_flag,
                           relu_flag);

                auto* out_data = out.mutable_data<float, cl::Buffer>();
                auto* mapped_out = static_cast<float*>(TargetWrapperCL::Map(
                    out_data, 0, sizeof(float) * out_dim.production()));

#ifdef PRINT_RESULT
                PrintData("mapped_x", static_cast<float*>(mapped_x), m, k);
                PrintData(
                    "mapped_filter", static_cast<float*>(mapped_filter), k, n);
                PrintData(
                    "mapped_bias", static_cast<float*>(mapped_bias), 1, n);
                PrintData(
                    "out_ref_data", static_cast<float*>(out_ref_data), m, n);
                PrintData("mapped_out", static_cast<float*>(mapped_out), m, n);
#endif

                for (int i = 0; i < out_dim.production(); i++) {
                  EXPECT_NEAR(mapped_out[i], out_ref_data[i], 1e-6);
                }

                TargetWrapperCL::Unmap(x_data, mapped_x);
                TargetWrapperCL::Unmap(filter_data, mapped_filter);
                TargetWrapperCL::Unmap(bias_data, mapped_bias);
                TargetWrapperCL::Unmap(out_data, mapped_out);
              }
            }
#ifdef LOOP_TEST
          }  // ic
        }    // iw
      }      // ih
    }        // oc
  }          // batch_size
#endif
}

TEST(conv2d, compute_conv2d_gemm) {
  std::unique_ptr<KernelContext> context(new KernelContext);
  context->As<OpenCLContext>().InitOnce();
  // x_dims:1 3 224 224
  // output_dims:1 32 112 112
  // filter_dims:32 3 3 3
  const int ksize = 3;
  const int stride = 2;
  const int pad = 1;
  const int group = 1;
  const int dilation = 1;

#ifdef LOOP_TEST
  for (int batch_size = 1; batch_size < 4; ++batch_size) {
    for (int oc = 1; oc < 213; oc += 71) {         // m
      for (int ih = 1; ih < 123; ih += 31) {       // ih
        for (int iw = 1; iw < 123; iw += 31) {     // iw
          for (int ic = 1; ic < 123; ic += 121) {  // k
#else
            const int batch_size = 8;
            const int oc = 32;
            const int ih = 224;
            const int iw = 224;
            const int ic = 3;
#endif
            for (bool bias_flag : {true, false}) {
              for (bool relu_flag : {true, false}) {
                const int oh = (ih + 2 * pad - ksize) / stride + 1;
                const int ow = (iw + 2 * pad - ksize) / stride + 1;
                auto kernels =
                    KernelRegistry::Global().Create("conv2d",
                                                    TARGET(kOpenCL),
                                                    PRECISION(kFloat),
                                                    DATALAYOUT(kNCHW));
                ASSERT_FALSE(kernels.empty());
                auto kernel = std::move(kernels.front());

                lite::Tensor x, filter, bias, out, out_ref;
                operators::ConvParam param;
                param.x = &x;
                param.filter = &filter;
                param.bias = bias_flag ? &bias : nullptr;
                param.output = &out;
                param.strides = {stride, stride};
                param.paddings = {pad, pad};
                param.groups = group;
                param.dilations = {dilation, dilation};
                param.fuse_relu = relu_flag;

                kernel->SetParam(param);
                std::unique_ptr<KernelContext> conv_context(new KernelContext);
                context->As<OpenCLContext>().CopySharedTo(
                    &(conv_context->As<OpenCLContext>()));
                kernel->SetContext(std::move(conv_context));

                const DDim x_dim =
                    DDim(std::vector<DDim::value_type>{batch_size, ic, ih, iw});
                const DDim filter_dim =
                    DDim(std::vector<DDim::value_type>{oc, ic, ksize, ksize});
                const DDim bias_dim = DDim(std::vector<DDim::value_type>{oc});
                const DDim out_dim =
                    DDim(std::vector<DDim::value_type>{batch_size, oc, oh, ow});

                x.Resize(x_dim);
                filter.Resize(filter_dim);
                bias.Resize(bias_dim);
                out.Resize(out_dim);
                out_ref.Resize(out_dim);

                auto* x_data =
                    x.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
                auto* filter_data =
                    filter.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
                auto* bias_data =
                    bias.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));

                std::default_random_engine engine;
                std::uniform_real_distribution<float> dist(-5, 5);
                auto* mapped_x = static_cast<float*>(TargetWrapperCL::Map(
                    x_data, 0, sizeof(float) * x_dim.production()));
                for (int i = 0; i < x_dim.production(); ++i) {
                  mapped_x[i] = static_cast<int>(dist(engine));
                }
                auto* mapped_filter = static_cast<float*>(TargetWrapperCL::Map(
                    filter_data, 0, sizeof(float) * filter_dim.production()));
                for (int i = 0; i < filter_dim.production(); ++i) {
                  mapped_filter[i] = static_cast<int>(dist(engine));
                }
                auto* mapped_bias = static_cast<float*>(TargetWrapperCL::Map(
                    bias_data, 0, sizeof(float) * bias_dim.production()));
                for (int i = 0; i < bias_dim.production(); ++i) {
                  mapped_bias[i] = static_cast<int>(dist(engine));
                }

                // run opencl kernel
                kernel->Launch();

                auto* wait_list = context->As<OpenCLContext>().cl_wait_list();
                auto* out_ptr = param.output->data<float, cl::Buffer>();
                auto it = wait_list->find(out_ptr);
                if (it != wait_list->end()) {
                  VLOG(4) << "--- Find the sync event for the target cl "
                             "tensor. ---";
                  auto& event = *(it->second);
                  event.wait();
                } else {
                  LOG(FATAL) << "Could not find the sync event for the target "
                                "cl tensor.";
                }

                // run cpu ref
                auto* out_ref_data = out_ref.mutable_data<float>(TARGET(kARM));
                conv_basic(mapped_x,
                           out_ref_data,
                           batch_size,
                           oc,
                           oh,
                           ow,
                           ic,
                           ih,
                           iw,
                           mapped_filter,
                           mapped_bias,
                           group,
                           ksize,
                           ksize,
                           stride,
                           stride,
                           dilation,
                           dilation,
                           pad,
                           pad,
                           bias_flag,
                           relu_flag);
                auto* out_data = out.mutable_data<float, cl::Buffer>();
                auto* mapped_out = static_cast<float*>(TargetWrapperCL::Map(
                    out_data, 0, sizeof(float) * out_dim.production()));

#ifdef PRINT_RESULT
                PrintData("mapped_x", static_cast<float*>(mapped_x), m, k);
                PrintData(
                    "mapped_filter", static_cast<float*>(mapped_filter), k, n);
                PrintData(
                    "mapped_bias", static_cast<float*>(mapped_bias), 1, n);
                PrintData(
                    "out_ref_data", static_cast<float*>(out_ref_data), m, n);
                PrintData("mapped_out", static_cast<float*>(mapped_out), m, n);
#endif

                for (int i = 0; i < out_dim.production(); i++) {
                  EXPECT_NEAR(mapped_out[i], out_ref_data[i], 1e-6);
                }

                TargetWrapperCL::Unmap(x_data, mapped_x);
                TargetWrapperCL::Unmap(filter_data, mapped_filter);
                TargetWrapperCL::Unmap(bias_data, mapped_bias);
                TargetWrapperCL::Unmap(out_data, mapped_out);
              }
            }
#ifdef LOOP_TEST
          }  // ic
        }    // iw
      }      // ih
    }        // oc
  }          // batch_size
#endif
}

}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(conv2d, kOpenCL, kFloat, kNCHW, def);
