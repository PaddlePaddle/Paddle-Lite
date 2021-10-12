// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/backends/metal/target_wrapper.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include <gtest/gtest.h>
#include <random>

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
    std::string flag_relu) {
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
                        Dtype2 bias_d = with_bias ? (bias_data[g * out_c_group + oc]) : (Dtype2)0;
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
                                    int widx = g * out_c_group * in_c_group * kernel_h * kernel_w +
                                               oc * in_c_group * kernel_h * kernel_w +
                                               ic * kernel_h * kernel_w + kh * kernel_w + kw;

                                    dst_data_ref[out_idx] += src_data[iidx] * weights_data[widx];
                                }
                            }
                        }
                        if (flag_relu == "relu") {
                            dst_data_ref[out_idx] = dst_data_ref[out_idx] > (Dtype2)0
                                                        ? dst_data_ref[out_idx]
                                                        : (Dtype2)0;
                        } else if (flag_relu == "relu6") {
                            auto dst_tmp = (dst_data_ref[out_idx] > (Dtype2)0)
                                               ? dst_data_ref[out_idx]
                                               : (Dtype2)0;
                            dst_data_ref[out_idx] = (dst_tmp < 6.f) ? dst_tmp : 6.f;
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

void PrintData(std::string name,
    float* a,
    const int rows,
    const int cols,
    const int batch_size = 1) {
    std::cout << "==== " << name << " ====" << std::endl;
    for (int b = 0; b < batch_size; ++b) {
        std::cout << "-- bidx = " << b << " --" << std::endl;
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                std::cout << " " << a[b * rows * cols + r * cols + c];
            }
            std::cout << std::endl;
        }
    }
}

TEST(depthwise_conv2d, buffer_compare) {
    Tensor a;
    const std::vector<DDim::value_type> vec_dim{16};
    const DDim x_dim = DDim(vec_dim);
    a.Resize(x_dim);

    auto image = a.mutable_data<MetalHalf, MetalImage>(x_dim);

    std::vector<float> input = {0.610179,
        -0.340205,
        3.278859,
        2.392035,
        0.952910,
        3.035821,
        3.706363,
        4.150465,
        -0.470437,
        0.273920,
        4.507082,
        2.403983,
        2.947067,
        1.090603,
        7.417082,
        5.028472};

    image->CopyFromNCHW<float>(input.data());

    std::vector<float> output;
    output.resize(16);

    image->CopyToNCHW<float>(output.data());

    //  for (int i = 0; i < 16; i++) {
    //    std::cout << input[i] << " : " << output[i] << std::endl;
    //  }
};

//// #define PRINT_RESULT
//#define LOOP_TEST
TEST(conv2d, compute_conv2d_gemm) {
    std::unique_ptr<KernelContext> context(new KernelContext);
    context->As<ContextMetal>().InitOnce();
    // x_dims:1 3 224 224
    // output_dims:1 32 112 112
    // filter_dims:32 3 3 3
    int ksize = 3;
    const int stride = 2;
    const int pad = 1;
    const int group = 1;
    const int dilation = 1;
    int loop_cnt = 0;

#ifdef LOOP_TEST
    for (int batch_size = 1; batch_size < 3; ++batch_size) {
        for (int oc = 1; oc < 10; oc += 1) {              // m
            for (int ih = 1; ih < 10; ih += 1) {          // ih
                for (int iw = 1; iw < 10; iw += 1) {      // iw
                    for (int ic = 1; ic < 10; ic += 1) {  // k
                        for (bool bias_flag : {true, false}) {
                            for (std::string relu_flag : {"relu", "relu6", "None"}) {
#else

    const int batch_size = 1;
    const int oc = 32;
    const int ih = 224;
    const int iw = 224;
    const int ic = 3;
    const bool bias_flag = true;
    const std::string relu_flag = "None";  // "relu", "relu6", "None"

#endif
                                const int oh = (ih + 2 * pad - ksize) / stride + 1;
                                const int ow = (iw + 2 * pad - ksize) / stride + 1;
                                // a: filter_d ==> <m, k> <=> <oc, ic>
                                // b: x_d      ==> <k, n> <=> <ic, ih*iw>
                                // c: output_d ==> <m, n> <=> <oc, ih*iw>

                                int m = oc;
                                int k = ic * ksize * ksize;
                                int n = oc;
                                LOG(INFO) << "bs=" << batch_size << " oc=" << oc << " ic=" << ic
                                          << " ih=" << ih << " iw=" << iw << " oh=" << oh
                                          << " ow=" << ow << " bias_flag=" << bias_flag
                                          << " relu_flag=" << relu_flag;
                                LOG(INFO) << "m=oc=" << oc
                                          << " k=ic*ksize*ksize=" << ic * ksize * ksize
                                          << " n=oc=" << oc;

                                auto kernels = KernelRegistry::Global().Create("conv2d",
                                    TARGET(kMetal),
                                    PRECISION(kFloat),
                                    DATALAYOUT(kMetalTexture2DArray));
                                ASSERT_FALSE(kernels.empty());
                                auto kernel = std::move(kernels.front());

                                lite::Tensor x, filter, bias, out, out_ref;
                                operators::ConvParam param;
                                param.x = &x;
                                param.filter = &filter;
                                param.bias = bias_flag ? &bias : nullptr;
                                param.output = &out;
                                param.strides = {stride, stride};
                                std::vector<int> paddings = {pad, pad, pad, pad};
                                param.groups = group;
                                std::vector<int> dilations = {dilation, dilation};
                                if (relu_flag == "relu") {
                                    param.fuse_relu = true;
                                } else if (relu_flag == "None") {
                                    param.fuse_relu = false;
                                } else if (relu_flag == "relu6") {
                                    param.activation_param.Relu_clipped_coef = 6.f;
                                    param.activation_param.has_active = true;
                                    param.activation_param.active_type =
                                        lite_api::ActivationType::kRelu6;
                                }
                                param.paddings = std::make_shared<std::vector<int>>(paddings);
                                param.dilations = std::make_shared<std::vector<int>>(dilations);

                                kernel->SetParam(param);
                                std::unique_ptr<KernelContext> conv_context(new KernelContext);
                                context->As<ContextMetal>().CopySharedTo(
                                    &(conv_context->As<ContextMetal>()));

                                auto mt = (MetalContext*)context->As<ContextMetal>().context();
                                mt->set_metal_path(
                                    "/Users/liuzheyuan/code/Paddle-Lite/cmake-build-debug/lite/"
                                    "backends/metal/lite.metallib");
                                kernel->SetContext(std::move(conv_context));
                                // a: filter_d ==> <m, k> <=> <oc, ic>
                                // b: x_d      ==> <k, n> <=> <ic, ih*iw>
                                // c: output_d ==> <m, n> <=> <oc, ih*iw>

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

                                auto* x_data_img = x.mutable_data<float, MetalImage>(x_dim);
                                auto* filter_data = filter.mutable_data<float>();
                                auto* bias_data = bias.mutable_data<float, MetalImage>(bias_dim);

                                std::default_random_engine engine;
                                std::uniform_real_distribution<float> dist(-5, 5);

                                // prepare input
                                std::vector<float> x_data_cpu(x_dim.production());
                                for (int i = 0; i < x_dim.production(); ++i) {
                                    x_data_cpu[i] = static_cast<float>(dist(engine));
                                }
                                x_data_img->CopyFromNCHW<float>(x_data_cpu.data());

                                // preparam kernel
                                for (int i = 0; i < filter_dim.production(); ++i) {
                                    filter_data[i] = static_cast<float>(dist(engine));
                                }

                                int index = 0;
                                for (int nn = 0; nn < filter_dim[0]; ++nn) {
                                    for (int cc = 0; cc < filter_dim[1]; ++cc) {
                                        for (int hh = 0; hh < filter_dim[2]; ++hh) {
                                            for (int ww = 0; ww < filter_dim[3]; ++ww) {
                                                filter_data[index] = cc;
                                                index++;
                                            }
                                        }
                                    }
                                }

                                float* bias_cpu_data =
                                    bias_flag ? reinterpret_cast<float*>(
                                                    calloc(sizeof(float), bias_dim.production()))
                                              : nullptr;
                                if (bias_flag) {
                                    for (int i = 0; i < bias_dim.production(); ++i) {
                                        bias_cpu_data[i] = 0;  // static_cast<float>(dist(engine));
                                    }
                                    bias_data->CopyFromNCHW<float>(bias_cpu_data);
                                }

                                // run metal kernel
                                kernel->Launch();

                                // run cpu ref
                                auto* out_ref_data = out_ref.mutable_data<float>(TARGET(kHost));
                                conv_basic<float, float>(x_data_cpu.data(),
                                    out_ref_data,
                                    batch_size,
                                    oc,
                                    oh,
                                    ow,
                                    ic,
                                    ih,
                                    iw,
                                    filter_data,
                                    bias_cpu_data,  // mapped_bias,
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

                                auto* out_data = out.data<float, MetalImage>();

                                std::vector<float> out_data_cpu(out_dim.production());
                                out_data->CopyToNCHW<float>(out_data_cpu.data());
#ifdef PRINT_RESULT
                                // a: filter_d ==> <m, k> <=> <oc, ic>
                                // b: x_d      ==> <k, n> <=> <ic, ih*iw>
                                // c: output_d ==> <m, n> <=> <oc, ih*iw>
                                PrintData(
                                    "mapped_filter", static_cast<float*>(mapped_filter), m, k);
                                PrintData(
                                    "mapped_x", static_cast<float*>(mapped_x), k, n, batch_size);
                                PrintData("mapped_bias", static_cast<float*>(mapped_bias), m, 1);
                                std::cout << "mapped_bias[0]:" << mapped_bias[0] << std::endl;
                                PrintData("out_ref_data",
                                    static_cast<float*>(out_ref_data),
                                    m,
                                    n,
                                    batch_size);
                                PrintData("mapped_out",
                                    static_cast<float*>(mapped_out),
                                    m,
                                    n,
                                    batch_size);
#endif

                                for (int i = 0; i < 100; i++) {
                                    EXPECT_NEAR(out_data_cpu.data()[i], out_ref_data[i], 1e-5);
                                    if (abs(out_data_cpu.data()[i] - out_ref_data[i]) > 1e-5) {
                                        LOG(INFO) << "error idx:" << i << "sub"
                                                  << abs(out_data_cpu.data()[i] - out_ref_data[i]);
                                    }
                                }
                                if (bias_cpu_data) free(bias_cpu_data);
#ifdef LOOP_TEST
                            }  // with_relu
                        }      // with_bias
                    }          // ic
                }              // iw
            }                  // ih
        }                      // oc
    }                          // batch_size
#endif
}

}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(conv2d, kMetal, kFloat, kMetalTexture2DArray, def);
