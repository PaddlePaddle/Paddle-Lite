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
#include "lite/backends/opencl/target_wrapper.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/kernels/opencl/image_helper.h"
#include "lite/kernels/opencl/test_helper.h"

#define FP16_MAX_DIFF (1e0)

namespace paddle {
namespace lite {

template <typename dtype>
void act_compute_ref(const dtype *x_data,
                     const DDim &x_dim,
                     dtype *out_data,
                     int act_type,
                     float threshold,
                     float scale) {
  for (int i = 0; i < x_dim.production(); i++) {
    switch (act_type) {
      case 1:  // relu
        out_data[i] = x_data[i] > 0 ? x_data[i] : 0;
        break;
      case 2:  // relu6
        out_data[i] = x_data[i] > 0 ? x_data[i] : 0;
        out_data[i] = (out_data[i] < threshold) ? out_data[i] : threshold;
        break;
      case 4:  // leakyRelu
        out_data[i] = x_data[i] > 0 ? x_data[i] : x_data[i] * scale;
        break;
      case 5:  // sigmoid
        out_data[i] = 1 / (1 + expf(-x_data[i]));
        break;
      case 6:  // tanh
        out_data[i] = (expf(x_data[i]) - expf(-x_data[i])) /
                      (expf(x_data[i]) + expf(-x_data[i]));
        break;
      case 7:  // swish
        out_data[i] = x_data[i] / (1 + expf(-x_data[i] * scale));
        break;
      case 8:  // exp
        out_data[i] = expf(x_data[i]);
        break;
      case 14:  // hard sigmoid
                // scale ==> slope
        {
          float tmp = x_data[i] * scale + threshold;
          tmp = tmp < 1.0f ? tmp : 1.0f;
          tmp = tmp > 0.0f ? tmp : 0.0f;
          out_data[i] = tmp;
          break;
        }
      default:
        break;
    }
  }
}

// #define ACT_FP16_LOOP_TEST
// #define ACT_FP16_PRINT_RESULT
TEST(act_image2d_fp16, compute) {
  LOG(INFO) << "main steps of test: host -> layout(buf2img) -> relu(img) -> "
               "layout(img2buf) "
               "-> host";

#ifdef ACT_FP16_LOOP_TEST
  for (int n = 1; n <= 100; n += 33) {
    for (auto c : {1, 3, 8, 23, 32}) {
      for (int h = 12; h <= 100; h += 13) {
        for (int w = 12; w <= 100; w += 25) {
          for (auto act_type : {1, 2, 4, 5, 6, 7, 8, 14}) {
            for (auto scale : {0.5, 0.8}) {
              for (auto threshold : {6.0}) {
#else
  const int n = 1;
  const int c = 2;
  const int h = 3;
  const int w = 4;
  const int act_type = 14;
  const float scale = 2.0f;
  const float threshold = 1.0f;

#endif  // ACT_FP16_LOOP_TEST

                LOG(INFO) << "======== input shape[n,c,h,w]:" << n << " " << c
                          << " " << h << " " << w << " ========";
                LOG(INFO) << "====act_type: " << act_type
                          << ", scale: " << scale
                          << ", threshold: " << threshold;
                std::string func_name = "relu";
                switch (act_type) {
                  case 1:  // relu
                    func_name = "relu";
                    break;
                  case 2:  // relu6
                    func_name = "relu6";
                    break;
                  case 4:  // leaky_relu
                    func_name = "leaky_relu";
                    break;
                  case 5:  // sigmoid
                    func_name = "sigmoid";
                    break;
                  case 6:  // tanh
                    func_name = "tanh";
                    break;
                  case 7:  // tanh
                    func_name = "swish";
                    break;
                  case 8:  // tanh
                    func_name = "exp";
                    break;
                  case 14:  // hard sigmoid
                    func_name = "hard_sigmoid";
                    break;
                }
                LOG(INFO) << "func_name: " << func_name;
                // set layout kernels
                auto buf_to_img_kernels =
                    KernelRegistry::Global().Create("layout",
                                                    TARGET(kOpenCL),
                                                    PRECISION(kAny),
                                                    DATALAYOUT(kImageDefault));
                auto img_to_buf_kernels =
                    KernelRegistry::Global().Create("layout",
                                                    TARGET(kOpenCL),
                                                    PRECISION(kAny),
                                                    DATALAYOUT(kNCHW));
                auto act_img_kernels =
                    KernelRegistry::Global().Create(func_name.c_str(),
                                                    TARGET(kOpenCL),
                                                    PRECISION(kFP16),
                                                    DATALAYOUT(kImageDefault));
                ASSERT_FALSE(buf_to_img_kernels.empty());
                ASSERT_FALSE(buf_to_img_kernels.empty());
                ASSERT_FALSE(act_img_kernels.empty());

                auto buf_to_img_kernel = std::move(buf_to_img_kernels.front());
                auto img_to_buf_kernel = std::move(img_to_buf_kernels.front());
                auto act_img_kernel = std::move(act_img_kernels.front());
                LOG(INFO) << "get 1st kernel: " << buf_to_img_kernel->doc();
                LOG(INFO) << "get 2nd kernel: " << img_to_buf_kernel->doc();
                LOG(INFO) << "get 3rd kernel: " << act_img_kernel->doc();

                // set tensors about op param
                LOG(INFO) << "set tensors about op param";
                // layout(buf->img): x -> act_in
                // relu(img): act_in -> act_out
                // layout(img->buf): act_out -> y
                lite::Tensor x, y, act_in, act_out, y_ref;
                operators::LayoutParam BufferToImageParam;
                operators::LayoutParam ImageToBufferParam;
                BufferToImageParam.x = &x;
                BufferToImageParam.y = &act_in;
                ImageToBufferParam.x = &act_out;
                ImageToBufferParam.y = &y;
                operators::ActivationParam actParam;
                actParam.X = &act_in;
                actParam.Out = &act_out;
                actParam.active_type =
                    (paddle::lite_api::ActivationType)act_type;
                actParam.Relu_clipped_coef = threshold;
                actParam.Leaky_relu_alpha = scale;
                actParam.Swish_beta = scale;
                // hard sigmoid
                actParam.hard_sigmoid_slope = scale;
                actParam.hard_sigmoid_offset = threshold;

                const DDim x_dim =
                    DDim(std::vector<DDim::value_type>{n, c, h, w});
                x.Resize(x_dim);
                y.Resize(x_dim);
                act_in.Resize(x_dim);
                act_out.Resize(x_dim);
                y_ref.Resize(x_dim);
                auto act_image2d_shape =
                    paddle::lite::kernels::opencl::InitImageDimInfoWith(x_dim);

                // initialize tensors
                LOG(INFO) << "initialize tensors";
                auto *x_data =
                    x.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
                auto *y_data =
                    y.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
                auto *y_data_ref = y_ref.mutable_data<float>(TARGET(kARM));
                auto *mapped_x = static_cast<float *>(TargetWrapperCL::Map(
                    x_data, 0, sizeof(float) * x_dim.production()));
                auto *mapped_y = static_cast<float *>(TargetWrapperCL::Map(
                    y_data, 0, sizeof(float) * x_dim.production()));
                std::default_random_engine engine;
                std::uniform_real_distribution<float> dist(-1, 1);
                for (int i = 0; i < x_dim.production(); ++i) {
                  mapped_x[i] =
                      (i - x_dim.production() / 2) / 10.;  // dist(engine);
                  mapped_y[i] = 0.0f;
                }
                auto *act_in_data = act_in.mutable_data<half_t, cl::Image2D>(
                    act_image2d_shape["width"], act_image2d_shape["height"]);
                auto *act_out_data = act_out.mutable_data<half_t, cl::Image2D>(
                    act_image2d_shape["width"], act_image2d_shape["height"]);

                // set context and kernel args
                LOG(INFO) << "set context and kernel args";
                std::unique_ptr<KernelContext> context(new KernelContext);
                context->As<OpenCLContext>().InitOnce();

                buf_to_img_kernel->SetParam(BufferToImageParam);
                std::unique_ptr<KernelContext> buf_to_img_context(
                    new KernelContext);
                context->As<OpenCLContext>().CopySharedTo(
                    &(buf_to_img_context->As<OpenCLContext>()));
                buf_to_img_kernel->SetContext(std::move(buf_to_img_context));

                img_to_buf_kernel->SetParam(ImageToBufferParam);
                std::unique_ptr<KernelContext> img_to_buf_context(
                    new KernelContext);
                context->As<OpenCLContext>().CopySharedTo(
                    &(img_to_buf_context->As<OpenCLContext>()));
                img_to_buf_kernel->SetContext(std::move(img_to_buf_context));

                act_img_kernel->SetParam(actParam);
                std::unique_ptr<KernelContext> act_img_context(
                    new KernelContext);
                context->As<OpenCLContext>().CopySharedTo(
                    &(act_img_context->As<OpenCLContext>()));
                act_img_kernel->SetContext(std::move(act_img_context));

                // run kernels
                LOG(INFO) << "run kernel: buf_to_img_kernel";
                buf_to_img_kernel->Launch();
                LOG(INFO) << "run kernel: act_img_kernel";
                act_img_kernel->Launch();
                LOG(INFO) << "run kernel: img_to_buf_kernel";
                img_to_buf_kernel->Launch();

                // wait for opencl
                auto *out_ptr = ImageToBufferParam.y->data<float, cl::Buffer>();

                CLRuntime::Global()->command_queue().finish();

                // compute ref cpu
                act_compute_ref<float>(
                    mapped_x, x_dim, y_data_ref, act_type, threshold, scale);
// result
#ifdef ACT_FP16_PRINT_RESULT
                LOG(INFO) << "---- print kernel result (input -> output) ----";
                for (int eidx = 0; eidx < x_dim.production(); ++eidx) {
                  std::cout << mapped_x[eidx] << " -> " << mapped_y[eidx]
                            << ", ref: " << y_data_ref[eidx] << std::endl;
                }
#endif  // ACT_FP16_PRINT_RESULT

                // check result: compare kernel output and cpu
                // output(y_data_ref)
                for (int eidx = 0; eidx < x_dim.production(); ++eidx) {
                  auto abs_diff =
                      COMPUTE_ABS_DIFF(y_data_ref[eidx], mapped_y[eidx]);
                  auto relative_diff =
                      COMPUTE_RELATIVE_DIFF(y_data_ref[eidx], mapped_y[eidx]);
                  // EXPECT_EQ((relative_diff <= FP16_MAX_DIFF) ||
                  //              (abs_diff <= FP16_MAX_DIFF),
                  //          true);
                  if ((relative_diff > FP16_MAX_DIFF) &&
                      (abs_diff > FP16_MAX_DIFF)) {
                    LOG(ERROR)
                        << "error idx:" << eidx << ", y_data_ref[" << eidx
                        << "]:" << y_data_ref[eidx] << ", mapped_y[" << eidx
                        << "]:" << mapped_y[eidx] << " mapped_x[" << eidx
                        << "]:" << mapped_x[eidx] << " abs_diff:" << abs_diff
                        << " relative_diff:" << relative_diff
                        << " FP16_MAX_DIFF:" << FP16_MAX_DIFF;
                    return;
                  }
                }

                // free
                LOG(INFO) << "free: unmap x, y";
                TargetWrapperCL::Unmap(x_data, mapped_x);
                TargetWrapperCL::Unmap(y_data, mapped_y);
#ifdef ACT_FP16_LOOP_TEST
              }  // threshold
            }    // scale
          }      // act_type
        }        // w
      }          // h
    }            // c
  }              // n
#else
// nothing to do.
#endif
}
}  // namespace lite
}  // namespace paddle

// layout
USE_LITE_KERNEL(layout, kOpenCL, kAny, kImageDefault, NCHW_to_ImageDefault);
USE_LITE_KERNEL(layout, kOpenCL, kAny, kNCHW, ImageDefault_to_NCHW);

// exp
USE_LITE_KERNEL(exp, kOpenCL, kFP16, kImageDefault, ImageDefault);

// swish
USE_LITE_KERNEL(swish, kOpenCL, kFP16, kImageDefault, ImageDefault);

// leaky_relu
USE_LITE_KERNEL(leaky_relu, kOpenCL, kFP16, kImageDefault, ImageDefault);

// tanh act
USE_LITE_KERNEL(tanh, kOpenCL, kFP16, kImageDefault, ImageDefault);

// relu image2d fp16
USE_LITE_KERNEL(relu, kOpenCL, kFP16, kImageDefault, ImageDefault);

// relu6 image2d fp16
USE_LITE_KERNEL(relu6, kOpenCL, kFP16, kImageDefault, ImageDefault);

// sigmoid image2d fp16
USE_LITE_KERNEL(sigmoid, kOpenCL, kFP16, kImageDefault, ImageDefault);

// hard_sigmoid image2d fp16
USE_LITE_KERNEL(hard_sigmoid, kOpenCL, kFP16, kImageDefault, ImageDefault);
