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
void relu_compute_ref(const dtype *x_data,
                      const DDim &x_dim,
                      dtype *out_data,
                      float threshold = 0.f) {
  if (abs(threshold) < 1e-5) {
    // relu
    for (int i = 0; i < x_dim.production(); ++i) {
      out_data[i] = (x_data[i] > threshold) ? x_data[i] : threshold;
    }
  } else {
    // relu6 or relu with threshold
    for (int i = 0; i < x_dim.production(); ++i) {
      auto out_tmp = (x_data[i] > 0) ? x_data[i] : 0;
      out_data[i] = (out_tmp < threshold) ? out_tmp : threshold;
    }
  }
}

template <typename dtype>
void sigmoid_compute_ref(const dtype *x_data,
                         const DDim &x_dim,
                         dtype *out_data) {
  for (int i = 0; i < x_dim.production(); ++i) {
    out_data[i] = 1 / (1 + expf(-x_data[i]));
  }
}

// #define RELU_FP16_LOOP_TEST
// #define RELU_FP16_PRINT_RESULT
TEST(relu_image2d_fp16, compute) {
  LOG(INFO) << "main steps of test: host -> layout(buf2img) -> relu(img) -> "
               "layout(img2buf) "
               "-> host";

#ifdef RELU_FP16_LOOP_TEST
  for (int n = 1; n <= 2; n += 1) {
    for (auto c : {1}) {
      for (int h = 12; h <= 100; h += 13) {
        for (int w = 12; w <= 100; w += 25) {
#else
  const int n = 1;
  const int c = 2;
  const int h = 3;
  const int w = 4;
#endif  // RELU_FP16_LOOP_TEST

          LOG(INFO) << "======== input shape[n,c,h,w]:" << n << " " << c << " "
                    << h << " " << w << " ========";
          // set layout kernels
          auto buf_to_img_kernels =
              KernelRegistry::Global().Create("layout",
                                              TARGET(kOpenCL),
                                              PRECISION(kAny),
                                              DATALAYOUT(kImageDefault));
          auto img_to_buf_kernels = KernelRegistry::Global().Create(
              "layout", TARGET(kOpenCL), PRECISION(kAny), DATALAYOUT(kNCHW));
          auto relu_img_kernels =
              KernelRegistry::Global().Create("relu",
                                              TARGET(kOpenCL),
                                              PRECISION(kFP16),
                                              DATALAYOUT(kImageDefault));
          ASSERT_FALSE(buf_to_img_kernels.empty());
          ASSERT_FALSE(buf_to_img_kernels.empty());
          ASSERT_FALSE(relu_img_kernels.empty());

          auto buf_to_img_kernel = std::move(buf_to_img_kernels.front());
          auto img_to_buf_kernel = std::move(img_to_buf_kernels.front());
          auto relu_img_kernel = std::move(relu_img_kernels.front());
          LOG(INFO) << "get 1st kernel: " << buf_to_img_kernel->doc();
          LOG(INFO) << "get 2nd kernel: " << img_to_buf_kernel->doc();
          LOG(INFO) << "get 3rd kernel: " << relu_img_kernel->doc();

          // set tensors about op param
          LOG(INFO) << "set tensors about op param";
          // layout(buf->img): x -> relu_in
          // relu(img): relu_in -> relu_out
          // layout(img->buf): relu_out -> y
          lite::Tensor x, y, relu_in, relu_out, y_ref;
          operators::LayoutParam BufferToImageParam;
          operators::LayoutParam ImageToBufferParam;
          BufferToImageParam.x = &x;
          BufferToImageParam.y = &relu_in;
          ImageToBufferParam.x = &relu_out;
          ImageToBufferParam.y = &y;
          operators::ActivationParam ReluParam;
          ReluParam.X = &relu_in;
          ReluParam.Out = &relu_out;

          const DDim x_dim = DDim(std::vector<DDim::value_type>{n, c, h, w});
          x.Resize(x_dim);
          y.Resize(x_dim);
          relu_in.Resize(x_dim);
          relu_out.Resize(x_dim);
          y_ref.Resize(x_dim);
          auto relu_image2d_shape =
              paddle::lite::kernels::opencl::InitImageDimInfoWith(x_dim);

          // initialize tensors
          LOG(INFO) << "initialize tensors";
          auto *x_data = x.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
          auto *y_data = y.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
          auto *y_data_ref = y_ref.mutable_data<float>(TARGET(kARM));
          auto *mapped_x = static_cast<float *>(TargetWrapperCL::Map(
              x_data, 0, sizeof(float) * x_dim.production()));
          auto *mapped_y = static_cast<float *>(TargetWrapperCL::Map(
              y_data, 0, sizeof(float) * x_dim.production()));
          for (int i = 0; i < x_dim.production(); ++i) {
            mapped_x[i] = static_cast<int>(i) - x_dim.production() / 2;
            mapped_y[i] = static_cast<int>(0);
          }
          auto *relu_in_data = relu_in.mutable_data<half_t, cl::Image2D>(
              relu_image2d_shape["width"], relu_image2d_shape["height"]);
          auto *relu_out_data = relu_out.mutable_data<half_t, cl::Image2D>(
              relu_image2d_shape["width"], relu_image2d_shape["height"]);

          // set context and kernel args
          LOG(INFO) << "set context and kernel args";
          std::unique_ptr<KernelContext> context(new KernelContext);
          context->As<OpenCLContext>().InitOnce();

          buf_to_img_kernel->SetParam(BufferToImageParam);
          std::unique_ptr<KernelContext> buf_to_img_context(new KernelContext);
          context->As<OpenCLContext>().CopySharedTo(
              &(buf_to_img_context->As<OpenCLContext>()));
          buf_to_img_kernel->SetContext(std::move(buf_to_img_context));

          img_to_buf_kernel->SetParam(ImageToBufferParam);
          std::unique_ptr<KernelContext> img_to_buf_context(new KernelContext);
          context->As<OpenCLContext>().CopySharedTo(
              &(img_to_buf_context->As<OpenCLContext>()));
          img_to_buf_kernel->SetContext(std::move(img_to_buf_context));

          relu_img_kernel->SetParam(ReluParam);
          std::unique_ptr<KernelContext> relu_img_context(new KernelContext);
          context->As<OpenCLContext>().CopySharedTo(
              &(relu_img_context->As<OpenCLContext>()));
          relu_img_kernel->SetContext(std::move(relu_img_context));

          // run kernels
          LOG(INFO) << "run kernel: buf_to_img_kernel";
          buf_to_img_kernel->Launch();
          LOG(INFO) << "run kernel: relu_img_kernel";
          relu_img_kernel->Launch();
          LOG(INFO) << "run kernel: img_to_buf_kernel";
          img_to_buf_kernel->Launch();

          // wait for opencl
          auto *wait_list = context->As<OpenCLContext>().cl_wait_list();
          auto *out_ptr = ImageToBufferParam.y->data<float, cl::Buffer>();
          auto it = wait_list->find(out_ptr);

          if (it != wait_list->end()) {
            VLOG(4) << "--- Find the sync event for the target cl "
                       "tensor. ---";
            auto &event = *(it->second);
            event.wait();
          } else {
            LOG(FATAL) << "Could not find the sync event for the target "
                          "cl tensor.";
          }

          // compute ref cpu
          relu_compute_ref<float>(mapped_x, x_dim, y_data_ref);
// result
#ifdef RELU_FP16_PRINT_RESULT
          LOG(INFO) << "---- print kernel result (input -> output) ----";
          for (int eidx = 0; eidx < x_dim.production(); ++eidx) {
            std::cout << mapped_x[eidx] << " -> " << mapped_y[eidx]
                      << ", ref: " << y_data_ref[eidx] << std::endl;
          }
#endif  // RELU_FP16_PRINT_RESULT

          // check result: compare kernel output and cpu output(y_data_ref)
          for (int eidx = 0; eidx < x_dim.production(); ++eidx) {
            auto abs_diff = COMPUTE_ABS_DIFF(y_data_ref[eidx], mapped_y[eidx]);
            auto relative_diff =
                COMPUTE_RELATIVE_DIFF(y_data_ref[eidx], mapped_y[eidx]);
            EXPECT_EQ(
                (relative_diff <= FP16_MAX_DIFF) || (abs_diff <= FP16_MAX_DIFF),
                true);
            if ((relative_diff > FP16_MAX_DIFF) && (abs_diff > FP16_MAX_DIFF)) {
              LOG(ERROR) << "error idx:" << eidx << ", y_data_ref[" << eidx
                         << "]:" << y_data_ref[eidx] << ", mapped_y[" << eidx
                         << "]:" << mapped_y[eidx] << " abs_diff:" << abs_diff
                         << " relative_diff:" << relative_diff
                         << " FP16_MAX_DIFF:" << FP16_MAX_DIFF;
              break;
            }
          }

          // free
          LOG(INFO) << "free: unmap x, y";
          TargetWrapperCL::Unmap(x_data, mapped_x);
          TargetWrapperCL::Unmap(y_data, mapped_y);
#ifdef RELU_FP16_LOOP_TEST
        }  // w
      }    // h
    }      // c
  }        // n
#else
// nothing to do.
#endif
}

//  #define RELU6_FP16_LOOP_TEST
// #define RELU6_FP16_PRINT_RESULT
TEST(relu6_image2d_fp16, compute) {
  LOG(INFO) << "main steps of test: host -> layout(buf2img) -> relu6(img) -> "
               "layout(img2buf) "
               "-> host";

#ifdef RELU6_FP16_LOOP_TEST
  for (int n = 1; n <= 100; n += 33) {
    for (auto c : {1, 3}) {
      for (int h = 12; h <= 100; h += 13) {
        for (int w = 12; w <= 100; w += 25) {
#else
          const int n = 1;
          const int c = 2;
          const int h = 3;
          const int w = 4;
#endif  // RELU6_FP16_LOOP_TEST

          LOG(INFO) << "======== input shape[n,c,h,w]:" << n << " " << c << " "
                    << h << " " << w << " ========";
          // set layout kernels
          auto buf_to_img_kernels =
              KernelRegistry::Global().Create("layout",
                                              TARGET(kOpenCL),
                                              PRECISION(kAny),
                                              DATALAYOUT(kImageDefault));
          auto img_to_buf_kernels = KernelRegistry::Global().Create(
              "layout", TARGET(kOpenCL), PRECISION(kAny), DATALAYOUT(kNCHW));
          auto relu_img_kernels =
              KernelRegistry::Global().Create("relu6",
                                              TARGET(kOpenCL),
                                              PRECISION(kFP16),
                                              DATALAYOUT(kImageDefault));
          ASSERT_FALSE(buf_to_img_kernels.empty());
          ASSERT_FALSE(buf_to_img_kernels.empty());
          ASSERT_FALSE(relu_img_kernels.empty());

          auto buf_to_img_kernel = std::move(buf_to_img_kernels.front());
          auto img_to_buf_kernel = std::move(img_to_buf_kernels.front());
          auto relu_img_kernel = std::move(relu_img_kernels.front());
          LOG(INFO) << "get 1st kernel: " << buf_to_img_kernel->doc();
          LOG(INFO) << "get 2nd kernel: " << img_to_buf_kernel->doc();
          LOG(INFO) << "get 3rd kernel: " << relu_img_kernel->doc();

          // set tensors about op param
          LOG(INFO) << "set tensors about op param";
          // layout(buf->img): x -> relu_in
          // relu(img): relu_in -> relu_out
          // layout(img->buf): relu_out -> y
          lite::Tensor x, y, relu_in, relu_out, y_ref;
          operators::LayoutParam BufferToImageParam;
          operators::LayoutParam ImageToBufferParam;
          BufferToImageParam.x = &x;
          BufferToImageParam.y = &relu_in;
          ImageToBufferParam.x = &relu_out;
          ImageToBufferParam.y = &y;
          operators::ActivationParam ReluParam;
          ReluParam.X = &relu_in;
          ReluParam.Out = &relu_out;
          ReluParam.Relu_clipped_coef = 6.f;

          const DDim x_dim = DDim(std::vector<DDim::value_type>{n, c, h, w});
          x.Resize(x_dim);
          y.Resize(x_dim);
          relu_in.Resize(x_dim);
          relu_out.Resize(x_dim);
          y_ref.Resize(x_dim);
          auto relu_image2d_shape =
              paddle::lite::kernels::opencl::InitImageDimInfoWith(x_dim);

          // initialize tensors
          LOG(INFO) << "initialize tensors";
          auto *x_data = x.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
          auto *y_data = y.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
          auto *y_data_ref = y_ref.mutable_data<float>(TARGET(kARM));
          auto *mapped_x = static_cast<float *>(TargetWrapperCL::Map(
              x_data, 0, sizeof(float) * x_dim.production()));
          auto *mapped_y = static_cast<float *>(TargetWrapperCL::Map(
              y_data, 0, sizeof(float) * x_dim.production()));
          for (int i = 0; i < x_dim.production(); ++i) {
            mapped_x[i] = static_cast<int>(i) - x_dim.production() / 2 * 0.1;
            mapped_y[i] = static_cast<int>(0);
          }
          auto *relu_in_data = relu_in.mutable_data<half_t, cl::Image2D>(
              relu_image2d_shape["width"], relu_image2d_shape["height"]);
          auto *relu_out_data = relu_out.mutable_data<half_t, cl::Image2D>(
              relu_image2d_shape["width"], relu_image2d_shape["height"]);

          // set context and kernel args
          LOG(INFO) << "set context and kernel args";
          std::unique_ptr<KernelContext> context(new KernelContext);
          context->As<OpenCLContext>().InitOnce();

          buf_to_img_kernel->SetParam(BufferToImageParam);
          std::unique_ptr<KernelContext> buf_to_img_context(new KernelContext);
          context->As<OpenCLContext>().CopySharedTo(
              &(buf_to_img_context->As<OpenCLContext>()));
          buf_to_img_kernel->SetContext(std::move(buf_to_img_context));

          img_to_buf_kernel->SetParam(ImageToBufferParam);
          std::unique_ptr<KernelContext> img_to_buf_context(new KernelContext);
          context->As<OpenCLContext>().CopySharedTo(
              &(img_to_buf_context->As<OpenCLContext>()));
          img_to_buf_kernel->SetContext(std::move(img_to_buf_context));

          relu_img_kernel->SetParam(ReluParam);
          std::unique_ptr<KernelContext> relu_img_context(new KernelContext);
          context->As<OpenCLContext>().CopySharedTo(
              &(relu_img_context->As<OpenCLContext>()));
          relu_img_kernel->SetContext(std::move(relu_img_context));

          // run kernels
          LOG(INFO) << "run kernel: buf_to_img_kernel";
          buf_to_img_kernel->Launch();
          LOG(INFO) << "run kernel: relu_img_kernel";
          relu_img_kernel->Launch();
          LOG(INFO) << "run kernel: img_to_buf_kernel";
          img_to_buf_kernel->Launch();

          // wait for opencl
          auto *wait_list = context->As<OpenCLContext>().cl_wait_list();
          auto *out_ptr = ImageToBufferParam.y->data<float, cl::Buffer>();
          auto it = wait_list->find(out_ptr);

          if (it != wait_list->end()) {
            VLOG(4) << "--- Find the sync event for the target cl "
                       "tensor. ---";
            auto &event = *(it->second);
            event.wait();
          } else {
            LOG(FATAL) << "Could not find the sync event for the target "
                          "cl tensor.";
          }

          // compute ref cpu
          relu_compute_ref<float>(mapped_x, x_dim, y_data_ref, 6.f);
// result
#ifdef RELU6_FP16_PRINT_RESULT
          LOG(INFO) << "---- print kernel result (input -> output) ----";
          for (int eidx = 0; eidx < x_dim.production(); ++eidx) {
            std::cout << mapped_x[eidx] << " -> " << mapped_y[eidx]
                      << ", ref: " << y_data_ref[eidx] << std::endl;
          }
#endif  // RELU6_FP16_PRINT_RESULT

          // check result: compare kernel output and cpu output(y_data_ref)
          for (int eidx = 0; eidx < x_dim.production(); eidx++) {
            EXPECT_NEAR(y_data_ref[eidx], mapped_y[eidx], FP16_MAX_DIFF);
            if (abs(y_data_ref[eidx] - mapped_y[eidx]) > FP16_MAX_DIFF) {
              LOG(INFO) << "1st diff in this case at eidx[from 0]:" << eidx
                        << " / " << x_dim.production() << ", y_data_ref["
                        << eidx << "]:" << y_data_ref[eidx] << ", mapped_y["
                        << eidx << "]:" << mapped_y[eidx];
              break;
            }
          }

          // free
          LOG(INFO) << "free: unmap x, y";
          TargetWrapperCL::Unmap(x_data, mapped_x);
          TargetWrapperCL::Unmap(y_data, mapped_y);
#ifdef RELU6_FP16_LOOP_TEST
        }  // w
      }    // h
    }      // c
  }        // n
#else
// nothing to do.
#endif
}

// #define SIGMOID_FP16_LOOP_TEST
// #define SIGMOID_FP16_PRINT_RESULT
TEST(sigmoid_image2d_fp16, compute) {
  LOG(INFO) << "main steps of test: host -> layout(buf2img) -> sigmoid(img) -> "
               "layout(img2buf) "
               "-> host";

#ifdef SIGMOID_FP16_LOOP_TEST
  for (int n = 1; n <= 100; n += 33) {
    for (auto c : {1, 3}) {
      for (int h = 12; h <= 100; h += 13) {
        for (int w = 12; w <= 100; w += 25) {
#else
          const int n = 1;
          const int c = 2;
          const int h = 3;
          const int w = 4;
#endif  // SIGMOID_FP16_LOOP_TEST

          LOG(INFO) << "======== input shape[n,c,h,w]:" << n << " " << c << " "
                    << h << " " << w << " ========";
          // set layout kernels
          auto buf_to_img_kernels =
              KernelRegistry::Global().Create("layout",
                                              TARGET(kOpenCL),
                                              PRECISION(kAny),
                                              DATALAYOUT(kImageDefault));
          auto img_to_buf_kernels = KernelRegistry::Global().Create(
              "layout", TARGET(kOpenCL), PRECISION(kAny), DATALAYOUT(kNCHW));
          auto sigmoid_img_kernels =
              KernelRegistry::Global().Create("sigmoid",
                                              TARGET(kOpenCL),
                                              PRECISION(kFP16),
                                              DATALAYOUT(kImageDefault));
          ASSERT_FALSE(buf_to_img_kernels.empty());
          ASSERT_FALSE(buf_to_img_kernels.empty());
          ASSERT_FALSE(sigmoid_img_kernels.empty());

          auto buf_to_img_kernel = std::move(buf_to_img_kernels.front());
          auto img_to_buf_kernel = std::move(img_to_buf_kernels.front());
          auto sigmoid_img_kernel = std::move(sigmoid_img_kernels.front());
          LOG(INFO) << "get 1st kernel: " << buf_to_img_kernel->doc();
          LOG(INFO) << "get 2nd kernel: " << img_to_buf_kernel->doc();
          LOG(INFO) << "get 3rd kernel: " << sigmoid_img_kernel->doc();

          // set tensors about op param
          LOG(INFO) << "set tensors about op param";
          // layout(buf->img): x -> sigmoid_in
          // sigmoid(img): sigmoid_in -> sigmoid_out
          // layout(img->buf): sigmoid_out -> y
          lite::Tensor x, y, sigmoid_in, sigmoid_out, y_ref;
          operators::LayoutParam BufferToImageParam;
          operators::LayoutParam ImageToBufferParam;
          BufferToImageParam.x = &x;
          BufferToImageParam.y = &sigmoid_in;
          ImageToBufferParam.x = &sigmoid_out;
          ImageToBufferParam.y = &y;
          operators::ActivationParam SigmoidParam;
          SigmoidParam.X = &sigmoid_in;
          SigmoidParam.Out = &sigmoid_out;

          const DDim x_dim = DDim(std::vector<DDim::value_type>{n, c, h, w});
          x.Resize(x_dim);
          y.Resize(x_dim);
          sigmoid_in.Resize(x_dim);
          sigmoid_out.Resize(x_dim);
          y_ref.Resize(x_dim);
          auto sigmoid_image2d_shape =
              paddle::lite::kernels::opencl::InitImageDimInfoWith(x_dim);

          // initialize tensors
          LOG(INFO) << "initialize tensors";
          auto *x_data = x.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
          auto *y_data = y.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
          auto *y_data_ref = y_ref.mutable_data<float>(TARGET(kARM));
          auto *mapped_x = static_cast<float *>(TargetWrapperCL::Map(
              x_data, 0, sizeof(float) * x_dim.production()));
          auto *mapped_y = static_cast<float *>(TargetWrapperCL::Map(
              y_data, 0, sizeof(float) * x_dim.production()));
          std::default_random_engine engine;
          std::uniform_real_distribution<float> dist(-1, 1);
          for (int i = 0; i < x_dim.production(); ++i) {
            mapped_x[i] = static_cast<float>(dist(engine));
          }
          auto *sigmoid_in_data = sigmoid_in.mutable_data<half_t, cl::Image2D>(
              sigmoid_image2d_shape["width"], sigmoid_image2d_shape["height"]);
          auto *sigmoid_out_data =
              sigmoid_out.mutable_data<half_t, cl::Image2D>(
                  sigmoid_image2d_shape["width"],
                  sigmoid_image2d_shape["height"]);

          // set context and kernel args
          LOG(INFO) << "set context and kernel args";
          std::unique_ptr<KernelContext> context(new KernelContext);
          context->As<OpenCLContext>().InitOnce();

          buf_to_img_kernel->SetParam(BufferToImageParam);
          std::unique_ptr<KernelContext> buf_to_img_context(new KernelContext);
          context->As<OpenCLContext>().CopySharedTo(
              &(buf_to_img_context->As<OpenCLContext>()));
          buf_to_img_kernel->SetContext(std::move(buf_to_img_context));

          img_to_buf_kernel->SetParam(ImageToBufferParam);
          std::unique_ptr<KernelContext> img_to_buf_context(new KernelContext);
          context->As<OpenCLContext>().CopySharedTo(
              &(img_to_buf_context->As<OpenCLContext>()));
          img_to_buf_kernel->SetContext(std::move(img_to_buf_context));

          sigmoid_img_kernel->SetParam(SigmoidParam);
          std::unique_ptr<KernelContext> sigmoid_img_context(new KernelContext);
          context->As<OpenCLContext>().CopySharedTo(
              &(sigmoid_img_context->As<OpenCLContext>()));
          sigmoid_img_kernel->SetContext(std::move(sigmoid_img_context));

          // run kernels
          LOG(INFO) << "run kernel: buf_to_img_kernel";
          buf_to_img_kernel->Launch();
          LOG(INFO) << "run kernel: sigmoid_img_kernel";
          sigmoid_img_kernel->Launch();
          LOG(INFO) << "run kernel: img_to_buf_kernel";
          img_to_buf_kernel->Launch();

          // wait for opencl
          auto *wait_list = context->As<OpenCLContext>().cl_wait_list();
          auto *out_ptr = ImageToBufferParam.y->data<float, cl::Buffer>();
          auto it = wait_list->find(out_ptr);

          if (it != wait_list->end()) {
            VLOG(4) << "--- Find the sync event for the target cl "
                       "tensor. ---";
            auto &event = *(it->second);
            event.wait();
          } else {
            LOG(FATAL) << "Could not find the sync event for the target "
                          "cl tensor.";
          }

          // compute ref cpu
          sigmoid_compute_ref<float>(mapped_x, x_dim, y_data_ref);
// result
#ifdef SIGMOID_FP16_PRINT_RESULT
          LOG(INFO) << "---- print kernel result (input -> output) ----";
          for (int eidx = 0; eidx < x_dim.production(); ++eidx) {
            std::cout << mapped_x[eidx] << " -> " << mapped_y[eidx]
                      << ", ref:" << y_data_ref[eidx] << std::endl;
          }
#endif  // SIGMOID_FP16_PRINT_RESULT

          // check result: compare kernel output and cpu output(y_data_ref)
          for (int eidx = 0; eidx < x_dim.production(); eidx++) {
            EXPECT_NEAR(y_data_ref[eidx], mapped_y[eidx], FP16_MAX_DIFF);
            if (abs(y_data_ref[eidx] - mapped_y[eidx]) > FP16_MAX_DIFF) {
              LOG(INFO) << "1st diff in this case at eidx[from 0]:" << eidx
                        << " / " << x_dim.production() << ", y_data_ref["
                        << eidx << "]: " << y_data_ref[eidx] << ", mapped_y["
                        << eidx << "]: " << mapped_y[eidx] << ", mapped_x["
                        << eidx << "]: " << mapped_x[eidx];
              break;
            }
          }

          // free
          LOG(INFO) << "free: unmap x, y";
          TargetWrapperCL::Unmap(x_data, mapped_x);
          TargetWrapperCL::Unmap(y_data, mapped_y);
#ifdef SIGMOID_FP16_LOOP_TEST
        }  // w
      }    // h
    }      // c
  }        // n
#else
// nothing to do.
#endif
}

}  // namespace lite
}  // namespace paddle

// layout
USE_LITE_KERNEL(layout, kOpenCL, kAny, kImageDefault, NCHW_to_ImageDefault);
USE_LITE_KERNEL(layout, kOpenCL, kAny, kNCHW, ImageDefault_to_NCHW);

// relu image2d fp16
USE_LITE_KERNEL(relu, kOpenCL, kFP16, kImageDefault, ImageDefault);

// relu6 image2d fp16
USE_LITE_KERNEL(relu6, kOpenCL, kFP16, kImageDefault, ImageDefault);

// sigmoid image2d fp16
USE_LITE_KERNEL(sigmoid, kOpenCL, kFP16, kImageDefault, ImageDefault);
