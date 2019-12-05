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

namespace paddle {
namespace lite {

template <typename dtype>
void relu_compute_ref(const dtype *x_data, const DDim &x_dim, dtype *out_data) {
  for (int i = 0; i < x_dim.production(); ++i) {
    out_data[i] = x_data[i] > 0.f ? x_data[i] : 0.f;
  }
}

TEST(opencl_relu_buffer, compute) {
  // prepare data
  const DDim x_dim = DDim(std::vector<DDim::value_type>{3, 6, 10, 10});
  lite::Tensor x, out;
  x.Resize(x_dim);
  out.Resize(x_dim);

  auto *x_data = x.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
  std::default_random_engine engine;
  std::uniform_real_distribution<float> dist(-10, 10);
  auto *mapped_x = static_cast<float *>(
      TargetWrapperCL::Map(x_data, 0, sizeof(float) * x_dim.production()));
  for (int i = 0; i < x_dim.production(); i++) {
    mapped_x[i] = dist(engine);
  }

  // set param and kernel, then run
  operators::ActivationParam param;
  param.X = &x;
  param.Out = &out;

  std::unique_ptr<KernelContext> context(new KernelContext);
  context->As<OpenCLContext>().InitOnce();
  auto kernels = KernelRegistry::Global().Create(
      "relu", TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNCHW));
  ASSERT_FALSE(kernels.empty());
  auto kernel = std::move(kernels.front());
  kernel->SetParam(param);
  std::unique_ptr<KernelContext> relu_context(new KernelContext);
  context->As<OpenCLContext>().CopySharedTo(
      &(relu_context->As<OpenCLContext>()));
  kernel->SetContext(std::move(relu_context));

  kernel->Launch();

  auto *wait_list = context->As<OpenCLContext>().cl_wait_list();
  auto *out_ptr = param.Out->data<float, cl::Buffer>();
  auto it = wait_list->find(out_ptr);
  if (it != wait_list->end()) {
    VLOG(4) << "--- Find the sync event for the target cl tensor. ---";
    auto &event = *(it->second);
    event.wait();
  } else {
    LOG(FATAL) << "Could not find the sync event for the target cl tensor.";
  }

  // run compute ref and check
  std::unique_ptr<float[]> out_ref(new float[x_dim.production()]);
  relu_compute_ref<float>(mapped_x, x_dim, out_ref.get());

  auto *out_data = out.mutable_data<float, cl::Buffer>();
  auto *mapped_out = static_cast<float *>(
      TargetWrapperCL::Map(out_data, 0, sizeof(float) * x_dim.production()));
  for (int i = 0; i < x_dim.production(); i++) {
    EXPECT_NEAR(mapped_out[i], out_ref[i], 1e-6);
  }
  TargetWrapperCL::Unmap(out_data, mapped_out);
  TargetWrapperCL::Unmap(x_data, mapped_x);
}

// #define LOOP_TEST
// #define PRINT_RESULT
TEST(layout, compute) {
  LOG(INFO) << "main steps of test: host -> layout(buf2img) -> layout(img2buf) "
               "-> device";

#ifdef LOOP_TEST
  for (int n = 1; n <= 100; n += 21) {
    for (auto c : {1, 3}) {
      for (int h = 1; h <= 100; h += 13) {
        for (int w = 1; w <= 100; w += 17) {
#else
  const int n = 1;
  const int c = 1;
  const int h = 1;
  const int w = 100;
#endif  // LOOP_TEST

          LOG(INFO) << "======== input shape[n,c,h,w]:" << n << " " << c << " "
                    << h << " " << w << " ========";
          // set layout kernels
          auto buf_to_img_kernels = KernelRegistry::Global().Create(
              "layout", TARGET(kOpenCL), PRECISION(kAny), DATALAYOUT(kNHWC));
          auto img_to_buf_kernels = KernelRegistry::Global().Create(
              "layout", TARGET(kOpenCL), PRECISION(kAny), DATALAYOUT(kNCHW));
          auto relu_img_kernels = KernelRegistry::Global().Create(
              "relu", TARGET(kOpenCL), PRECISION(kAny), DATALAYOUT(kNHWC));
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
          lite::Tensor x, y_image, y, relu_in, relu_out;
          operators::LayoutParam BufferToImageParam;
          operators::LayoutParam ImageToBufferParam;
          BufferToImageParam.x = &x;
          BufferToImageParam.y = &y_image;
          ImageToBufferParam.x = &y_image;
          ImageToBufferParam.y = &y;
          operators::ActivationParam ReluParam;
          ReluParam.X = &relu_in;
          ReluParam.Out = &relu_out;

          const DDim x_dim = DDim(std::vector<DDim::value_type>{n, c, h, w});
          x.Resize(x_dim);
          y_image.Resize(x_dim);  // useless for image2D
          y.Resize(x_dim);
          std::map<std::string, size_t> &relu_image2d_shape =
              InitImageDimInfoWith(x_dims);

          // initialize tensors
          LOG(INFO) << "initialize tensors";
          auto *x_data = x.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
          auto *y_data = y.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
          auto image_shape =
              paddle::lite::kernels::opencl::InitImageDimInfoWith(x_dim);
          auto *y_image_data = y_image.mutable_data<float, cl::Image2D>(
              image_shape["width"], image_shape["height"]);
          auto *mapped_x = static_cast<float *>(TargetWrapperCL::Map(
              x_data, 0, sizeof(float) * x_dim.production()));
          auto *mapped_y = static_cast<float *>(TargetWrapperCL::Map(
              y_data, 0, sizeof(float) * x_dim.production()));
          for (int i = 0; i < x_dim.production(); ++i) {
            mapped_x[i] = static_cast<int>(i);
            mapped_y[i] = static_cast<int>(0);
          }
          auto *relu_in_data = relu_in.mutable_data<float, cl::Image2D>(
              TARGET(kOpenCL),
              relu_image2d_shape["width"],
              relu_image2d_shape["height"]);
          auto *relu_out_data = relu_out.mutable_data<float, cl::Image2D>(
              TARGET(kOpenCL),
              relu_image2d_shape["width"],
              relu_image2d_shape["height"]);

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

// result
#ifdef PRINT_RESULT
          LOG(INFO) << "---- print result ----";
          for (int eidx = 0; i < x_dim.production(); ++eidx) {
            std::cout << mapped_x[eidx] << " -> " << mapped_y[eidx]
                      << std::endl;
          }
#endif  // PRINT_RESULT

          // check result: compare input and output
          for (int eidx = 0; eidx < x_dim.production(); eidx++) {
            EXPECT_NEAR(max(0.0f, mapped_x[eidx]), mapped_y[eidx], 1e-6);
            if (abs(max(0.0f, mapped_x[eidx]) - mapped_y[eidx]) > 1e-6) {
              LOG(INFO) << "1st diff in this case at eidx[from 0]:" << eidx
                        << " / " << x_dim.production() << ", mapped_x[" << eidx
                        << "]:" << max(0.0f, mapped_x[eidx]) << ", mapped_y["
                        << eidx << "]:" << mapped_y[eidx];
              break;
            }
          }

          // free
          LOG(INFO) << "free: unmap x, y";
          TargetWrapperCL::Unmap(x_data, mapped_x);
          TargetWrapperCL::Unmap(y_data, mapped_y);
#ifdef LOOP_TEST
        }  // w
      }    // h
    }      // c
  }        // n
#else
// nothing to do.
#endif
}

TEST(opencl_relu_image2d, compute) {
  // prepare data
  const DDim x_dim = DDim(std::vector<DDim::value_type>{3, 6, 10, 10});
  lite::Tensor x, out;
  x.Resize(x_dim);
  out.Resize(x_dim);

  auto *x_data = x.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
  std::default_random_engine engine;
  std::uniform_real_distribution<float> dist(-10, 10);
  auto *mapped_x = static_cast<float *>(
      TargetWrapperCL::Map(x_data, 0, sizeof(float) * x_dim.production()));
  for (int i = 0; i < x_dim.production(); i++) {
    mapped_x[i] = dist(engine);
  }

  // set param and kernel, then run
  operators::ActivationParam param;
  param.X = &x;
  param.Out = &out;

  std::unique_ptr<KernelContext> context(new KernelContext);
  context->As<OpenCLContext>().InitOnce();
  auto kernels = KernelRegistry::Global().Create(
      "relu", TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNCHW));
  ASSERT_FALSE(kernels.empty());
  auto kernel = std::move(kernels.front());
  kernel->SetParam(param);
  std::unique_ptr<KernelContext> relu_context(new KernelContext);
  context->As<OpenCLContext>().CopySharedTo(
      &(relu_context->As<OpenCLContext>()));
  kernel->SetContext(std::move(relu_context));

  kernel->Launch();

  auto *wait_list = context->As<OpenCLContext>().cl_wait_list();
  auto *out_ptr = param.Out->data<float, cl::Buffer>();
  auto it = wait_list->find(out_ptr);
  if (it != wait_list->end()) {
    VLOG(4) << "--- Find the sync event for the target cl tensor. ---";
    auto &event = *(it->second);
    event.wait();
  } else {
    LOG(FATAL) << "Could not find the sync event for the target cl tensor.";
  }

  // run compute ref and check
  std::unique_ptr<float[]> out_ref(new float[x_dim.production()]);
  relu_compute_ref<float>(mapped_x, x_dim, out_ref.get());

  auto *out_data = out.mutable_data<float, cl::Buffer>();
  auto *mapped_out = static_cast<float *>(
      TargetWrapperCL::Map(out_data, 0, sizeof(float) * x_dim.production()));
  for (int i = 0; i < x_dim.production(); i++) {
    EXPECT_NEAR(mapped_out[i], out_ref[i], 1e-6);
  }
  TargetWrapperCL::Unmap(out_data, mapped_out);
  TargetWrapperCL::Unmap(x_data, mapped_x);
}

}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(relu, kOpenCL, kFloat, kNCHW, def);
USE_LITE_KERNEL(relu, kOpenCL, kFloat, kNHWC, def);
