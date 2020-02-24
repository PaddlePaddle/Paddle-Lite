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

namespace paddle {
namespace lite {

// #define LOOP_TEST
// #define PRINT_RESULT
TEST(layout_ImageDefault, compute) {
  LOG(INFO) << "main steps of test: host -> layout(buf2img) -> layout(img2buf) "
               "-> device";

#ifdef LOOP_TEST
  for (int n = 1; n <= 2; n += 1) {
    for (auto c : {1, 3}) {
      for (int h = 1; h <= 10; h += 1) {
        for (int w = 1; w <= 10; w += 1) {
#else
  const int n = 1;
  const int c = 2;
  const int h = 3;
  const int w = 4;
#endif  // LOOP_TEST

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
          ASSERT_FALSE(buf_to_img_kernels.empty());
          ASSERT_FALSE(buf_to_img_kernels.empty());

          auto buf_to_img_kernel = std::move(buf_to_img_kernels.front());
          auto img_to_buf_kernel = std::move(img_to_buf_kernels.front());
          LOG(INFO) << "get 1st kernel: " << buf_to_img_kernel->doc();
          LOG(INFO) << "get 2nd kernel: " << img_to_buf_kernel->doc();

          // set tensors about op param
          LOG(INFO) << "set tensors about op param";
          lite::Tensor x, y_image, y;
          operators::LayoutParam BufferToImageParam;
          operators::LayoutParam ImageToBufferParam;
          BufferToImageParam.x = &x;
          BufferToImageParam.y = &y_image;
          ImageToBufferParam.x = &y_image;
          ImageToBufferParam.y = &y;

          const DDim x_dim = DDim(std::vector<DDim::value_type>{n, c, h, w});
          x.Resize(x_dim);
          y_image.Resize(x_dim);  // useless for image2D
          y.Resize(x_dim);

          // initialize tensors
          LOG(INFO) << "initialize tensors";
          auto* x_data = x.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
          auto* y_data = y.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
          auto image_shape =
              paddle::lite::kernels::opencl::InitImageDimInfoWith(x_dim);
          auto* y_image_data = y_image.mutable_data<uint16_t, cl::Image2D>(
              image_shape["width"], image_shape["height"]);
          auto* mapped_x = static_cast<float*>(TargetWrapperCL::Map(
              x_data, 0, sizeof(float) * x_dim.production()));
          auto* mapped_y = static_cast<float*>(TargetWrapperCL::Map(
              y_data, 0, sizeof(float) * x_dim.production()));
          for (int i = 0; i < x_dim.production(); ++i) {
            mapped_x[i] = static_cast<float>(i) * 2;
          }

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

          // run kernels
          LOG(INFO) << "run kernel: buf_to_img_kernel";
          buf_to_img_kernel->Launch();
          LOG(INFO) << "run kernel: img_to_buf_kernel";
          img_to_buf_kernel->Launch();

// result
#ifdef PRINT_RESULT
          LOG(INFO) << "---- print result ----";
          for (int eidx = 0; eidx < x_dim.production(); ++eidx) {
            std::cout << mapped_x[eidx] << " -> "
                      << static_cast<float>(mapped_y[eidx]) << std::endl;
          }
#endif  // PRINT_RESULT

          // check result: compare input and output
          float MAX_PASS_DIFF = 1e-4;
          for (int eidx = 0; eidx < x_dim.production(); eidx++) {
            EXPECT_NEAR(mapped_x[eidx], mapped_y[eidx], MAX_PASS_DIFF);
            if (abs(mapped_x[eidx] - mapped_y[eidx]) > MAX_PASS_DIFF) {
              LOG(INFO) << "1st diff in this case at eidx[from 0]:" << eidx
                        << " / " << x_dim.production() << ", mapped_x[" << eidx
                        << "]:" << mapped_x[eidx] << ", mapped_y[" << eidx
                        << "]:" << mapped_y[eidx];
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

#if 0
TEST(layout_ImageNW, compute) {
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
          auto buf_to_img_nw_kernels =
              KernelRegistry::Global().Create("layout_once",
                                              TARGET(kOpenCL),
                                              PRECISION(kFloat),
                                              DATALAYOUT(kImageNW));
          ASSERT_FALSE(buf_to_img_nw_kernels.empty());
          auto buf_to_img_nw_kernel = std::move(buf_to_img_nw_kernels.front());
          LOG(INFO) << "get 1st kernel: " << buf_to_img_nw_kernel->doc();

          // set tensors about op param
          operators::LayoutParam bufferToImageParam;
          lite::Tensor x, y, cpu_y;
          bufferToImageParam.x = &x;
          bufferToImageParam.y = &y;

          const DDim x_dim = DDim(std::vector<DDim::value_type>{n, c, h, w});
          x.Resize(x_dim);
          y.Resize(x_dim);  // useless for image2D
          cpu_y.Resize(x_dim);

          // initialize tensors
          LOG(INFO) << "initialize tensors";

          // mute in buffer
          auto* x_data = x.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
          // mute out image nw
          size_t image_width = w * ((n + 3) / 4);
          size_t image_height = c * h;
          auto* y_data =
              y.mutable_data<float, cl::Image2D>(image_width, image_height);
          auto* cpu_y_data =
              cpu_y.mutable_data<float, cl::Image2D>(image_width, image_height);

          auto* mapped_x = static_cast<float*>(TargetWrapperCL::Map(
              x_data, 0, sizeof(float) * x_dim.production()));

          const size_t cl_image2d_row_pitch{0};
          const size_t cl_image2d_slice_pitch{0};

          auto* mapped_y = static_cast<float*>(
              TargetWrapperCL::MapImage(y_data,
                                        image_width,
                                        image_height,
                                        cl_image2d_row_pitch,
                                        cl_image2d_slice_pitch));

          auto* mapped_cpu_y = static_cast<float*>(
              TargetWrapperCL::MapImage(cpu_y_data,
                                        image_width,
                                        image_height,
                                        cl_image2d_row_pitch,
                                        cl_image2d_slice_pitch));

          // random datas
          std::default_random_engine engine;
          std::uniform_real_distribution<float> gen(-5, 5);

          for (int i = 0; i < x_dim.production(); ++i) {
            mapped_x[i] = gen(engine);
          }

          // gen cpu y_data
          CLImageConverterNWBlock nw_converter;
          nw_converter.NCHWToImage(mapped_x, mapped_cpu_y, x_dim);

          // set context and kernel args
          LOG(INFO) << "set context and kernel args";
          std::unique_ptr<KernelContext> context(new KernelContext);
          context->As<OpenCLContext>().InitOnce();

          // set kernel params
          buf_to_img_nw_kernel->SetParam(bufferToImageParam);

          std::unique_ptr<KernelContext> buf_to_img_context(new KernelContext);
          context->As<OpenCLContext>().CopySharedTo(
              &(buf_to_img_context->As<OpenCLContext>()));

          // set context
          buf_to_img_nw_kernel->SetContext(std::move(buf_to_img_context));

          // run kernels
          LOG(INFO) << "run kernel: buf_to_img_kernel";
          buf_to_img_nw_kernel->Launch();

// result
#ifdef PRINT_RESULT
          LOG(INFO) << "---- print result ----";
          for (int eidx = 0; eidx < x_dim.production(); ++eidx) {
            std::cout << mapped_x[eidx] << " -> " << mapped_y[eidx]
                      << std::endl;
          }
#endif  // PRINT_RESULT

          // check result: compare input and output
          for (int eidx = 0; eidx < x_dim.production(); eidx++) {
            EXPECT_NEAR(mapped_cpu_y[eidx], mapped_y[eidx], 1e-3);
            if (abs(mapped_cpu_y[eidx] - mapped_y[eidx]) > 1e-3) {
              LOG(INFO) << "1st diff in this case at eidx[from 0]:" << eidx
                        << " / " << x_dim.production() << ", mapped_x[" << eidx
                        << "]:" << mapped_cpu_y[eidx] << ", mapped_y[" << eidx
                        << "]:" << mapped_y[eidx];
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
#endif

}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(layout, kOpenCL, kAny, kImageDefault, NCHW_to_ImageDefault);
USE_LITE_KERNEL(layout, kOpenCL, kAny, kNCHW, ImageDefault_to_NCHW);
// USE_LITE_KERNEL(layout_once, kOpenCL, kFloat, kImageNW, NCHW_to_ImageNW);
