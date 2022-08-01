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
#include "lite/core/parallel_defines.h"
#include "lite/core/tensor.h"
#include "lite/kernels/opencl/image_helper.h"

namespace paddle {
namespace lite {

void pad2d_ref(const float *x_data,
               Tensor *y,
               std::string mode,
               int pad_h0,
               int pad_h1,
               int pad_w0,
               int pad_w1,
               float pad_value) {
  auto *out_data = y->mutable_data<float>();
  auto output_dims = y->dims();
  int n = output_dims[0];
  int c = output_dims[1];
  int h = output_dims[2];
  int w = output_dims[3];
  int pad_mode;
  if (mode == "constant") {
    pad_mode = 0;
  } else if (mode == "reflect") {
    pad_mode = 2;
  } else if (mode == "edge") {
    pad_mode = 1;
  } else {
    LOG(FATAL) << "Unknown mode type";
  }
  int in_w = w - pad_w0 - pad_w1;
  int in_h = h - pad_h0 - pad_h1;
  int spatial_size_out = w * h;
  int spatial_size_in = in_w * in_h;
  LITE_PARALLEL_BEGIN(i, tid, n * c) {
    const float *din_batch = x_data + i * spatial_size_in;
    float *dout_batch = out_data + i * spatial_size_out;
    int in_y = 0;
    int in_x = 0;
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        switch (pad_mode) {
          case 0:
            in_y = y - pad_h0;
            in_x = x - pad_w0;
            dout_batch[y * w + x] =
                (in_x >= 0 && in_x < in_w) && (in_y >= 0 && in_y < in_h)
                    ? din_batch[in_y * in_w + in_x]
                    : pad_value;
            break;
          case 1:
            in_x = std::min(std::max(pad_w0, x), in_w + pad_w0 - 1) - pad_w0;
            in_y = std::min(std::max(pad_h0, y), in_h + pad_h0 - 1) - pad_h0;
            dout_batch[y * w + x] = din_batch[in_y * in_w + in_x];
            break;
          case 2:
            in_y = y - pad_h0;
            in_x = x - pad_w0;
            in_y = std::max(in_y, -in_y);
            in_y = std::min(in_y, 2 * in_h - in_y - 2);
            in_x = std::max(in_x, -in_x);
            in_x = std::min(in_x, 2 * in_w - in_x - 2);
            dout_batch[y * w + x] = din_batch[in_y * in_w + in_x];
            break;
          default:
            LOG(ERROR) << "ERROR: unknown pad mode:" << pad_mode;
        }
      }
    }
  }
  LITE_PARALLEL_END();
}

// #define LOOP_TEST
// #define PRINT_RESULT
TEST(pad2d_image2d, compute) {
  LOG(INFO) << "main steps of test: host -> layout(buf2img) -> "
               "pad2d(img) -> "
               "layout(img2buf) "
               "-> host";

#ifdef LOOP_TEST
  for (int n : {1, 3}) {
    for (auto c : {1, 3}) {
      for (int h : {12, 112}) {
        for (int w : {12, 112}) {
          for (int pad_h0 : {0, 1, 2}) {
            for (int pad_h1 : {0, 1, 2}) {
              for (int pad_w0 : {0, 1, 2}) {
                for (int pad_w1 : {0, 1, 2}) {
                  for (float pad_value : {10.f}) {
                    for (std::string pad_mode :
                         {"constant", "reflect", "edge"}) {
#else
  const int n = 1;
  const int c = 3;
  const int h = 12;
  const int w = 112;
  const int pad_h0 = 1;
  const int pad_h1 = 2;
  const int pad_w0 = 1;
  const int pad_w1 = 2;
  const float pad_value = 10.f;
  std::string pad_mode = "reflect";
#endif  // LOOP_TEST

                      LOG(INFO) << "======== input shape[n,c,h,w]:" << n << " "
                                << c << " " << h << " " << w;
                      LOG(INFO) << "======== pad_h0: " << pad_h0
                                << ", pad_h1: " << pad_h1
                                << ", pad_w0: " << pad_w0
                                << ", pad_w1: " << pad_w1
                                << ",  pad_value: " << pad_value
                                << ", pad_mode: " << pad_mode;
                      // set layout kernels
                      auto buf_to_img_kernels = KernelRegistry::Global().Create(
                          "layout",
                          TARGET(kOpenCL),
                          PRECISION(kAny),
                          DATALAYOUT(kImageDefault));
                      auto img_to_buf_kernels =
                          KernelRegistry::Global().Create("layout",
                                                          TARGET(kOpenCL),
                                                          PRECISION(kAny),
                                                          DATALAYOUT(kNCHW));
                      auto pad2d_img_kernels = KernelRegistry::Global().Create(
                          "pad2d",
                          TARGET(kOpenCL),
                          PRECISION(kFP16),
                          DATALAYOUT(kImageDefault));
                      ASSERT_FALSE(buf_to_img_kernels.empty());
                      ASSERT_FALSE(buf_to_img_kernels.empty());
                      ASSERT_FALSE(pad2d_img_kernels.empty());

                      auto buf_to_img_kernel =
                          std::move(buf_to_img_kernels.front());
                      auto img_to_buf_kernel =
                          std::move(img_to_buf_kernels.front());
                      auto pad2d_img_kernel =
                          std::move(pad2d_img_kernels.front());
                      LOG(INFO) << "get 1st kernel: "
                                << buf_to_img_kernel->doc();
                      LOG(INFO) << "get 2nd kernel: "
                                << img_to_buf_kernel->doc();
                      LOG(INFO) << "get 3rd kernel: "
                                << pad2d_img_kernel->doc();

                      // set tensors about op param
                      LOG(INFO) << "set tensors about op param";
                      // layout(buf->img): x -> pad2d_in
                      // pad2d(img): pad2d_in -> pad2d_out
                      // layout(img->buf): pad2d_out -> y
                      lite::Tensor x, y, pad2d_in, pad2d_out, y_ref;
                      operators::LayoutParam BufferToImageParam;
                      operators::LayoutParam ImageToBufferParam;
                      BufferToImageParam.x = &x;
                      BufferToImageParam.y = &pad2d_in;
                      ImageToBufferParam.x = &pad2d_out;
                      ImageToBufferParam.y = &y;
                      operators::Pad2dParam Pad2dParam;
                      Pad2dParam.X = &pad2d_in;
                      Pad2dParam.Out = &pad2d_out;
                      Pad2dParam.paddings = {pad_h0, pad_h1, pad_w0, pad_w1};
                      Pad2dParam.pad_value = pad_value;
                      Pad2dParam.mode = pad_mode;

                      int64_t out_h = h + pad_h0 + pad_h1;
                      int64_t out_w = w + pad_w0 + pad_w1;
                      const DDim x_dim =
                          DDim(std::vector<DDim::value_type>{n, c, h, w});
                      const DDim y_dim = DDim(
                          std::vector<DDim::value_type>{n, c, out_h, out_w});
                      x.Resize(x_dim);
                      y.Resize(y_dim);
                      pad2d_in.Resize(x_dim);
                      pad2d_out.Resize(y_dim);
                      y_ref.Resize(y_dim);
                      auto pad2d_image2d_shape =
                          paddle::lite::kernels::opencl::InitImageDimInfoWith(
                              x_dim);

                      // initialize tensors
                      LOG(INFO) << "initialize tensors";
                      auto *x_data =
                          x.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
                      auto *y_data =
                          y.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
                      auto *y_data_ref =
                          y_ref.mutable_data<float>(TARGET(kARM));
                      auto *mapped_x =
                          static_cast<float *>(TargetWrapperCL::Map(
                              x_data, 0, sizeof(float) * x_dim.production()));
                      auto *mapped_y =
                          static_cast<float *>(TargetWrapperCL::Map(
                              y_data, 0, sizeof(float) * y_dim.production()));
                      std::default_random_engine engine;
                      std::uniform_real_distribution<float> dist(-1, 1);
                      for (int i = 0; i < x_dim.production(); ++i) {
                        mapped_x[i] = dist(engine);
                      }
                      auto *pad2d_in_data =
                          pad2d_in.mutable_data<half_t, cl::Image2D>(
                              pad2d_image2d_shape["width"],
                              pad2d_image2d_shape["height"]);
                      auto *pad2d_out_data =
                          pad2d_out.mutable_data<half_t, cl::Image2D>(y_dim[3],
                                                                      y_dim[2]);

                      // set context and kernel args
                      LOG(INFO) << "set context and kernel args";
                      std::unique_ptr<KernelContext> context(new KernelContext);
                      context->As<OpenCLContext>().InitOnce();

                      buf_to_img_kernel->SetParam(BufferToImageParam);
                      std::unique_ptr<KernelContext> buf_to_img_context(
                          new KernelContext);
                      context->As<OpenCLContext>().CopySharedTo(
                          &(buf_to_img_context->As<OpenCLContext>()));
                      buf_to_img_kernel->SetContext(
                          std::move(buf_to_img_context));

                      img_to_buf_kernel->SetParam(ImageToBufferParam);
                      std::unique_ptr<KernelContext> img_to_buf_context(
                          new KernelContext);
                      context->As<OpenCLContext>().CopySharedTo(
                          &(img_to_buf_context->As<OpenCLContext>()));
                      img_to_buf_kernel->SetContext(
                          std::move(img_to_buf_context));

                      pad2d_img_kernel->SetParam(Pad2dParam);
                      std::unique_ptr<KernelContext> pad2d_img_context(
                          new KernelContext);
                      context->As<OpenCLContext>().CopySharedTo(
                          &(pad2d_img_context->As<OpenCLContext>()));
                      pad2d_img_kernel->SetContext(
                          std::move(pad2d_img_context));

                      // run kernels
                      LOG(INFO) << "run kernel: buf_to_img_kernel";
                      buf_to_img_kernel->Launch();
                      LOG(INFO) << "run kernel: pad2d_img_kernel";
                      pad2d_img_kernel->Launch();
                      LOG(INFO) << "run kernel: img_to_buf_kernel";
                      img_to_buf_kernel->Launch();

                      // wait for opencl

                      CLRuntime::Global()->command_queue().finish();

                      // compute ref cpu
                      pad2d_ref(mapped_x,
                                &y_ref,
                                pad_mode,
                                pad_h0,
                                pad_h1,
                                pad_w0,
                                pad_w1,
                                pad_value);
// result
#ifdef PRINT_RESULT
                      LOG(INFO)
                          << "---- print kernel result (input -> output) ----";
                      for (int eidx = 0; eidx < x_dim.production(); ++eidx) {
                        std::cout << mapped_x[eidx] << " ";
                      }
                      std::cout << std::endl;
                      for (int eidx = 0; eidx < y_dim.production(); ++eidx) {
                        std::cout << mapped_y[eidx] << " ";
                      }
                      std::cout << std::endl;
                      for (int eidx = 0; eidx < y_dim.production(); ++eidx) {
                        std::cout << y_data_ref[eidx] << " ";
                      }
                      std::cout << std::endl;
#endif  // PRINT_RESULT
                      // check result: compare kernel output and cpu
                      // output(y_data_ref)
                      for (int eidx = 0; eidx < y_dim.production(); eidx++) {
                        EXPECT_NEAR(y_data_ref[eidx], mapped_y[eidx], 1e-3);
                        if (abs(y_data_ref[eidx] - mapped_y[eidx]) > 1e-3) {
                          LOG(FATAL) << "1st diff in this case at eidx[from 0]:"
                                     << eidx << " / " << y_dim.production()
                                     << ", y_data_ref[" << eidx
                                     << "]:" << y_data_ref[eidx]
                                     << ", mapped_y[" << eidx
                                     << "]:" << mapped_y[eidx];
                          break;
                        }
                      }

                      // free
                      LOG(INFO) << "free: unmap x, y";
                      TargetWrapperCL::Unmap(x_data, mapped_x);
                      TargetWrapperCL::Unmap(y_data, mapped_y);
#ifdef LOOP_TEST
                    }  // pad_mode
                  }    // pad_value
                }      // pad_w1
              }        // pad_w0
            }          // pad_h1
          }            // pad_h0
        }              // w
      }                // h
    }                  // c
  }                    // n
#else
// nothing to do.
#endif
}

}  // namespace lite
}  // namespace paddle

// pad2d image2d fp32
USE_LITE_KERNEL(layout, kOpenCL, kAny, kImageDefault, NCHW_to_ImageDefault);
USE_LITE_KERNEL(layout, kOpenCL, kAny, kNCHW, ImageDefault_to_NCHW);

// pad image2d fp16
USE_LITE_KERNEL(pad2d, kOpenCL, kFP16, kImageDefault, ImageDefault);
