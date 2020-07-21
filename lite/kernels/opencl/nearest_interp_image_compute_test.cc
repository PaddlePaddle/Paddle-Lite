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

template <typename dtype>
void nearest_interp_compute_ref(const dtype *src,
                                int w_in,
                                int h_in,
                                dtype *dst,
                                int w_out,
                                int h_out,
                                float scale_x,
                                float scale_y,
                                bool with_align = false) {
  float scale_w_new = (with_align)
                          ? (static_cast<float>(w_in - 1) / (w_out - 1))
                          : (static_cast<float>(w_in) / (w_out));
  float scale_h_new = (with_align)
                          ? (static_cast<float>(h_in - 1) / (h_out - 1))
                          : (static_cast<float>(h_in) / (h_out));
  if (with_align) {
    for (int h = 0; h < h_out; ++h) {
      dtype *dst_p = dst + h * w_out;
      int near_y = static_cast<int>(scale_h_new * h + 0.5);
      for (int w = 0; w < w_out; ++w) {
        int near_x = static_cast<int>(scale_w_new * w + 0.5);
        *dst_p++ = src[near_y * w_in + near_x];
      }
    }
  } else {
    for (int h = 0; h < h_out; ++h) {
      dtype *dst_p = dst + h * w_out;
      int near_y = static_cast<int>(scale_h_new * h);
      for (int w = 0; w < w_out; ++w) {
        int near_x = static_cast<int>(scale_w_new * w);
        *dst_p++ = src[near_y * w_in + near_x];
      }
    }
  }
}
// #define LOOP_TEST
// #define PRINT_RESULT
TEST(nearest_interp_image2d, compute) {
  LOG(INFO) << "main steps of test: host -> layout(buf2img) -> "
               "nearest_interp(img) -> "
               "layout(img2buf) "
               "-> host";

#ifdef LOOP_TEST
  for (int n : {1, 3}) {
    for (auto c : {1, 3}) {
      for (int h : {12, 20, 50, 112}) {
        for (int w : {12, 20, 50, 112}) {
          for (int out_h : {36, 60, 90, 224}) {
            for (int out_w : {36, 60, 90, 224}) {
              if (out_w < w || out_h < h) {
                continue;
              }
#else
  const int n = 1;
  const int c = 2;
  const int h = 3;
  const int w = 4;
  const int out_h = 6;
  const int out_w = 8;
#endif  // LOOP_TEST

              float scale_x = out_w / w;
              float scale_y = out_h / h;

              LOG(INFO) << "======== input shape[n,c,h,w]:" << n << " " << c
                        << " " << h << " " << w << " ========" << out_h << " "
                        << out_w;
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
              auto nearest_interp_img_kernels =
                  KernelRegistry::Global().Create("nearest_interp",
                                                  TARGET(kOpenCL),
                                                  PRECISION(kFP16),
                                                  DATALAYOUT(kImageDefault));
              ASSERT_FALSE(buf_to_img_kernels.empty());
              ASSERT_FALSE(buf_to_img_kernels.empty());
              ASSERT_FALSE(nearest_interp_img_kernels.empty());

              auto buf_to_img_kernel = std::move(buf_to_img_kernels.front());
              auto img_to_buf_kernel = std::move(img_to_buf_kernels.front());
              auto nearest_interp_img_kernel =
                  std::move(nearest_interp_img_kernels.front());
              LOG(INFO) << "get 1st kernel: " << buf_to_img_kernel->doc();
              LOG(INFO) << "get 2nd kernel: " << img_to_buf_kernel->doc();
              LOG(INFO) << "get 3rd kernel: "
                        << nearest_interp_img_kernel->doc();

              // set tensors about op param
              LOG(INFO) << "set tensors about op param";
              // layout(buf->img): x -> nearest_interp_in
              // nearest_interp(img): nearest_interp_in -> nearest_interp_out
              // layout(img->buf): nearest_interp_out -> y
              lite::Tensor x, y, nearest_interp_in, nearest_interp_out, y_ref;
              operators::LayoutParam BufferToImageParam;
              operators::LayoutParam ImageToBufferParam;
              BufferToImageParam.x = &x;
              BufferToImageParam.y = &nearest_interp_in;
              ImageToBufferParam.x = &nearest_interp_out;
              ImageToBufferParam.y = &y;
              operators::InterpolateParam NearestInterpParam;
              NearestInterpParam.X = &nearest_interp_in;
              NearestInterpParam.Out = &nearest_interp_out;
              NearestInterpParam.out_h = out_h;
              NearestInterpParam.out_w = out_w;

              const DDim x_dim =
                  DDim(std::vector<DDim::value_type>{n, c, h, w});
              const DDim y_dim =
                  DDim(std::vector<DDim::value_type>{n, c, out_h, out_w});
              x.Resize(x_dim);
              y.Resize(y_dim);
              nearest_interp_in.Resize(x_dim);
              nearest_interp_out.Resize(y_dim);
              y_ref.Resize(y_dim);
              auto nearest_interp_image2d_shape =
                  paddle::lite::kernels::opencl::InitImageDimInfoWith(x_dim);

              // initialize tensors
              LOG(INFO) << "initialize tensors";
              auto *x_data = x.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
              auto *y_data = y.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
              auto *y_data_ref = y_ref.mutable_data<float>(TARGET(kARM));
              memset(reinterpret_cast<char *>(y_data_ref), 0, y_ref.numel());
              auto *mapped_x = static_cast<float *>(TargetWrapperCL::Map(
                  x_data, 0, sizeof(float) * x_dim.production()));
              auto *mapped_y = static_cast<float *>(TargetWrapperCL::Map(
                  y_data, 0, sizeof(float) * y_dim.production()));
              for (int i = 0; i < x_dim.production(); ++i) {
                mapped_x[i] = static_cast<int>(i) - x_dim.production() / 2;
              }
              for (int i = 0; i < y_dim.production(); ++i) {
                mapped_y[i] = static_cast<int>(0);
              }
              auto *nearest_interp_in_data =
                  nearest_interp_in.mutable_data<half_t, cl::Image2D>(
                      nearest_interp_image2d_shape["width"],
                      nearest_interp_image2d_shape["height"]);
              auto *nearest_interp_out_data =
                  nearest_interp_out.mutable_data<half_t, cl::Image2D>(
                      y_dim[3], y_dim[2]);

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

              nearest_interp_img_kernel->SetParam(NearestInterpParam);
              std::unique_ptr<KernelContext> nearest_interp_img_context(
                  new KernelContext);
              context->As<OpenCLContext>().CopySharedTo(
                  &(nearest_interp_img_context->As<OpenCLContext>()));
              nearest_interp_img_kernel->SetContext(
                  std::move(nearest_interp_img_context));

              // run kernels
              LOG(INFO) << "run kernel: buf_to_img_kernel";
              buf_to_img_kernel->Launch();
              LOG(INFO) << "run kernel: nearest_interp_img_kernel";
              nearest_interp_img_kernel->Launch();
              LOG(INFO) << "run kernel: img_to_buf_kernel";
              img_to_buf_kernel->Launch();

              CLRuntime::Global()->command_queue().finish();

              // compute ref cpu
              for (int nid = 0; nid < x_dim[0]; ++nid) {
                for (int cid = 0; cid < x_dim[1]; ++cid) {
                  float *x_nc =
                      mapped_x + (nid * x_dim[1] + cid) * x_dim[3] * x_dim[2];
                  float *y_nc =
                      y_data_ref + (nid * x_dim[1] + cid) * y_dim[3] * y_dim[2];
                  nearest_interp_compute_ref<float>(x_nc,
                                                    x_dim[3],
                                                    x_dim[2],
                                                    y_nc,
                                                    y_dim[3],
                                                    y_dim[2],
                                                    1 / scale_x,
                                                    1 / scale_y);
                }
              }
// result
#ifdef PRINT_RESULT
              LOG(INFO) << "---- print kernel result (input -> output) ----";
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

              // check result: compare kernel output and cpu output(y_data_ref)
              for (int eidx = 0; eidx < y_dim.production(); eidx++) {
                EXPECT_NEAR(y_data_ref[eidx], mapped_y[eidx], 1e-6);
                if (abs(y_data_ref[eidx] - mapped_y[eidx]) > 1e-6) {
                  LOG(FATAL) << "1st diff in this case at eidx[from 0]:" << eidx
                             << " / " << x_dim.production() << ", y_data_ref["
                             << eidx << "]:" << y_data_ref[eidx]
                             << ", mapped_y[" << eidx << "]:" << mapped_y[eidx];
                  break;
                }
              }

              // free
              LOG(INFO) << "free: unmap x, y";
              TargetWrapperCL::Unmap(x_data, mapped_x);
              TargetWrapperCL::Unmap(y_data, mapped_y);
#ifdef LOOP_TEST
            }
          }
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

// nearest_interp image2d fp32
USE_LITE_KERNEL(layout, kOpenCL, kAny, kImageDefault, NCHW_to_ImageDefault);
USE_LITE_KERNEL(layout, kOpenCL, kAny, kNCHW, ImageDefault_to_NCHW);

// nearest_interp image2d fp16
USE_LITE_KERNEL(nearest_interp, kOpenCL, kFP16, kImageDefault, ImageDefault);
