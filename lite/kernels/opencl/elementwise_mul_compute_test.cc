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
#include <algorithm>
#include <random>
#include "lite/backends/opencl/target_wrapper.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {

template <typename dtype>
void elementwise_compute_ref(const dtype *x_data,
                             const dtype *y_data,
                             dtype *out_data,
                             const DDim &x_dims,
                             const DDim &y_dims,
                             int axis,
                             const std::string elt_type,
                             bool use_relu = false) {
  if (axis < 0) {
    axis = x_dims.size() - y_dims.size();
  }
  int batch = 1;
  int channels = 1;
  int num = 1;
  for (int i = 0; i < axis; ++i) {
    batch *= x_dims[i];
  }
  for (int i = 0; i < y_dims.size(); ++i) {
    channels *= y_dims[i];
  }
  for (int i = y_dims.size() + axis; i < x_dims.size(); ++i) {
    num *= x_dims[i];
  }
  // do elementwise add/sub/max/...
  if (elt_type == "add") {
    for (int i = 0; i < batch; ++i) {
      for (int j = 0; j < channels; ++j) {
        int offset = (i * channels + j) * num;
        const dtype *din_ptr = x_data + offset;
        const dtype diny_data = y_data[j];
        dtype *dout_ptr = out_data + offset;
        for (int k = 0; k < num; ++k) {
          *dout_ptr = *din_ptr + diny_data;
          if (use_relu) {
            *dout_ptr = std::max(*dout_ptr, static_cast<dtype>(0));
          }
          dout_ptr++;
          din_ptr++;
        }
      }
    }
  } else if (elt_type == "sub") {
    for (int i = 0; i < batch; ++i) {
      for (int j = 0; j < channels; ++j) {
        int offset = (i * channels + j) * num;
        const dtype *din_ptr = x_data + offset;
        const dtype diny_data = y_data[j];
        dtype *dout_ptr = out_data + offset;
        for (int k = 0; k < num; ++k) {
          *dout_ptr = *din_ptr - diny_data;
          if (use_relu) {
            *dout_ptr = std::max(*dout_ptr, static_cast<dtype>(0));
          }
          dout_ptr++;
          din_ptr++;
        }
      }
    }
  } else if (elt_type == "mul") {
    for (int i = 0; i < batch; ++i) {
      for (int j = 0; j < channels; ++j) {
        int offset = (i * channels + j) * num;
        const dtype *din_ptr = x_data + offset;
        const dtype diny_data = y_data[j];
        dtype *dout_ptr = out_data + offset;
        for (int k = 0; k < num; ++k) {
          *dout_ptr = *din_ptr * diny_data;
          if (use_relu) {
            *dout_ptr = std::max(*dout_ptr, static_cast<dtype>(0));
          }
          dout_ptr++;
          din_ptr++;
        }
      }
    }
  } else {
    LOG(FATAL) << "unsupported Elementwise type: " << elt_type << std::endl;
  }
}

#if 0
TEST(elementwise_mul_image_fp32, compute) {
  LOG(INFO) << "to get kernel ...";
  auto kernels = KernelRegistry::Global().Create(
      "elementwise_mul", TARGET(kOpenCL), PRECISION(kFloat),
      DATALAYOUT(kImageDefault));
  ASSERT_FALSE(kernels.empty());

  auto kernel = std::move(kernels.front());

  LOG(INFO) << "get kernel" << kernel->doc();

  lite::Tensor x, y, out;
  operators::ElementwiseParam param;
  param.X = &x;
  param.Y = &y;
  param.Out = &out;
  param.axis = -1;

  std::unique_ptr<KernelContext> context(new KernelContext);
  context->As<OpenCLContext>().InitOnce();

  kernel->SetParam(param);
  std::unique_ptr<KernelContext> ele_context(new KernelContext);
  context->As<OpenCLContext>().CopySharedTo(
      &(ele_context->As<OpenCLContext>()));
  kernel->SetContext(std::move(ele_context));

  const DDim x_dim = DDim(std::vector<DDim::value_type>{3, 2, 1, 5});
  const DDim y_dim = DDim(std::vector<DDim::value_type>{2, 1, 5});
  const DDim out_dim = DDim(std::vector<DDim::value_type>{3, 2, 1, 5});
  x.Resize(x_dim);
  y.Resize(y_dim);
  out.Resize(out_dim);

  auto *x_data = x.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
  auto *y_data = y.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));

  std::default_random_engine engine;
  std::uniform_real_distribution<float> dist(-10, 10);
  auto *mapped_x = static_cast<float *>(
      TargetWrapperCL::Map(x_data, 0, sizeof(float) * x_dim.production()));
  for (int i = 0; i < x_dim.production(); i++) {
    mapped_x[i] = dist(engine);
  }
  auto *mapped_y = static_cast<float *>(
      TargetWrapperCL::Map(y_data, 0, sizeof(float) * y_dim.production()));
  for (int i = 0; i < y_dim.production(); i++) {
    mapped_y[i] = dist(engine);
  }

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

  std::unique_ptr<float[]> out_ref(new float[out_dim.production()]);
  elementwise_compute_ref<float>(
      mapped_x, mapped_y, out_ref.get(), x_dim, y_dim, param.axis, "add");

  TargetWrapperCL::Unmap(x_data, mapped_x);
  TargetWrapperCL::Unmap(y_data, mapped_y);
  auto *out_data = out.mutable_data<float, cl::Buffer>();
  auto *mapped_out = static_cast<float *>(
      TargetWrapperCL::Map(out_data, 0, sizeof(float) * out_dim.production()));
  for (int i = 0; i < out_dim.production(); i++) {
    EXPECT_NEAR(mapped_out[i], out_ref[i], 1e-6);
  }
  TargetWrapperCL::Unmap(out_data, mapped_out);
}

// #define LOOP_TEST
// #define PRINT_RESULT
TEST(elemul_image2d_fp32, compute) {
  LOG(INFO) << "main steps of test: host -> layout(buf2img) -> elemul(img) -> "
               "layout(img2buf) "
               "-> host";

#ifdef LOOP_TEST
  for (int n = 1; n <= 100; n += 33) {
    for (auto c : {1, 3}) {
      for (int h = 12; h <= 100; h += 13) {
        for (int w = 12; w <= 100; w += 25) {
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
          auto elemul_img_kernels =
              KernelRegistry::Global().Create("elemul",
                                              TARGET(kOpenCL),
                                              PRECISION(kFloat),
                                              DATALAYOUT(kImageDefault));
          ASSERT_FALSE(buf_to_img_kernels.empty());
          ASSERT_FALSE(buf_to_img_kernels.empty());
          ASSERT_FALSE(elemul_img_kernels.empty());

          auto buf_to_img_kernel = std::move(buf_to_img_kernels.front());
          auto img_to_buf_kernel = std::move(img_to_buf_kernels.front());
          auto elemul_img_kernel = std::move(elemul_img_kernels.front());
          LOG(INFO) << "get 1st kernel: " << buf_to_img_kernel->doc();
          LOG(INFO) << "get 2nd kernel: " << img_to_buf_kernel->doc();
          LOG(INFO) << "get 3rd kernel: " << elemul_img_kernel->doc();

          // set tensors about op param
          LOG(INFO) << "set tensors about op param";
          // layout(buf->img): x -> elemul_in
          // elemul(img): elemul_in -> elemul_out
          // layout(img->buf): elemul_out -> y
          lite::Tensor x, y, elemul_in, elemul_out, y_ref;
          operators::LayoutParam BufferToImageParam;
          operators::LayoutParam ImageToBufferParam;
          BufferToImageParam.x = &x;
          BufferToImageParam.y = &elemul_in;
          ImageToBufferParam.x = &elemul_out;
          ImageToBufferParam.y = &y;
          operators::ActivationParam elemulParam;
          elemulParam.X = &elemul_in;
          elemulParam.Out = &elemul_out;

          const DDim x_dim = DDim(std::vector<DDim::value_type>{n, c, h, w});
          x.Resize(x_dim);
          y.Resize(x_dim);
          elemul_in.Resize(x_dim);
          elemul_out.Resize(x_dim);
          y_ref.Resize(x_dim);
          auto elemul_image2d_shape =
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
          auto *elemul_in_data = elemul_in.mutable_data<float, cl::Image2D>(
              elemul_image2d_shape["width"], elemul_image2d_shape["height"]);
          auto *elemul_out_data = elemul_out.mutable_data<float, cl::Image2D>(
              elemul_image2d_shape["width"], elemul_image2d_shape["height"]);

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

          elemul_img_kernel->SetParam(elemulParam);
          std::unique_ptr<KernelContext> elemul_img_context(new KernelContext);
          context->As<OpenCLContext>().CopySharedTo(
              &(elemul_img_context->As<OpenCLContext>()));
          elemul_img_kernel->SetContext(std::move(elemul_img_context));

          // run kernels
          LOG(INFO) << "run kernel: buf_to_img_kernel";
          buf_to_img_kernel->Launch();
          LOG(INFO) << "run kernel: elemul_img_kernel";
          elemul_img_kernel->Launch();
          LOG(INFO) << "run kernel: img_to_buf_kernel";
          img_to_buf_kernel->Launch();

          // compute ref cpu
          elementwise_compute_ref<float>(
              mapped_x, mapped_y, out_ref.get(), x_dim, y_dim, param.axis,
               "mul");
//`          elementwise_compute_ref<float>(mapped_x, x_dim, y_data_ref);
// result
#ifdef PRINT_RESULT
          LOG(INFO) << "---- print kernel result (input -> output) ----";
          for (int eidx = 0; eidx < x_dim.production(); ++eidx) {
            std::cout << mapped_x[eidx] << " -> " << mapped_y[eidx]
                      << std::endl;
          }
#endif  // PRINT_RESULT

          // check result: compare kernel output and cpu output(y_data_ref)
          for (int eidx = 0; eidx < x_dim.production(); eidx++) {
            EXPECT_NEAR(y_data_ref[eidx], mapped_y[eidx], 1e-6);
            if (abs(y_data_ref[eidx] - mapped_y[eidx]) > 1e-6) {
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

USE_LITE_KERNEL(elementwise_mul, kOpenCL, kFloat, kImageDefault, def);
