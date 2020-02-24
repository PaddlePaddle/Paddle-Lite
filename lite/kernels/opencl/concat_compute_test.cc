// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
// //
// // Licensed under the Apache License, Version 2.0 (the "License");
// // you may not use this file except in compliance with the License.
// // You may obtain a copy of the License at
// //
// //     http://www.apache.org/licenses/LICENSE-2.0
// //
// // Unless required by applicable law or agreed to in writing, software
// // distributed under the License is distributed on an "AS IS" BASIS,
// // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// // See the License for the specific language governing permissions and
// // limitations under the License.

#include <gtest/gtest.h>
#include <random>
#include "lite/backends/opencl/target_wrapper.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/kernels/opencl/image_helper.h"

namespace paddle {
namespace lite {

template <typename dtype>
void concat2_compute_ref(const dtype *in0,
                         const dtype *in1,
                         const int axis,
                         const DDim in0_dim,
                         const DDim in1_dim,
                         const DDim out_dim,
                         dtype *out_data) {
  int pre_size = 1;
  int post_size = 1;
  for (int i = 0; i < axis; i++) {
    pre_size *= in0_dim[i];
  }
  for (int i = axis + 1; i < in0_dim.size(); i++) {
    post_size *= in0_dim[i];
  }
  int axis_size = out_dim[axis];
  for (int i = 0; i < pre_size; i++) {
    for (int j = 0; j < axis_size; j++) {
      if (j < in0_dim[axis]) {
        memcpy(out_data, in0, sizeof(dtype) * post_size);
        in0 += post_size;
        out_data += post_size;
      }
    }
  }
}

template <typename dtype>
void concat_mul_compute_ref(std::vector<const dtype *> ins_data,
                            std::vector<const DDim> ins_dim,
                            int axis,
                            const DDim out_dim,
                            dtype *out_data) {
  int pre_size = 1;
  int post_size = 1;
  for (int i = 0; i < axis; i++) {
    pre_size *= ins_dim[0][i];
  }
  for (int i = axis + 1; i < ins_dim[0].size(); i++) {
    post_size *= ins_dim[0][i];
  }
  int axis_size = out_dim[axis];
  for (int i = 0; i < pre_size; i++) {
    for (int j = 0; j < ins_data.size(); j++) {
      int size = post_size * ins_dim[j][axis];
      memcpy(out_data, ins_data[j], sizeof(dtype) * size);
      out_data += size;
    }
  }
}
#if 0   // concat_buffer
TEST(opencl_concat_buffer, compute) {
  // prepare data
  const DDim x0_dim = DDim(std::vector<DDim::value_type>{1, 2, 3, 4});
  const DDim x1_dim = DDim(std::vector<DDim::value_type>{1, 2, 3, 4});
  const DDim x2_dim = DDim(std::vector<DDim::value_type>{1, 2, 3, 4});
  const DDim out_dim = DDim(std::vector<DDim::value_type>{1, 6, 3, 4});
  lite::Tensor x0, x1, x2, out, out_ref;
  x0.Resize(x0_dim);
  x1.Resize(x1_dim);
  x2.Resize(x2_dim);
  out.Resize(out_dim);
  out_ref.Resize(out_dim);

  auto *x0_data = x0.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
  auto *x1_data = x1.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
  auto *x2_data = x2.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
  std::default_random_engine engine;
  std::uniform_real_distribution<float> dist(-10, 10);
  auto *mapped_x0 = static_cast<float *>(
      TargetWrapperCL::Map(x0_data, 0, sizeof(float) * x0_dim.production()));
  auto *mapped_x1 = static_cast<float *>(
      TargetWrapperCL::Map(x1_data, 0, sizeof(float) * x1_dim.production()));
  auto *mapped_x2 = static_cast<float *>(
      TargetWrapperCL::Map(x2_data, 0, sizeof(float) * x2_dim.production()));
  for (int i = 0; i < x0_dim.production(); i++) {
    mapped_x0[i] = dist(engine);
  }
  for (int i = 0; i < x1_dim.production(); i++) {
    mapped_x1[i] = dist(engine);
  }
  for (int i = 0; i < x2_dim.production(); i++) {
    mapped_x2[i] = dist(engine);
  }

  // set param and kernel, then run
  operators::ConcatParam param;
  std::vector<lite::Tensor *> ins;
  ins.push_back(&x0);
  ins.push_back(&x1);
  ins.push_back(&x2);
  auto axis = 1;
  param.x = ins;
  param.output = &out;
  param.axis = axis;

  std::vector<const float *> ins_data;
  std::vector<const DDim> ins_dim;

  ins_data.push_back(mapped_x0);
  ins_data.push_back(mapped_x1);
  ins_data.push_back(mapped_x2);
  ins_dim.push_back(x0_dim);
  ins_dim.push_back(x1_dim);
  ins_dim.push_back(x2_dim);

  std::unique_ptr<KernelContext> context(new KernelContext);
  context->As<OpenCLContext>().InitOnce();
  auto kernels = KernelRegistry::Global().Create(
      "concat", TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNCHW));
  ASSERT_FALSE(kernels.empty());
  auto kernel = std::move(kernels.front());
  kernel->SetParam(param);
  std::unique_ptr<KernelContext> concat_context(new KernelContext);
  context->As<OpenCLContext>().CopySharedTo(
      &(concat_context->As<OpenCLContext>()));
  kernel->SetContext(std::move(concat_context));
  kernel->Launch();

  auto *wait_list = context->As<OpenCLContext>().cl_wait_list();
  auto *out_ptr = param.output->data<float, cl::Buffer>();
  auto it = wait_list->find(out_ptr);
  if (it != wait_list->end()) {
    VLOG(4) << "--- Find the sync event for the target cl tensor. ---";
    auto &event = *(it->second);
    event.wait();
  } else {
    LOG(FATAL) << "Could not find the sync event for the target cl tensor.";
  }

  // run compute ref and check
  auto *out_ref_data = out_ref.mutable_data<float>(TARGET(kARM));
  concat_mul_compute_ref<float>(ins_data, ins_dim, axis, out_dim, out_ref_data);

  auto *out_data = out.mutable_data<float, cl::Buffer>();
  auto *mapped_out = static_cast<float *>(
      TargetWrapperCL::Map(out_data, 0, sizeof(float) * out_dim.production()));
  for (int i = 0; i < out_dim.production(); i++) {
    EXPECT_NEAR(mapped_out[i], out_ref_data[i], 1e-6);
  }
  TargetWrapperCL::Unmap(out_data, mapped_out);
  TargetWrapperCL::Unmap(x0_data, mapped_x0);
  TargetWrapperCL::Unmap(x1_data, mapped_x1);
  TargetWrapperCL::Unmap(x2_data, mapped_x2);
}
#endif  // concat_buffer

// #define LOOP_TEST
// #define PRINT_RESULT
TEST(concat_image2d_fp32, compute) {
  LOG(INFO) << "main steps of test: host -> layout(buf2img) -> concat(img) -> "
               "layout(img2buf) "
               "-> host";

#ifdef LOOP_TEST
  for (int n = 1; n <= 100; n += 33) {
    for (auto c : {1, 3}) {
      for (int h = 12; h <= 100; h += 13) {
        for (int w = 12; w <= 100; w += 25) {
          for (atuo &axis : {0, 1, 2, 3}) {
#else
  const int n = 1;
  const int c = 2;
  const int h = 3;
  const int w = 4;
  const int axis = 1;
#endif  // LOOP_TEST
            LOG(INFO) << "======== input shape[n,c,h,w]:" << n << " " << c
                      << " " << h << " " << w << " ========";
            LOG(INFO) << "======== axis: " << axis;
            // set layout kernels
            auto buf_to_img_kernels =
                KernelRegistry::Global().Create("layout",
                                                TARGET(kOpenCL),
                                                PRECISION(kAny),
                                                DATALAYOUT(kImageDefault));
            auto buf_to_img_kernels1 =
                KernelRegistry::Global().Create("layout",
                                                TARGET(kOpenCL),
                                                PRECISION(kAny),
                                                DATALAYOUT(kImageDefault));
            auto img_to_buf_kernels = KernelRegistry::Global().Create(
                "layout", TARGET(kOpenCL), PRECISION(kAny), DATALAYOUT(kNCHW));
            auto concat_img_kernels =
                KernelRegistry::Global().Create("concat",
                                                TARGET(kOpenCL),
                                                PRECISION(kFloat),
                                                DATALAYOUT(kImageDefault));
            ASSERT_FALSE(buf_to_img_kernels.empty());
            ASSERT_FALSE(buf_to_img_kernels1.empty());
            ASSERT_FALSE(img_to_buf_kernels.empty());
            ASSERT_FALSE(concat_img_kernels.empty());

            auto buf_to_img_kernel = std::move(buf_to_img_kernels.front());
            auto buf_to_img_kernel1 = std::move(buf_to_img_kernels1.front());
            auto img_to_buf_kernel = std::move(img_to_buf_kernels.front());
            auto concat_img_kernel = std::move(concat_img_kernels.front());
            LOG(INFO) << "get 1st kernel: " << buf_to_img_kernel->doc();
            LOG(INFO) << "get 1st-1 kernel: " << buf_to_img_kernel1->doc();
            LOG(INFO) << "get 2nd kernel: " << img_to_buf_kernel->doc();
            LOG(INFO) << "get 3rd kernel: " << concat_img_kernel->doc();

            // set tensors about op param
            LOG(INFO) << "set tensors about op param";
            lite::Tensor x0, x1, y, concat_in0, concat_in1, concat_out, y_ref;
            operators::LayoutParam BufferToImageParam0, BufferToImageParam1;
            operators::LayoutParam ImageToBufferParam;
            BufferToImageParam0.x = &x0;
            BufferToImageParam0.y = &concat_in0;
            BufferToImageParam1.x = &x1;
            BufferToImageParam1.y = &concat_in1;
            ImageToBufferParam.x = &concat_out;
            ImageToBufferParam.y = &y;
            std::vector<lite::Tensor *> ins;
            operators::ConcatParam concatParam;
            ins.push_back(&concat_in0);
            ins.push_back(&concat_in1);
            concatParam.x = ins;
            concatParam.axis = axis;
            concatParam.output = &concat_out;

            const DDim x0_dim = DDim(std::vector<DDim::value_type>{n, c, h, w});
            DDim x1_dim = DDim(std::vector<DDim::value_type>{n, c, h, w});
            DDim out_dim = DDim(std::vector<DDim::value_type>{n, c, h, w});
            x1_dim[axis] += 2;
            out_dim[axis] = x0_dim[axis] + x1_dim[axis];
            x0.Resize(x0_dim);
            x1.Resize(x1_dim);
            y.Resize(out_dim);
            concat_in0.Resize(x0_dim);
            concat_in1.Resize(x1_dim);
            concat_out.Resize(out_dim);
            y_ref.Resize(out_dim);
            auto concat_image2d_shape =
                paddle::lite::kernels::opencl::InitImageDimInfoWith(out_dim);
            auto concat_image2d_shape_in0 =
                paddle::lite::kernels::opencl::InitImageDimInfoWith(x0_dim);
            auto concat_image2d_shape_in1 =
                paddle::lite::kernels::opencl::InitImageDimInfoWith(x1_dim);

            // initialize tensors
            LOG(INFO) << "initialize tensors";
            auto *x_data0 = x0.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
            auto *x_data1 = x1.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
            auto *y_data = y.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
            auto *y_data_ref = y_ref.mutable_data<float>(TARGET(kARM));
            auto *mapped_x0 = static_cast<float *>(TargetWrapperCL::Map(
                x_data0, 0, sizeof(float) * x0_dim.production()));
            auto *mapped_x1 = static_cast<float *>(TargetWrapperCL::Map(
                x_data1, 0, sizeof(float) * x1_dim.production()));
            auto *mapped_y = static_cast<float *>(TargetWrapperCL::Map(
                y_data, 0, sizeof(float) * out_dim.production()));
            for (int i = 0; i < x0_dim.production(); ++i) {
              mapped_x0[i] = static_cast<int>(i) - x0_dim.production() / 2;
            }
            for (int i = 0; i < x1_dim.production(); ++i) {
              mapped_x1[i] = static_cast<int>(i) - x1_dim.production() / 2;
            }
            for (int i = 0; i < out_dim.production(); ++i) {
              mapped_y[i] = static_cast<int>(0);
            }
            auto *concat_in_data0 = concat_in0.mutable_data<float, cl::Image2D>(
                concat_image2d_shape_in0["width"],
                concat_image2d_shape_in0["height"]);
            auto *concat_in_data1 = concat_in1.mutable_data<float, cl::Image2D>(
                concat_image2d_shape_in1["width"],
                concat_image2d_shape_in1["height"]);
            auto *concat_out_data = concat_out.mutable_data<float, cl::Image2D>(
                concat_image2d_shape["width"], concat_image2d_shape["height"]);

            // set context and kernel args
            LOG(INFO) << "set context and kernel args";
            std::unique_ptr<KernelContext> context(new KernelContext);
            context->As<OpenCLContext>().InitOnce();

            buf_to_img_kernel->SetParam(BufferToImageParam0);
            std::unique_ptr<KernelContext> buf_to_img_context(
                new KernelContext);
            context->As<OpenCLContext>().CopySharedTo(
                &(buf_to_img_context->As<OpenCLContext>()));
            buf_to_img_kernel->SetContext(std::move(buf_to_img_context));
            buf_to_img_kernel1->SetParam(BufferToImageParam1);
            std::unique_ptr<KernelContext> buf_to_img_context1(
                new KernelContext);
            context->As<OpenCLContext>().CopySharedTo(
                &(buf_to_img_context1->As<OpenCLContext>()));
            buf_to_img_kernel1->SetContext(std::move(buf_to_img_context1));

            img_to_buf_kernel->SetParam(ImageToBufferParam);
            std::unique_ptr<KernelContext> img_to_buf_context(
                new KernelContext);
            context->As<OpenCLContext>().CopySharedTo(
                &(img_to_buf_context->As<OpenCLContext>()));
            img_to_buf_kernel->SetContext(std::move(img_to_buf_context));

            concat_img_kernel->SetParam(concatParam);
            std::unique_ptr<KernelContext> concat_img_context(
                new KernelContext);
            context->As<OpenCLContext>().CopySharedTo(
                &(concat_img_context->As<OpenCLContext>()));
            concat_img_kernel->SetContext(std::move(concat_img_context));

            // run kernels
            LOG(INFO) << "run kernel: buf_to_img_kernel";
            buf_to_img_kernel->Launch();
            buf_to_img_kernel1->Launch();
            LOG(INFO) << "run kernel: concat_img_kernel";
            concat_img_kernel->Launch();
            LOG(INFO) << "run kernel: img_to_buf_kernel";
            img_to_buf_kernel->Launch();

            // compute ref cp_u
            std::vector<const float *> ins_ptr;
            std::vector<const DDim> in_dim;
            ins_ptr.push_back(mapped_x0);
            ins_ptr.push_back(mapped_x1);
            in_dim.push_back(x0_dim);
            in_dim.push_back(x1_dim);
            concat_mul_compute_ref<float>(
                ins_ptr, in_dim, axis, out_dim, y_data_ref);
// result
#ifdef PRINT_RESULT
            LOG(INFO) << "---- print kernel result (input -> output) ----";
            for (int eidx = 0; eidx < out_dim.production(); ++eidx) {
              std::cout << mapped_x0[eidx] << ", " << mapped_x1[eidx] << " -> "
                        << mapped_y[eidx] << std::endl;
            }
#endif  // PRINT_RESULT

            // check result: compare kernel output and cpu output(y_data_ref)
            for (int eidx = 0; eidx < out_dim.production(); eidx++) {
              EXPECT_NEAR(y_data_ref[eidx], mapped_y[eidx], 1e-6);
              if (abs(y_data_ref[eidx] - mapped_y[eidx]) > 1e-6) {
                LOG(INFO) << "1st diff in this case at eidx[from 0]:" << eidx
                          << " / " << x0_dim.production() << ", y_data_ref["
                          << eidx << "]:" << y_data_ref[eidx] << ", mapped_y["
                          << eidx << "]:" << mapped_y[eidx];
                break;
              }
            }
            // free
            LOG(INFO) << "free: unmap x, y";
            TargetWrapperCL::Unmap(x_data0, mapped_x0);
            TargetWrapperCL::Unmap(x_data1, mapped_x1);
            TargetWrapperCL::Unmap(y_data, mapped_y);
#ifdef LOOP_TEST
          }  // axis
        }    // w
      }      // h
    }        // c
  }          // n
#else
// nothing to do.
#endif
}
}  // namespace lite
}  // namespace paddle

// concat buffer
// USE_LITE_KERNEL(concat, kOpenCL, kFloat, kNCHW, def);

// concat image2d fp32
USE_LITE_KERNEL(layout, kOpenCL, kAny, kImageDefault, NCHW_to_ImageDefault);
USE_LITE_KERNEL(layout, kOpenCL, kAny, kNCHW, ImageDefault_to_NCHW);
USE_LITE_KERNEL(concat, kOpenCL, kFloat, kImageDefault, ImageDefault);
