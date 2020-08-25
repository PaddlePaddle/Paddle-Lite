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
#include "lite/kernels/opencl/test_helper.h"

#define FP16_MAX_DIFF (5e-1)

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

//  #define LOOP_TEST_CONCAT2
//  #define PRINT_RESULT_CONCAT2
#define CHECK_RESULT_CONCAT2
TEST(concat_image2d_concat2, compute) {
  LOG(INFO) << "main steps of test: host -> layout(buf2img) -> concat(img) -> "
               "layout(img2buf) "
               "-> host";
#ifdef LOOP_TEST_CONCAT2
  const int axis = 1;
  const int n = 1;
  for (int c = 1; c < 4; ++c) {
    for (int h = 1; h < 50; ++h) {
      for (int w = 1; w < 50; ++w) {
#else
  const int axis = 1;
  const int n = 1;
  const int c = 3;   // 1;
  const int h = 15;  // 2;
  const int w = 15;  // 42;
#endif  // LOOP_TEST_CONCAT2
        LOG(INFO) << "======== input shape[n,c,h,w]:" << n << " " << c << " "
                  << h << " " << w << " ========";
        LOG(INFO) << "======== axis: " << axis;

        // set layout kernels
        std::list<std::unique_ptr<KernelBase>> buf_to_img_kernels =
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
                                            PRECISION(kFP16),
                                            DATALAYOUT(kImageDefault));
        ASSERT_FALSE(buf_to_img_kernels.empty());
        ASSERT_FALSE(buf_to_img_kernels1.empty());
        ASSERT_FALSE(img_to_buf_kernels.empty());
        ASSERT_FALSE(concat_img_kernels.empty());

        auto buf_to_img_kernel = std::move(buf_to_img_kernels.front());
        auto buf_to_img_kernel1 = std::move(buf_to_img_kernels1.front());
        auto img_to_buf_kernel = std::move(img_to_buf_kernels.front());
        auto concat_img_kernel = std::move(concat_img_kernels.front());
        LOG(INFO) << "get 1st-0 kernel: " << buf_to_img_kernel->doc();
        LOG(INFO) << "get 1st-1 kernel: " << buf_to_img_kernel1->doc();
        LOG(INFO) << "get 2nd kernel: " << img_to_buf_kernel->doc();
        LOG(INFO) << "get 3rd kernel: " << concat_img_kernel->doc();

        // set tensors about op param
        LOG(INFO) << "set tensors about op param";
        lite::Tensor x0, x1, y, concat_in0, concat_in1, concat_out, y_ref;
        operators::LayoutParam BufferToImageParam0, BufferToImageParam1,
            BufferToImageParam2;
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

        DDim x0_dim = DDim(std::vector<DDim::value_type>{n, c, h, w});
        DDim x1_dim = DDim(std::vector<DDim::value_type>{n, c, h, w});
        DDim out_dim = DDim(std::vector<DDim::value_type>{n, c, h, w});
        // note: used to make cases with different channel
        // x1_dim[axis] += 2;
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
          mapped_x0[i] = i + 1;  //  (i+1)  * 0.01;//*/static_cast<int>(i) -
                                 //  x0_dim.production() / 2 * 0.00987;
        }
        for (int i = 0; i < x1_dim.production(); ++i) {
          mapped_x1[i] = x0_dim.production() + i +
                         1;  //  (i+1) * 0.1;//*/static_cast<int>(i) -
                             //  x1_dim.production() / 2 * 0.00987;
        }
        for (int i = 0; i < out_dim.production(); ++i) {
          mapped_y[i] = static_cast<int>(0);
        }
        auto *concat_in_data0 = concat_in0.mutable_data<half_t, cl::Image2D>(
            concat_image2d_shape_in0["width"],
            concat_image2d_shape_in0["height"]);
        auto *concat_in_data1 = concat_in1.mutable_data<half_t, cl::Image2D>(
            concat_image2d_shape_in1["width"],
            concat_image2d_shape_in1["height"]);
        auto *concat_out_data = concat_out.mutable_data<half_t, cl::Image2D>(
            concat_image2d_shape["width"], concat_image2d_shape["height"]);

        // set context and kernel args
        LOG(INFO) << "set context and kernel args";
        std::unique_ptr<KernelContext> context(new KernelContext);
        context->As<OpenCLContext>().InitOnce();

        buf_to_img_kernel->SetParam(BufferToImageParam0);
        std::unique_ptr<KernelContext> buf_to_img_context(new KernelContext);
        context->As<OpenCLContext>().CopySharedTo(
            &(buf_to_img_context->As<OpenCLContext>()));
        buf_to_img_kernel->SetContext(std::move(buf_to_img_context));

        buf_to_img_kernel1->SetParam(BufferToImageParam1);
        std::unique_ptr<KernelContext> buf_to_img_context1(new KernelContext);
        context->As<OpenCLContext>().CopySharedTo(
            &(buf_to_img_context1->As<OpenCLContext>()));
        buf_to_img_kernel1->SetContext(std::move(buf_to_img_context1));

        img_to_buf_kernel->SetParam(ImageToBufferParam);
        std::unique_ptr<KernelContext> img_to_buf_context(new KernelContext);
        context->As<OpenCLContext>().CopySharedTo(
            &(img_to_buf_context->As<OpenCLContext>()));
        img_to_buf_kernel->SetContext(std::move(img_to_buf_context));

        concat_img_kernel->SetParam(concatParam);
        std::unique_ptr<KernelContext> concat_img_context(new KernelContext);
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

        CLRuntime::Global()->command_queue().finish();

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
#ifdef PRINT_RESULT_CONCAT2
        LOG(INFO) << "---- print kernel result (input -> output) ----";
        for (int i = 0; i < out_dim.production(); ++i) {
          auto abs_diff = abs(y_data_ref[i] - mapped_y[i]);
          auto relative_diff =
              COMPUTE_RELATIVE_DIFF(y_data_ref[i], mapped_y[i]);
          LOG(INFO) << "idx:" << i << " mapped_y[" << i << "]:" << mapped_y[i]
                    << " y_data_ref[" << i << "]:" << y_data_ref[i]
                    << " abs_diff:" << abs_diff
                    << " relative_diff:" << relative_diff
                    << " FP16_MAX_DIFF:" << FP16_MAX_DIFF;
#if 0  // note: used to write to file and check
              std::cout << "idx:" << i << " mapped_y[" << i
                        << "]:" << mapped_y[i] << " y_data_ref[" << i
                        << "]:" << y_data_ref[i] << " abs_diff:" << abs_diff
                        << " relative_diff:" << relative_diff
                        << " FP16_MAX_DIFF:" << FP16_MAX_DIFF << "\n";
#endif
        }
#endif  // PRINT_RESULT_CONCAT2

#ifdef CHECK_RESULT_CONCAT2
        // check result: compare kernel output and cpu output(y_data_ref)
        for (int i = 0; i < out_dim.production(); i++) {
          auto abs_diff = abs(y_data_ref[i] - mapped_y[i]);
          auto relative_diff =
              COMPUTE_RELATIVE_DIFF(y_data_ref[i], mapped_y[i]);
          EXPECT_EQ(
              (relative_diff <= FP16_MAX_DIFF) || (abs_diff <= FP16_MAX_DIFF),
              true);
          if ((relative_diff > FP16_MAX_DIFF) && (abs_diff > FP16_MAX_DIFF)) {
            LOG(FATAL) << "error idx:" << i << " mapped_y[" << i
                       << "]:" << mapped_y[i] << " y_data_ref[" << i
                       << "]:" << y_data_ref[i] << " abs_diff:" << abs_diff
                       << " relative_diff:" << relative_diff
                       << " FP16_MAX_DIFF:" << FP16_MAX_DIFF;
            break;
          }
        }
#endif
        // free
        LOG(INFO) << "free: unmap x, y";
        TargetWrapperCL::Unmap(x_data0, mapped_x0);
        TargetWrapperCL::Unmap(x_data1, mapped_x1);
        TargetWrapperCL::Unmap(y_data, mapped_y);
#ifdef LOOP_TEST_CONCAT2
      }  // w
    }    // h
  }      // c
#else
// nothing to do.
#endif
}

//  #define LOOP_TEST_CONCAT_MUL
//  #define PRINT_RESULT_CONCAT_MUL
#define CHECK_RESULT_CONCAT_MUL
TEST(concat_image2d_multi, compute) {
  LOG(INFO) << "main steps of test: host -> layout(buf2img) -> concat(img) -> "
               "layout(img2buf) "
               "-> host";
#ifdef LOOP_TEST_CONCAT_MUL
  const int axis = 1;
  const int n = 1;
  for (int c = 1; c < 4; ++c) {
    for (int h = 1; h < 50; ++h) {
      for (int w = 1; w < 50; ++w) {
#else
        const int axis = 1;
        const int n = 1;
        const int c = 3;
        const int h = 4;
        const int w = 5;
#endif  // LOOP_TEST_CONCAT_MUL
        LOG(INFO) << "======== input shape[n,c,h,w]:" << n << " " << c << " "
                  << h << " " << w << " ========";
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
        auto buf_to_img_kernels2 =
            KernelRegistry::Global().Create("layout",
                                            TARGET(kOpenCL),
                                            PRECISION(kAny),
                                            DATALAYOUT(kImageDefault));
        auto img_to_buf_kernels = KernelRegistry::Global().Create(
            "layout", TARGET(kOpenCL), PRECISION(kAny), DATALAYOUT(kNCHW));
        auto concat_img_kernels =
            KernelRegistry::Global().Create("concat",
                                            TARGET(kOpenCL),
                                            PRECISION(kFP16),
                                            DATALAYOUT(kImageDefault));
        ASSERT_FALSE(buf_to_img_kernels.empty());
        ASSERT_FALSE(buf_to_img_kernels1.empty());
        ASSERT_FALSE(buf_to_img_kernels2.empty());
        ASSERT_FALSE(img_to_buf_kernels.empty());
        ASSERT_FALSE(concat_img_kernels.empty());

        auto buf_to_img_kernel = std::move(buf_to_img_kernels.front());
        auto buf_to_img_kernel1 = std::move(buf_to_img_kernels1.front());
        auto buf_to_img_kernel2 = std::move(buf_to_img_kernels2.front());
        auto img_to_buf_kernel = std::move(img_to_buf_kernels.front());
        auto concat_img_kernel = std::move(concat_img_kernels.front());
        LOG(INFO) << "get 1st-0 kernel: " << buf_to_img_kernel->doc();
        LOG(INFO) << "get 1st-1 kernel: " << buf_to_img_kernel1->doc();
        LOG(INFO) << "get 1st-2 kernel: " << buf_to_img_kernel2->doc();
        LOG(INFO) << "get 2nd kernel: " << img_to_buf_kernel->doc();
        LOG(INFO) << "get 3rd kernel: " << concat_img_kernel->doc();

        // set tensors about op param
        LOG(INFO) << "set tensors about op param";
        lite::Tensor x0, x1, x2, y, concat_in0, concat_in1, concat_in2,
            concat_out, y_ref;
        operators::LayoutParam BufferToImageParam0, BufferToImageParam1,
            BufferToImageParam2;
        operators::LayoutParam ImageToBufferParam;
        BufferToImageParam0.x = &x0;
        BufferToImageParam0.y = &concat_in0;
        BufferToImageParam1.x = &x1;
        BufferToImageParam1.y = &concat_in1;
        BufferToImageParam2.x = &x2;
        BufferToImageParam2.y = &concat_in2;
        ImageToBufferParam.x = &concat_out;
        ImageToBufferParam.y = &y;
        std::vector<lite::Tensor *> ins;
        operators::ConcatParam concatParam;
        ins.push_back(&concat_in0);
        ins.push_back(&concat_in1);
        ins.push_back(&concat_in2);
        concatParam.x = ins;
        concatParam.axis = axis;
        concatParam.output = &concat_out;

        DDim x0_dim = DDim(std::vector<DDim::value_type>{n, c, h, w});
        DDim x1_dim = DDim(std::vector<DDim::value_type>{n, c, h, w});
        DDim x2_dim = DDim(std::vector<DDim::value_type>{n, c, h, w});
        DDim out_dim = DDim(std::vector<DDim::value_type>{n, c * 3, h, w});
        // x1_dim[axis] += 2;
        out_dim[axis] = x0_dim[axis] + x1_dim[axis] + x2_dim[axis];
        x0.Resize(x0_dim);
        x1.Resize(x1_dim);
        x2.Resize(x2_dim);
        y.Resize(out_dim);
        concat_in0.Resize(x0_dim);
        concat_in1.Resize(x1_dim);
        concat_in2.Resize(x2_dim);
        concat_out.Resize(out_dim);
        y_ref.Resize(out_dim);
        auto concat_image2d_shape =
            paddle::lite::kernels::opencl::InitImageDimInfoWith(out_dim);
        auto concat_image2d_shape_in0 =
            paddle::lite::kernels::opencl::InitImageDimInfoWith(x0_dim);
        auto concat_image2d_shape_in1 =
            paddle::lite::kernels::opencl::InitImageDimInfoWith(x1_dim);
        auto concat_image2d_shape_in2 =
            paddle::lite::kernels::opencl::InitImageDimInfoWith(x2_dim);

        // initialize tensors
        LOG(INFO) << "initialize tensors";
        auto *x_data0 = x0.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
        auto *x_data1 = x1.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
        auto *x_data2 = x2.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
        auto *y_data = y.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
        auto *y_data_ref = y_ref.mutable_data<float>(TARGET(kARM));
        auto *mapped_x0 = static_cast<float *>(TargetWrapperCL::Map(
            x_data0, 0, sizeof(float) * x0_dim.production()));
        auto *mapped_x1 = static_cast<float *>(TargetWrapperCL::Map(
            x_data1, 0, sizeof(float) * x1_dim.production()));
        auto *mapped_x2 = static_cast<float *>(TargetWrapperCL::Map(
            x_data2, 0, sizeof(float) * x2_dim.production()));
        auto *mapped_y = static_cast<float *>(TargetWrapperCL::Map(
            y_data, 0, sizeof(float) * out_dim.production()));
        for (int i = 0; i < x0_dim.production(); ++i) {
          mapped_x0[i] = i + 1;  //  (i+1)  * 0.01;//*/static_cast<int>(i) -
                                 //  x0_dim.production() / 2 * 0.00987;
        }
        for (int i = 0; i < x1_dim.production(); ++i) {
          mapped_x1[i] = x0_dim.production() + i +
                         1;  //  (i+1) * 0.1;//*/static_cast<int>(i) -
                             //  x1_dim.production() / 2 * 0.00987;
        }
        for (int i = 0; i < x2_dim.production(); ++i) {
          mapped_x2[i] = x0_dim.production() + x1_dim.production() + i +
                         1;  //  (i+1) * 1.;//*/ static_cast<int>(i) -
                             //  x2_dim.production() / 2 * 0.00987;
        }
        for (int i = 0; i < out_dim.production(); ++i) {
          mapped_y[i] = static_cast<int>(0);
        }
        auto *concat_in_data0 = concat_in0.mutable_data<half_t, cl::Image2D>(
            concat_image2d_shape_in0["width"],
            concat_image2d_shape_in0["height"]);
        auto *concat_in_data1 = concat_in1.mutable_data<half_t, cl::Image2D>(
            concat_image2d_shape_in1["width"],
            concat_image2d_shape_in1["height"]);
        auto *concat_in_data2 = concat_in2.mutable_data<half_t, cl::Image2D>(
            concat_image2d_shape_in2["width"],
            concat_image2d_shape_in2["height"]);
        auto *concat_out_data = concat_out.mutable_data<half_t, cl::Image2D>(
            concat_image2d_shape["width"], concat_image2d_shape["height"]);

        // set context and kernel args
        LOG(INFO) << "set context and kernel args";
        std::unique_ptr<KernelContext> context(new KernelContext);
        context->As<OpenCLContext>().InitOnce();

        buf_to_img_kernel->SetParam(BufferToImageParam0);
        std::unique_ptr<KernelContext> buf_to_img_context(new KernelContext);
        context->As<OpenCLContext>().CopySharedTo(
            &(buf_to_img_context->As<OpenCLContext>()));
        buf_to_img_kernel->SetContext(std::move(buf_to_img_context));

        buf_to_img_kernel1->SetParam(BufferToImageParam1);
        std::unique_ptr<KernelContext> buf_to_img_context1(new KernelContext);
        context->As<OpenCLContext>().CopySharedTo(
            &(buf_to_img_context1->As<OpenCLContext>()));
        buf_to_img_kernel1->SetContext(std::move(buf_to_img_context1));

        buf_to_img_kernel2->SetParam(BufferToImageParam2);
        std::unique_ptr<KernelContext> buf_to_img_context2(new KernelContext);
        context->As<OpenCLContext>().CopySharedTo(
            &(buf_to_img_context2->As<OpenCLContext>()));
        buf_to_img_kernel2->SetContext(std::move(buf_to_img_context2));

        img_to_buf_kernel->SetParam(ImageToBufferParam);
        std::unique_ptr<KernelContext> img_to_buf_context(new KernelContext);
        context->As<OpenCLContext>().CopySharedTo(
            &(img_to_buf_context->As<OpenCLContext>()));
        img_to_buf_kernel->SetContext(std::move(img_to_buf_context));

        concat_img_kernel->SetParam(concatParam);
        std::unique_ptr<KernelContext> concat_img_context(new KernelContext);
        context->As<OpenCLContext>().CopySharedTo(
            &(concat_img_context->As<OpenCLContext>()));
        concat_img_kernel->SetContext(std::move(concat_img_context));

        // run kernels
        LOG(INFO) << "run kernel: buf_to_img_kernel";
        buf_to_img_kernel->Launch();
        buf_to_img_kernel1->Launch();
        buf_to_img_kernel2->Launch();
        LOG(INFO) << "run kernel: concat_img_kernel";
        concat_img_kernel->Launch();
        LOG(INFO) << "run kernel: img_to_buf_kernel";
        img_to_buf_kernel->Launch();

        CLRuntime::Global()->command_queue().finish();

        // compute ref cp_u
        std::vector<const float *> ins_ptr;
        std::vector<const DDim> in_dim;
        ins_ptr.push_back(mapped_x0);
        ins_ptr.push_back(mapped_x1);
        ins_ptr.push_back(mapped_x2);
        in_dim.push_back(x0_dim);
        in_dim.push_back(x1_dim);
        in_dim.push_back(x2_dim);
        concat_mul_compute_ref<float>(
            ins_ptr, in_dim, axis, out_dim, y_data_ref);
// result
#ifdef PRINT_RESULT_CONCAT_MUL
        LOG(INFO) << "---- print kernel result (input -> output) ----";
        for (int i = 0; i < out_dim.production(); ++i) {
          auto abs_diff = abs(y_data_ref[i] - mapped_y[i]);
          auto relative_diff =
              COMPUTE_RELATIVE_DIFF(y_data_ref[i], mapped_y[i]);
          LOG(INFO) << "idx:" << i << " mapped_y[" << i << "]:" << mapped_y[i]
                    << " y_data_ref[" << i << "]:" << y_data_ref[i]
                    << " abs_diff:" << abs_diff
                    << " relative_diff:" << relative_diff
                    << " FP16_MAX_DIFF:" << FP16_MAX_DIFF;
#if 0  // note: used to write to file and check
              std::cout << "idx:" << i << " mapped_y[" << i
                        << "]:" << mapped_y[i] << " y_data_ref[" << i
                        << "]:" << y_data_ref[i] << " abs_diff:" << abs_diff
                        << " relative_diff:" << relative_diff
                        << " FP16_MAX_DIFF:" << FP16_MAX_DIFF << "\n";
#endif
        }
#endif  // PRINT_RESULT_CONCAT_MUL

#ifdef CHECK_RESULT_CONCAT_MUL
        // check result: compare kernel output and cpu output(y_data_ref)
        for (int i = 0; i < out_dim.production(); i++) {
          auto abs_diff = abs(y_data_ref[i] - mapped_y[i]);
          auto relative_diff =
              COMPUTE_RELATIVE_DIFF(y_data_ref[i], mapped_y[i]);
          if ((relative_diff > FP16_MAX_DIFF) && (abs_diff > FP16_MAX_DIFF)) {
            LOG(FATAL) << "error idx:" << i << " mapped_y[" << i
                       << "]:" << mapped_y[i] << " y_data_ref[" << i
                       << "]:" << y_data_ref[i] << " abs_diff:" << abs_diff
                       << " relative_diff:" << relative_diff
                       << " FP16_MAX_DIFF:" << FP16_MAX_DIFF;
            break;
          }
        }
#endif
        // free
        LOG(INFO) << "free: unmap x, y";
        TargetWrapperCL::Unmap(x_data0, mapped_x0);
        TargetWrapperCL::Unmap(x_data1, mapped_x1);
        TargetWrapperCL::Unmap(x_data2, mapped_x2);
        TargetWrapperCL::Unmap(y_data, mapped_y);
#ifdef LOOP_TEST_CONCAT_MUL
      }  // w
    }    // h
  }      // c
#else
// nothing to do.
#endif
}
}  // namespace lite
}  // namespace paddle

// concat buffer
// USE_LITE_KERNEL(concat, kOpenCL, kFP16, kNCHW, def);

// concat image2d fp32
USE_LITE_KERNEL(layout, kOpenCL, kAny, kImageDefault, NCHW_to_ImageDefault);
USE_LITE_KERNEL(layout, kOpenCL, kAny, kNCHW, ImageDefault_to_NCHW);
USE_LITE_KERNEL(concat, kOpenCL, kFP16, kImageDefault, ImageDefault);
