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
void fill_data(dtype *x, const int length, int set_value = -1) {
  if (set_value == -1) {
    for (size_t idx = 0; idx < length; ++idx) {
      x[idx] = idx;
    }
  } else if (set_value != -1) {
    for (size_t idx = 0; idx < length; ++idx) {
      x[idx] = set_value;
    }
  }
}

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
  VLOG(4) << "axis:" << axis;
  VLOG(4) << "batch:" << batch;
  VLOG(4) << "cahnnels:" << channels;
  VLOG(4) << "num:" << num;
  // do elementwise add/sub/max/...
  if (elt_type == "add" && axis == 1 && y_dims.size() == 1) {
    for (int i = 0; i < x_dims.production(); ++i) {
      auto w = i % y_dims.production();
      out_data[i] = x_data[i] + y_data[w];
    }
  } else if (elt_type == "add") {
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
  } else {
    LOG(FATAL) << "unsupported Elementwise type: " << elt_type;
  }
}

// buffer
#if 0
TEST(elementwise_add_buffer, compute) {
  LOG(INFO) << "to get kernel ...";
  auto kernels = KernelRegistry::Global().Create(
      "elementwise_add", TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNCHW));
  ASSERT_FALSE(kernels.empty());

  auto kernel = std::move(kernels.front());

  LOG(INFO) << "get kernel";

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

  CLRuntime::Global()->command_queue().finish();

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

TEST(fusion_elementwise_add_activation_buffer, compute) {
  LOG(INFO) << "to get kernel ...";
  auto kernels =
      KernelRegistry::Global().Create("fusion_elementwise_add_activation",
                                      TARGET(kOpenCL),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW));
  ASSERT_FALSE(kernels.empty());

  auto kernel = std::move(kernels.front());

  LOG(INFO) << "get kernel";

  lite::Tensor x, y, out;
  operators::FusionElementwiseActivationParam param;
  param.X = &x;
  param.Y = &y;
  param.Out = &out;
  param.axis = -1;
  param.act_type = "relu";

  std::unique_ptr<KernelContext> context(new KernelContext);
  context->As<OpenCLContext>().InitOnce();

  kernel->SetParam(param);
  std::unique_ptr<KernelContext> ele_context(new KernelContext);
  context->As<OpenCLContext>().CopySharedTo(
      &(ele_context->As<OpenCLContext>()));
  kernel->SetContext(std::move(ele_context));

  const DDim x_dim = DDim(std::vector<DDim::value_type>{30, 20, 10, 50});
  const DDim y_dim = DDim(std::vector<DDim::value_type>{20, 10, 50});
  const DDim out_dim = DDim(std::vector<DDim::value_type>{30, 20, 10, 50});
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

  CLRuntime::Global()->command_queue().finish();

  std::unique_ptr<float[]> out_ref(new float[out_dim.production()]);
  elementwise_compute_ref<float>(
      mapped_x, mapped_y, out_ref.get(), x_dim, y_dim, param.axis, "add", true);

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
#endif

// image
TEST(elementwise_add_image2d_fp32, compute) {
  LOG(INFO) << "main steps of test: host -> layout(buf2img on cpu) -> "
               "elementwise_add(img) -> "
               "layout(img2buf on cpu) "
               "-> host";

  // elementwise_add's 3 kernels selection routing strategy:
  // --------------------------------------------------------
  //  1. elementwise_add: Need y_dim.size() == 4
  //  2. elementwise_add (used by fuse_elementwise_activation op):
  //                      Need y_dim.size() == 4 && act_type == "relu"
  //  3. width_add:       Need y_dim.size() == 1 && x_dim.size() == 4 && axis ==
  //  3
  //  4. channel_add:     Need y_dim.size() == 1 && x_dim.size() == 4 && axis ==
  //  1

  // dims
  const int n = 1;
  const int c = 3;
  const int h = 2;
  const int w = 2;

  const DDim x_dim = DDim(std::vector<DDim::value_type>{n, c, h, w});
  auto out_dim = x_dim;
  // y_dim / axis / relu_flag
  std::vector<DDim> y_dim_v{DDim(std::vector<DDim::value_type>{n, c, h, w}),
                            DDim(std::vector<DDim::value_type>{n, c, h, w}),
                            DDim(std::vector<DDim::value_type>{w}),
                            DDim(std::vector<DDim::value_type>{w})};
  std::vector<int> axis_v{-1, -1, 3, 1};
  std::vector<bool> relu_flag_v{false, true, false, false};
  CHECK(y_dim_v.size() == axis_v.size() && axis_v.size() == relu_flag_v.size())
      << "y_dim_v.size() == axis_v.size() == relu_flag_v.size() should be "
         "same, and be corresponding "
         "one by one";

  // start loop
  for (size_t case_idx = 0; case_idx < y_dim_v.size(); ++case_idx) {
    auto y_dim = y_dim_v[case_idx];
    auto axis = axis_v[case_idx];
    auto relu_flag = relu_flag_v[case_idx];
    LOG(INFO) << "================== elementwise_add, case_idx:" << case_idx + 1
              << "/" << y_dim_v.size() << " ===================";
    LOG(INFO) << "x_dim:" << x_dim;
    LOG(INFO) << "y_dim:" << y_dim;
    LOG(INFO) << "out_dim:" << out_dim;
    LOG(INFO) << "axis:" << axis;
    LOG(INFO) << "relu_flag:" << relu_flag;

    // tensor
    VLOG(4) << "set tensors about op param";
    lite::Tensor eleadd_x, eleadd_y, eleadd_out;
    eleadd_x.Resize(x_dim);
    eleadd_y.Resize(y_dim);
    eleadd_out.Resize(out_dim);

    // initialize tensors
    VLOG(4) << "initialize tensors";
    paddle::lite::CLImageConverterDefault default_convertor;
    // x
    std::vector<float> x_v(x_dim.production());
    fill_data<float>(x_v.data(), x_v.size());  // fill with index value
    auto x_img_shape = default_convertor.InitImageDimInfoWith(x_dim);  // w, h
    auto x_img_w = x_img_shape[0];
    auto x_img_h = x_img_shape[1];
    std::vector<float> x_img_v(x_img_w * x_img_h * 4);  // 4: RGBA
    default_convertor.NCHWToImage(x_v.data(), x_img_v.data(), x_dim);
    eleadd_x.mutable_data<float, cl::Image2D>(x_img_w, x_img_h, x_img_v.data());

    // y
    std::vector<float> y_v(y_dim.production());
    fill_data<float>(y_v.data(), y_v.size());  // fill with index value
    auto y_img_shape = default_convertor.InitImageDimInfoWith(y_dim);  // w, h
    auto y_img_w = y_img_shape[0];
    auto y_img_h = y_img_shape[1];
    std::vector<float> y_img_v(y_img_shape[0] * y_img_shape[1] * 4);  // 4: RGBA
    default_convertor.NCHWToImage(y_v.data(), y_img_v.data(), y_dim);
    eleadd_y.mutable_data<float, cl::Image2D>(y_img_w, y_img_h, y_img_v.data());

    // out
    auto out_img_shape =
        default_convertor.InitImageDimInfoWith(out_dim);  // w, h
    auto out_img_w = out_img_shape[0];
    auto out_img_h = out_img_shape[1];
    eleadd_out.mutable_data<float, cl::Image2D>(out_img_w, out_img_h);

    std::vector<float> out_img_v(out_img_w * out_img_h * 4);
    fill_data<float>(
        out_img_v.data(), out_img_v.size(), 0);  // fill with zero value

    std::vector<float> out_v(out_dim.production());

    // operator param
    operators::FusionElementwiseActivationParam
        fuseEleaddParam;  // enabled if relu_flag is true
    fuseEleaddParam.X = &eleadd_x;
    fuseEleaddParam.Y = &eleadd_y;
    fuseEleaddParam.Out = &eleadd_out;
    fuseEleaddParam.axis = axis;
    fuseEleaddParam.act_type = relu_flag ? "relu" : "";

    operators::ElementwiseParam eleaddParam;
    eleaddParam.X = &eleadd_x;
    eleaddParam.Y = &eleadd_y;
    eleaddParam.Out = &eleadd_out;
    eleaddParam.axis = axis;

    auto op_param = relu_flag ? fuseEleaddParam : eleaddParam;

    // set kernel
    auto eleadd_img_kernels =
        KernelRegistry::Global().Create("elementwise_add",
                                        TARGET(kOpenCL),
                                        PRECISION(kFloat),
                                        DATALAYOUT(kImageDefault));
    ASSERT_FALSE(eleadd_img_kernels.empty());

    auto eleadd_img_kernel = std::move(eleadd_img_kernels.front());
    VLOG(4) << "get eleadd kernel: " << eleadd_img_kernel->doc();

    // set context and kernel args
    VLOG(4) << "set context and kernel args";
    std::unique_ptr<KernelContext> context(new KernelContext);
    context->As<OpenCLContext>().InitOnce();

    eleadd_img_kernel->SetParam(op_param);
    std::unique_ptr<KernelContext> eleadd_img_context(new KernelContext);
    context->As<OpenCLContext>().CopySharedTo(
        &(eleadd_img_context->As<OpenCLContext>()));
    eleadd_img_kernel->SetContext(std::move(eleadd_img_context));

    // run kernel
    VLOG(4) << "run kernel";
    eleadd_img_kernel->Launch();

    // download gpu result to cpu
    const size_t cl_image2d_row_pitch{0};
    const size_t cl_image2d_slice_pitch{0};
    TargetWrapperCL::ImgcpySync(out_img_v.data(),
                                eleadd_out.data<float, cl::Image2D>(),
                                out_img_w,
                                out_img_h,
                                cl_image2d_row_pitch,
                                cl_image2d_slice_pitch,
                                IoDirection::DtoH);
    default_convertor.ImageToNCHW(
        out_img_v.data(), out_v.data(), out_img_shape, out_dim);

    // compute cpu reference
    std::unique_ptr<float[]> out_ref(new float[out_dim.production()]);
    elementwise_compute_ref<float>(x_v.data(),
                                   y_v.data(),
                                   out_ref.get(),
                                   x_dim,
                                   y_dim,
                                   op_param.axis,
                                   "add",
                                   relu_flag);

#if 0  // enable to check value of x and y
    for (int eidx = 0; eidx < out_dim.production(); eidx++) {
      auto value = out_v[eidx];
      auto ref_value = out_ref.get()[eidx];
        LOG(INFO) << "1st diff in this case at eidx[from 0]:" << eidx << " / "
                  << out_dim.production() << ", x_v[" << eidx << "]:"
                  << x_v[eidx] << ", value[" << eidx << "]:" << value
                  << ", ref_value[" << eidx << "]:" << ref_value;
    }

    for (int i = 0; i < y_v.size(); i++) {
      LOG(INFO) << "y_v[" << i << "]:" << y_v[i];
    }
#endif

    for (int eidx = 0; eidx < out_dim.production(); eidx++) {
      auto value = out_v[eidx];
      auto ref_value = out_ref.get()[eidx];
      EXPECT_NEAR(value, ref_value, 1e-6);
      if (abs(value - ref_value) > 1e-6) {
        LOG(INFO) << "1st diff in this case at eidx[from 0]:" << eidx << " / "
                  << out_dim.production() << ", value[" << eidx << "]:" << value
                  << ", ref_value[" << eidx << "]:" << ref_value;
        break;
      }
    }
  }
}

}  // namespace lite
}  // namespace paddle

// USE_LITE_KERNEL(elementwise_add, kOpenCL, kFloat, kNCHW, def);
// USE_LITE_KERNEL(fusion_elementwise_add_activation, kOpenCL, kFloat, kNCHW,
// def);

USE_LITE_KERNEL(elementwise_add, kOpenCL, kFloat, kImageDefault, def);
USE_LITE_KERNEL(
    fusion_elementwise_add_activation, kOpenCL, kFloat, kImageDefault, def);
