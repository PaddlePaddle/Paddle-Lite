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

  if (elt_type == "mul") {
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

// #define PRINT_RESULT
TEST(elemul_image2d_fp32, compute_kernel_elemenwise_mul) {
  LOG(INFO)
      << "main steps of test: host -> layout(buf2img on cpu) -> elemul(img) -> "
         "layout(img2buf on cpu) "
         "-> host";

  const int n = 1;
  const int c = 2;
  const int h = 3;
  const int w = 4;

  LOG(INFO) << "======== input shape[n,c,h,w]:" << n << " " << c << " " << h
            << " " << w << " ========";

  const DDim x_dim = DDim(std::vector<DDim::value_type>{n, c, h, w});
  auto y_dim = x_dim;
  auto out_dim = x_dim;

  LOG(INFO) << "set tensors about op param";
  lite::Tensor x, y, out;     // cpu
  lite::Tensor x_img, y_img;  // cpu
  lite::Tensor elemul_x, elemul_y, elemul_out;

  x.Resize(x_dim);
  y.Resize(y_dim);
  out.Resize(out_dim);

  elemul_x.Resize(x_dim);
  elemul_y.Resize(y_dim);
  elemul_out.Resize(out_dim);

  paddle::lite::CLImageConverterDefault default_convertor;

  // initialize tensors
  LOG(INFO) << "initialize tensors";
  auto *x_data = x.mutable_data<float>();
  fill_data<float>(x_data, x.dims().production());
  auto *y_data = y.mutable_data<float>();
  fill_data<float>(y_data, y.dims().production());
  auto *out_data = out.mutable_data<float>();
  fill_data<float>(out_data, out.dims().production(), 0);

  // x
  std::vector<float> x_v(x_dim.production());
  fill_data<float>(x_v.data(), x_v.size());  // fill with index value
  auto x_img_shape = default_convertor.InitImageDimInfoWith(x_dim);  // w, h
  auto x_img_w = x_img_shape[0];
  auto x_img_h = x_img_shape[1];
  std::vector<float> x_img_v(x_img_w * x_img_h * 4);  // 4: RGBA
  default_convertor.NCHWToImage(x_v.data(), x_img_v.data(), x_dim);
  elemul_x.mutable_data<float, cl::Image2D>(x_img_w, x_img_h, x_img_v.data());

  // y
  std::vector<float> y_v(y_dim.production());
  fill_data<float>(y_v.data(), y_v.size());  // fill with index value
  auto y_img_shape = default_convertor.InitImageDimInfoWith(y_dim);  // w, h
  auto y_img_w = y_img_shape[0];
  auto y_img_h = y_img_shape[1];
  std::vector<float> y_img_v(y_img_shape[0] * y_img_shape[1] * 4);  // 4: RGBA
  default_convertor.NCHWToImage(y_v.data(), y_img_v.data(), y_dim);
  elemul_y.mutable_data<float, cl::Image2D>(y_img_w, y_img_h, y_img_v.data());

  // out
  auto out_img_shape = default_convertor.InitImageDimInfoWith(out_dim);  // w, h
  auto out_img_w = out_img_shape[0];
  auto out_img_h = out_img_shape[1];
  // elemul_out.mutable_data<float, cl::Image2D>(out_img_w, out_img_h);

  std::vector<float> out_img_v(out_img_w * out_img_h * 4);
  fill_data<float>(
      out_img_v.data(), out_img_v.size(), 0);  // fill with zero value

  std::vector<float> out_v(out_dim.production());

  // operator param
  operators::ElementwiseParam elemulParam;
  elemulParam.X = &elemul_x;
  elemulParam.Y = &elemul_y;
  elemulParam.Out = &elemul_out;
  elemulParam.axis = -1;

  // set kernel
  auto elemul_img_kernels =
      KernelRegistry::Global().Create("elementwise_mul",
                                      TARGET(kOpenCL),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kImageDefault));
  ASSERT_FALSE(elemul_img_kernels.empty());

  auto elemul_img_kernel = std::move(elemul_img_kernels.front());
  LOG(INFO) << "get elemul kernel: " << elemul_img_kernel->doc();

  // set context and kernel args
  LOG(INFO) << "set context and kernel args";
  std::unique_ptr<KernelContext> context(new KernelContext);
  context->As<OpenCLContext>().InitOnce();

  elemul_img_kernel->SetParam(elemulParam);
  std::unique_ptr<KernelContext> elemul_img_context(new KernelContext);
  context->As<OpenCLContext>().CopySharedTo(
      &(elemul_img_context->As<OpenCLContext>()));
  elemul_img_kernel->SetContext(std::move(elemul_img_context));

  // run kernel
  LOG(INFO) << "run kernel: elemul_img_kernel";
  elemul_img_kernel->Launch();

  // download gpu result to cpu
  const size_t cl_image2d_row_pitch{0};
  const size_t cl_image2d_slice_pitch{0};
  TargetWrapperCL::ImgcpySync(out_img_v.data(),
                              elemul_out.data<float, cl::Image2D>(),
                              out_img_w,
                              out_img_h,
                              cl_image2d_row_pitch,
                              cl_image2d_slice_pitch,
                              IoDirection::DtoH);
  default_convertor.ImageToNCHW(
      out_img_v.data(), out_v.data(), out_img_shape, out_dim);

#ifdef PRINT_RESULT  // top10
  for (int i = 0; i < 10; i++) {
    LOG(INFO) << "x_v[" << i << "]:" << x_v[i] << "\tout_v[" << i
              << "]:" << out_v[i];
  }
  for (int i = 0; i < 10; i++) {
    LOG(INFO) << "x_img_v[" << i << "]:" << x_img_v[i] << "\tout_img_v[" << i
              << "]:" << out_img_v.data()[i];
  }
#endif

  // compute cpu reference
  std::unique_ptr<float[]> out_ref(new float[out_dim.production()]);
  elementwise_compute_ref<float>(
      x_data, y_data, out_ref.get(), x_dim, y_dim, elemulParam.axis, "mul");

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

}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(elementwise_mul, kOpenCL, kFloat, kImageDefault, def);
