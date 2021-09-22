// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include <ctime>
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
  } else if (set_value == 250) {  // index
    for (size_t idx = 0; idx < length; ++idx) {
      x[idx] = length - 1 - idx;
    }
  } else if (set_value != -1) {
    for (size_t idx = 0; idx < length; ++idx) {
      x[idx] = set_value;
    }
  }
}
template <typename dtype>
void gather_compute_ref(const dtype *x_data,
                        const dtype *index_data,
                        int axis,
                        dtype *out_data,
                        const DDim &x_dims,
                        const DDim &index_dims) {
  if (axis == 0) {
    for (int i = 0; i < index_dims[0]; i++) {
      for (int j = 0; j < x_dims[1]; j++) {
        int idx = index_data[i] * x_dims[1] + j;
        out_data[i * x_dims[1] + j] = x_data[idx];
      }
    }
  } else if (axis == 1) {
    for (int i = 0; i < x_dims[0]; i++) {
      for (int j = 0; j < index_dims[0]; j++) {
        int idx = i * x_dims[1] + index_data[j];
        out_data[i * index_dims[0] + j] = x_data[idx];
      }
    }
  } else {
    LOG(FATAL) << "axis Out of support！ axis: " << axis;
    CHECK_EQ(1, 0);
  }
}
// #define PRINT_RESULT
// image
TEST(gather_image, compute) {
  // dims
  const int n = 1;
  const int c = 1;
  const int h = 7;
  const int w = 3;

  const DDim x_dim = DDim(std::vector<DDim::value_type>{h, w});

  std::vector<DDim> index_dim_v{DDim(std::vector<DDim::value_type>{2}),
                                DDim(std::vector<DDim::value_type>{h})};
  const DDim axis_dim = DDim(std::vector<DDim::value_type>{1});

  std::vector<float> axis_map{1, 0, 1, 0, 0};

  // start loop
  double time_mi = 1000000000000, time_mx = 0;
  for (size_t case_idx = 0; case_idx < 5; ++case_idx) {
    auto index_dim = index_dim_v[case_idx & 1];

    auto axis_m = axis_map[case_idx];
    DDim out_dim;
    if (axis_m == 0)
      out_dim = DDim(std::vector<DDim::value_type>{index_dim[0], w});
    else
      out_dim = DDim(std::vector<DDim::value_type>{h, index_dim[0]});

    // tensor

    lite::Tensor ga_x, ga_index, ga_out, ga_axis;
    ga_x.Resize(x_dim);
    ga_index.Resize(index_dim);
    ga_out.Resize(out_dim);
    ga_axis.Resize(axis_dim);
    // initialize tensors

    paddle::lite::CLImageConverterDefault default_convertor;
    // x
    std::vector<float> x_v(x_dim.production());

    fill_data<float>(x_v.data(), x_v.size());  // fill with index value
    auto x_img_shape = default_convertor.InitImageDimInfoWith(x_dim);  // w, h
    auto x_img_w = x_img_shape[0];
    auto x_img_h = x_img_shape[1];
    std::vector<half_t> x_img_v(x_img_w * x_img_h * 4);  // 4: RGBA
    default_convertor.NCHWToImage(x_v.data(), x_img_v.data(), x_dim);

    ga_x.mutable_data<half_t, cl::Image2D>(x_img_w, x_img_h, x_img_v.data());

    // index
    std::vector<float> index_v(index_dim.production());
    fill_data<float>(
        index_v.data(), index_v.size(), 250);  // fill with index value
    // index_v[0]=1;
    auto index_img_shape =
        default_convertor.InitImageDimInfoWith(index_dim);  // w, h
    auto index_img_w = index_img_shape[0];
    auto index_img_h = index_img_shape[1];
    std::vector<half_t> index_img_v(index_img_w * index_img_h * 4);  // 4: RGBA
    default_convertor.NCHWToImage(
        index_v.data(), index_img_v.data(), index_dim);
    ga_index.mutable_data<half_t, cl::Image2D>(
        index_img_w, index_img_h, index_img_v.data());

    // axis
    std::vector<float> axis_v(axis_dim.production());
    axis_v[0] = axis_m;
    auto axis_img_shape =
        default_convertor.InitImageDimInfoWith(axis_dim);  // w, h
    auto axis_img_w = axis_img_shape[0];
    auto axis_img_h = axis_img_shape[1];
    std::vector<half_t> axis_img_v(axis_img_w * axis_img_h * 4);  // 4: RGBA
    default_convertor.NCHWToImage(axis_v.data(), axis_img_v.data(), axis_dim);
    ga_axis.mutable_data<half_t, cl::Image2D>(
        axis_img_w, axis_img_h, axis_img_v.data());
    VLOG(1) << "axis   " << axis_v[0];

    // out
    auto out_img_shape =
        default_convertor.InitImageDimInfoWith(out_dim);  // w, h
    auto out_img_w = out_img_shape[0];
    auto out_img_h = out_img_shape[1];
    ga_out.mutable_data<half_t, cl::Image2D>(out_img_w, out_img_h);

    std::vector<half_t> out_img_v(out_img_w * out_img_h * 4);
    fill_data<half_t>(
        out_img_v.data(), out_img_v.size(), 0);  // fill with zero value

    std::vector<float> out_v(out_dim.production());

    operators::GatherParam gaParam;
    gaParam.X = &ga_x;
    gaParam.Index = &ga_index;
    gaParam.Out = &ga_out;
    gaParam.Axis = &ga_axis;

    auto op_param = gaParam;

    // set kernel
    auto ga_img_kernels = KernelRegistry::Global().Create(
        "gather", TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kImageDefault));
    ASSERT_FALSE(ga_img_kernels.empty());

    auto ga_img_kernel = std::move(ga_img_kernels.front());
    VLOG(4) << "get gather kernel: " << ga_img_kernel->doc();

    // set context and kernel args
    VLOG(4) << "set context and kernel args";
    std::unique_ptr<KernelContext> context(new KernelContext);
    context->As<OpenCLContext>().InitOnce();

    ga_img_kernel->SetParam(op_param);
    std::unique_ptr<KernelContext> ga_img_context(new KernelContext);
    context->As<OpenCLContext>().CopySharedTo(
        &(ga_img_context->As<OpenCLContext>()));
    ga_img_kernel->SetContext(std::move(ga_img_context));

    // run kernel
    VLOG(4) << "run kernel";

    clock_t start, finish;
    start = clock();
    ga_img_kernel->Launch();
    // download gpu result to cpu
    const size_t cl_image2d_row_pitch{0};
    const size_t cl_image2d_slice_pitch{0};

    CLRuntime::Global()->command_queue().finish();
    finish = clock();
    double time_ = static_cast<double>(finish - start) / CLOCKS_PER_SEC;
    std::cout << "run id：" << case_idx << std::endl;
    std::cout << "run time：    " << time_ << std::endl;

    if (time_ < time_mi) time_mi = time_;
    if (time_ > time_mx) time_mx = time_;

    TargetWrapperCL::ImgcpySync(out_img_v.data(),
                                ga_out.data<half_t, cl::Image2D>(),
                                out_img_w,
                                out_img_h,
                                cl_image2d_row_pitch,
                                cl_image2d_slice_pitch,
                                IoDirection::DtoH);
    start = clock();
    default_convertor.ImageToNCHW(
        out_img_v.data(), out_v.data(), out_img_shape, out_dim);

    std::unique_ptr<float[]> out_ref(new float[out_dim.production()]);
    gather_compute_ref<float>(
        x_v.data(), index_v.data(), axis_v[0], out_ref.get(), x_dim, index_dim);
    int bug = 0;
    for (int eidx = 0; eidx < out_dim[0]; eidx++) {
      for (int j = 0; j < out_dim[1]; j++) {
        auto value = out_v[eidx * out_dim[1] + j];
        if (value != out_ref[eidx * out_dim[1] + j]) bug++;
      }
    }
    if (bug > 1) {
      VLOG(1) << "cpu Result and GPU result verification error，bug：" << bug;
      CHECK_EQ(bug, 0);
    } else {
      std::cout << "The CPU result and GPU result are verified to be correct\n"
                << std::endl;
    }
  }
  std::cout << " Maximum time：" << time_mx << "秒\nMinimum time： " << time_mi
            << "秒" << std::endl;
}

}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(gather, kOpenCL, kFP16, kImageDefault, def);
