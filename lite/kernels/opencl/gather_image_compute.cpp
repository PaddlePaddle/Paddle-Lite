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

#include "lite/kernels/opencl/gather_image_compute.h"
#include <memory>
#include "lite/backends/opencl/cl_include.h"
#include "lite/backends/opencl/cl_utility.h"
#include "lite/core/op_registry.h"
#include "lite/utils/replace_stl/stream.h"

#undef LITE_WITH_LOG

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

void GatherImageCompute::PrepareForRun() {
  // VLOG(1) << "走到了PrepareForRun()" ;
  if (param_.is_type<param_t>()) {
    //   VLOG(1) << "执行if  param_.is_type<param_t>()判断" ;
    ga_param_ = param_.get_mutable<param_t>();
    // VLOG(1) << "执行 ga_param_获取成功" ;

  } else {
    std::cout << "预处理出错，预处理需要第二线路" << std::endl;
  }
  // VLOG(1) << "acm  666666";
  if (ga_param_ == nullptr)
    VLOG(1) << "ga_param为空";
  else
    VLOG(1) << "ga_param不为空";
  if (ga_param_->X == nullptr)
    VLOG(1) << "ga_param_->X为空";
  else
    VLOG(1) << "ga_param_->X不为空";
  if (ga_param_->Axis == nullptr) {
    VLOG(1) << "ga_param_->Axis为空";
    if (axis_ != 0) axis_ = 0, axis_change = true;
    //  goto LOOP;
    kernel_func_name_ = "gather_axis0";
    //   VLOG(1) << "跳过读取，直接顺利执行6" ;
    auto& context = ctx_->As<OpenCLContext>();
    context.cl_context()->AddKernel(kernel_func_name_,
                                    "image/gather_kernel.cl",
                                    build_options_,
                                    time_stamp_);
    //  VLOG(1) << "跳过读取，直接顺利执行7" ;
    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
    kernel_ = context.cl_context()->GetKernel(kernel_key.str());
    //   VLOG(1) << "顺利执行完全PrepareForRun" ;
    return;
  } else {
    VLOG(1) << "ga_param_->Axis不为空";
  }
  if (ga_param_->Index == nullptr)
    VLOG(1) << "ga_param_->Index为空";
  else
    VLOG(1) << "ga_param_->Index不为空";

  auto* axis = ga_param_->Axis;

  CHECK_EQ(axis->dims().production(), 1);
  //  CHECK(axis->dims().production() == 1);  // 指定维度只能是一个元素

  auto axis_dims = axis->dims();

  auto* axis_data = axis->data<half_t, cl::Image2D>();
  std::vector<half_t> axis_image_v(axis_dims.production() * 4);
  std::vector<float> axis_v(axis_dims.production());

  TargetWrapperCL::ImgcpySync(axis_image_v.data(),
                              axis_data,
                              axis->dims().production(),
                              1,
                              {0},
                              {0},
                              IoDirection::DtoH);

  paddle::lite::CLImageConverterDefault default_convertor;
  DDim axis_image_shape = default_convertor.InitImageDimInfoWith(axis->dims());

  default_convertor.ImageToNCHW(
      axis_image_v.data(), axis_v.data(), axis_image_shape, axis_dims);

  CHECK(axis_v[0] <= 1 && axis_v[0] >= 0);  // 目前支持的axis区间在[0,1];
  // CHECK(1==0);
  //  for(int i=0;i<axis_dims.production();i++)
  //    std::cout<<"试试看，你猜是啥axis？     "<<(axis_v[i])<<std::endl;
  if (axis_v[0] != axis_) axis_ = axis_v[0], axis_change = true;
  // choose kernel

  if (axis_ == 0) {
    kernel_func_name_ = "gather_axis0";
  } else {
    kernel_func_name_ = "gather_axis1";
    //   std::cout<<"axis==1的内核代码还没写啊魂淡"<<"\n"<<std::endl;
    //  CHECK(1==0);
  }

  auto& context = ctx_->As<OpenCLContext>();
  context.cl_context()->AddKernel(
      kernel_func_name_, "image/gather_kernel.cl", build_options_, time_stamp_);
  STL::stringstream kernel_key;
  kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
  kernel_ = context.cl_context()->GetKernel(kernel_key.str());
  VLOG(1) << "顺利执行完全PrepareForRun";
}

void GatherImageCompute::ReInitWhenNeeded() {
  // VLOG(1) << "走到了ReInitWhenNeeded()" ;
  auto* x = ga_param_->X;
  CHECK_EQ(x->dims().size(), 2);
  // CHECK(x->dims().size() == 2);  // 保证输入是一个二维矩阵
  auto* index = ga_param_->Index;
  CHECK_EQ(index->dims().size(), 1);
  // CHECK(index->dims().size() == 1);  // 裁剪的一定是一个一维的

  auto x_dims = ga_param_->X->dims();
  if ((!first_epoch_for_reinit_ &&
       ((x_dims != last_x_dims_) || (axis_change))) ||
      first_epoch_for_reinit_) {
    last_x_dims_ = x_dims;
    axis_change = false;  // axis是否变化了
    first_epoch_for_reinit_ = false;
    // compute image shape
    paddle::lite::CLImageConverterDefault default_convertor;

    index_img_shape_ =
        default_convertor.InitImageDimInfoWith(index->dims());  // w, h
    x_img_shape_ = default_convertor.InitImageDimInfoWith(x->dims());
    // compute global work size
    GetGlobalWorkSize();
    //   std::cout<<"axis_      :"<<axis_<<std::endl;
  }
}

void GatherImageCompute::GetGlobalWorkSize() {
  // VLOG(1) << "走到了GetGlobalWorkSize()" ;
  if (axis_ == 0) {
    global_work_size_ =
        cl::NDRange{static_cast<cl::size_type>(index_img_shape_[0]),
                    static_cast<cl::size_type>(x_img_shape_[0])};
  } else {
    global_work_size_ =
        cl::NDRange{static_cast<cl::size_type>(index_img_shape_[0]),
                    static_cast<cl::size_type>(x_img_shape_[1])};
  }
  if (axis_ == 0) {
    VLOG(1) << "global_work_size:[2D]:" << index_img_shape_[0] << " "
            << x_img_shape_[0] << "\n";
  } else {
    VLOG(1) << "global_work_size:[2D]:" << index_img_shape_[0] << " "
            << x_img_shape_[1] << "\n";
  }
}

void GatherImageCompute::Run() {
  //  VLOG(1) << "走到了gather的Run()" ;
  auto* x = ga_param_->X;
  auto* index = ga_param_->Index;
  auto* out = ga_param_->Out;
  auto* axis = ga_param_->Axis;
  auto x_dims = x->dims();
  auto index_dims = index->dims();
  auto out_dims = out->dims();
  auto* x_img = GET_DATA_GPU(x);
  //  GET_DATA_GPU(index);
  auto* index_img = index->data<int, cl::Image2D>();

  auto* out_img =
      MUTABLE_DATA_GPU(out, out_img_shape_[0], out_img_shape_[1], nullptr);

// #define LITE_test_x_index_LOG
#ifdef LITE_test_x_index_LOG
  // 取出X和index数据检验是否正确。
  const size_t cl_image2d_row_pitch{0};
  const size_t cl_image2d_slice_pitch{0};
  CLImageConverterDefault* default_converter = new CLImageConverterDefault();
  DDim x_image_shape = default_converter->InitImageDimInfoWith(x_dims);
  DDim index_image_shape = default_converter->InitImageDimInfoWith(index_dims);
  std::vector<half_t> x_image_data(x_dims.production() * 4);
  std::vector<half_t> index_image_data(index_dims.production() * 4);
  std::vector<float> x_v(x_dims.production());
  std::vector<float> index_v(index_dims.production());
  TargetWrapperCL::ImgcpySync(x_image_data.data(),
                              x_img,
                              x_image_shape[0],
                              x_image_shape[1],
                              cl_image2d_row_pitch,
                              cl_image2d_slice_pitch,
                              IoDirection::DtoH);
  TargetWrapperCL::ImgcpySync(index_image_data.data(),
                              index_img,
                              index_image_shape[0],
                              index_image_shape[1],
                              cl_image2d_row_pitch,
                              cl_image2d_slice_pitch,
                              IoDirection::DtoH);
  default_converter->ImageToNCHW(
      x_image_data.data(), x_v.data(), x_image_shape, x_dims);
  default_converter->ImageToNCHW(
      index_image_data.data(), index_v.data(), index_image_shape, index_dims);

  VLOG(1) << "x_dims.production() :   " << x_dims.production();
  VLOG(1) << "x_dims  :";
  auto x_dims_data = x_dims.data();
  // for(int i=0; i<x_dims.size();i++)
  //   VLOG(1) << i<<":  "<<x_dims_data[i];
  VLOG(1) << "index_dims.production() :   " << index_dims.production();
  VLOG(1) << " index_dims.size()   " << index_dims.size();
  for (int i = 0; i < index_dims.production(); i++) {
    VLOG(1) << "index的第 " << i << "个值是 ：" << index_v[i];
  }
  VLOG(1) << "out_dims.production() :   " << out->dims().production();
  VLOG(1) << "out_dims :   " << x_dims.size();

  // 8191
  for (int i = 0; i < 10; i++) {
    VLOG(1) << "x的第 " << i << "个值是 ：" << x_v[i];
    if (i & 1) VLOG(1) << "h : " << (i >> 1);
  }
#endif

  cl_int status;
  auto kernel = kernel_;
  if (kernel_func_name_ == "gather_axis0") {  // axis=0
    //   std::cout << "x_W   :" << x_w<<"\n";
    status = kernel.setArg(0, *x_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(1, *index_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(2, *out_img);
    CL_CHECK_FATAL(status);
  } else if (kernel_func_name_ == "gather_axis1") {
    //   std::cout << "x_W   :" << x_w<<"\n";
    status = kernel.setArg(0, *x_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(1, *index_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(2, *out_img);
    CL_CHECK_FATAL(status);
    //  std::cout << "Unsupported kernel: " << kernel_func_name_<<std::endl;
    //  std::cout << "第二个kernel尚未完成 " <<std::endl;
    //  CHECK(false);
  } else {
    std::cout << "Unsupported kernel: " << kernel_func_name_ << std::endl;
    CHECK(false);
  }

  auto& context = ctx_->As<OpenCLContext>();
  CHECK(context.cl_context() != nullptr);

  status = EnqueueNDRangeKernel(context,
                                kernel,
                                cl::NullRange,
                                global_work_size_,
                                cl::NullRange,
                                nullptr,
                                event_);

#ifdef LITE_test_x_index_LOG
  CLRuntime::Global()->command_queue().finish();
  DDim out_image_shape = default_converter->InitImageDimInfoWith(out_dims);
  std::vector<half_t> out_image_data(out_dims.production() * 4);
  std::vector<float> out_v(out_dims.production());
  auto* out_p = GET_DATA_GPU(out);
  if (out_p == nullptr) VLOG(1) << "out_p== nullptr   ";
  TargetWrapperCL::ImgcpySync(out_image_data.data(),
                              out_img,
                              out_image_shape[0],
                              out_image_shape[1],
                              cl_image2d_row_pitch,
                              cl_image2d_slice_pitch,
                              IoDirection::DtoH);
  default_converter->ImageToNCHW(
      out_image_data.data(), out_v.data(), out_image_shape, out_dims);
  for (int i = 0; i < out_dims.production(); i++) {
    auto value = out_v[i];
    VLOG(1) << "out的第 " << i << "个值是 ：" << value;
    if (i & 1) VLOG(1) << "h : " << (i >> 1);
  }
#endif
  CL_CHECK_FATAL(status);
}

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

namespace ocl = paddle::lite::kernels::opencl;


REGISTER_LITE_KERNEL(
    gather, kOpenCL, kFP16, kImageDefault, ocl::GatherImageCompute, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("Index",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("Axis",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();
#define LITE_WITH_LOG
