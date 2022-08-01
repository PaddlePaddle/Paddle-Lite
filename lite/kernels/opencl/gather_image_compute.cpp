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
  if (param_.is_type<param_t>()) {
    ga_param_ = param_.get_mutable<param_t>();
  } else {
    std::cout << "Preprocessing error, preprocessing requires a second line"
              << std::endl;
  }

  if (ga_param_ == nullptr)
    VLOG(1) << "ga_param is null";
  else
    VLOG(1) << "ga_param is not null";
  if (ga_param_->X == nullptr)
    VLOG(1) << "ga_param_->X is null ";
  else
    VLOG(1) << "ga_param_->X is not null";
  if (ga_param_->Axis == nullptr) {
    VLOG(1) << "ga_param_->Axis is null";
    if (axis_ != 0) axis_ = 0, axis_change = true;

    kernel_func_name_ = "gather_axis0";

    auto& context = ctx_->As<OpenCLContext>();
    context.cl_context()->AddKernel(kernel_func_name_,
                                    "image/gather_kernel.cl",
                                    build_options_,
                                    time_stamp_);

    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
    kernel_ = context.cl_context()->GetKernel(kernel_key.str());

    return;
  } else {
    VLOG(1) << "ga_param_->Axis isnot null";
  }
  if (ga_param_->Index == nullptr)
    VLOG(1) << "ga_param_->Index is null";
  else
    VLOG(1) << "ga_param_->Index is not null";

  auto* axis = ga_param_->Axis;

  CHECK_EQ(axis->dims().production(), 1);

  auto axis_dims = axis->dims();

  auto* axis_v = axis->data<int>();

  CHECK(axis_v[0] <= 1 && axis_v[0] >= 0);

  if (axis_v[0] != axis_) axis_ = axis_v[0], axis_change = true;

  // choose kernel
  if (axis_ == 0) {
    kernel_func_name_ = "gather_axis0";
  } else {
    kernel_func_name_ = "gather_axis1";
  }

  auto& context = ctx_->As<OpenCLContext>();
  context.cl_context()->AddKernel(
      kernel_func_name_, "image/gather_kernel.cl", build_options_, time_stamp_);
  STL::stringstream kernel_key;
  kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
  kernel_ = context.cl_context()->GetKernel(kernel_key.str());
}

void GatherImageCompute::ReInitWhenNeeded() {
  auto* x = ga_param_->X;
  CHECK_EQ(x->dims().size(), 2);

  auto* index = ga_param_->Index;
  CHECK_EQ(index->dims().size(), 1);

  auto x_dims = ga_param_->X->dims();
  if ((!first_epoch_for_reinit_ &&
       ((x_dims != last_x_dims_) || (axis_change))) ||
      first_epoch_for_reinit_) {
    last_x_dims_ = x_dims;
    axis_change = false;
    first_epoch_for_reinit_ = false;

    paddle::lite::CLImageConverterDefault default_convertor;
    x_img_shape_ = default_convertor.InitImageDimInfoWith(x->dims());
    out_img_shape_ =
        default_convertor.InitImageDimInfoWith(ga_param_->Out->dims());

    auto index_t = ga_param_->Index;
    index_gpu_t_ = std::unique_ptr<Tensor>(new Tensor);
    auto index_gpu_data =
        index_gpu_t_->mutable_data(TARGET(kOpenCL), index_t->memory_size());
    TargetWrapperCL::MemcpySync(index_gpu_data,
                                index_t->raw_data(),
                                index_t->memory_size(),
                                IoDirection::HtoD);

    GetGlobalWorkSize();
  }
}

void GatherImageCompute::GetGlobalWorkSize() {
  // VLOG(1) << "走到了GetGlobalWorkSize()" ;
  if (axis_ == 0) {
    global_work_size_ =
        cl::NDRange{static_cast<cl::size_type>(ga_param_->Index->dims()[0]),
                    static_cast<cl::size_type>(x_img_shape_[0])};
  } else {
    global_work_size_ =
        cl::NDRange{static_cast<cl::size_type>(ga_param_->Index->dims()[0]),
                    static_cast<cl::size_type>(x_img_shape_[1])};
  }
  if (axis_ == 0) {
    VLOG(1) << "global_work_size:[2D]:" << ga_param_->Index->dims()[0] << " "
            << x_img_shape_[0] << "\n";
  } else {
    VLOG(1) << "global_work_size:[2D]:" << ga_param_->Index->dims()[0] << " "
            << x_img_shape_[1] << "\n";
  }
}

void GatherImageCompute::Run() {
  auto* x = ga_param_->X;
  auto* index = ga_param_->Index;
  auto* out = ga_param_->Out;
  auto* axis = ga_param_->Axis;
  auto x_dims = x->dims();
  auto index_dims = index->dims();
  auto out_dims = out->dims();
  auto* x_img = GET_DATA_GPU(x);
  //  GET_DATA_GPU(index);
  auto index_buf = index_gpu_t_->data<int, cl::Buffer>();

  auto* out_img =
      MUTABLE_DATA_GPU(out, out_img_shape_[0], out_img_shape_[1], nullptr);
  cl_int status;
  auto kernel = kernel_;
  if (kernel_func_name_ == "gather_axis0") {  // axis=0
    status = kernel.setArg(0, *x_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(1, *index_buf);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(2, *out_img);
    CL_CHECK_FATAL(status);
  } else if (kernel_func_name_ == "gather_axis1") {
    status = kernel.setArg(0, *x_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(1, *index_buf);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(2, *out_img);
    CL_CHECK_FATAL(status);
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
  CLImageConverterDefault default_converter;
  std::vector<float> out_image_data(out_img_shape_[0] * out_img_shape_[1] * 4);
  std::vector<float> out_v(out_dims.production());
  auto* out_p = GET_DATA_GPU(out);
  if (out_p == nullptr) VLOG(1) << "out_p== nullptr   ";
  TargetWrapperCL::ImgcpySync(out_image_data.data(),
                              out_img,
                              out_img_shape_[0],
                              out_img_shape_[1],
                              0,
                              0,
                              IoDirection::DtoH);
  default_converter.ImageToNCHW(
      out_image_data.data(), out_v.data(), out_img_shape_, out_dims);

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
    .BindInput("Index", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Axis", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();
#define LITE_WITH_LOG
