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

#include <iostream>
#include "lite/core/op_registry.h"
#include "lite/backends/opencl/cl_include.h"
#include "lite/backends/opencl/cl_runtime.h"
#include "lite/backends/opencl/cl_utility.h"
#include "lite/kernels/opencl/image_helper.h"
#ifdef LITE_WITH_PROFILE
#include "lite/core/profile/profiler.h"
#endif

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

class GatherBufferCompute : public KernelLite<TARGET(kOpenCL),
                                              PRECISION(kFloat),
                                              DATALAYOUT(kNCHW)> {
public:
  using param_t = operators::GatherParam;

  std::string doc() const override { return "Gather using cl::Buffer, kFloat"; }

  void PrepareForRun() override {
    kernel_func_paths_.push_back("buffer/gather_kernel.cl");
    gather_param_ = param_.get_mutable<param_t>();
  }

  void ReInitWhenNeeded() override {
    const auto x_dims = gather_param_->X->dims();
    if ((!first_epoch_for_reinit_ && x_dims != last_x_dims_) ||
        first_epoch_for_reinit_) {
      last_x_dims_ = x_dims;
      first_epoch_for_reinit_ = false;
    }
  }

  void Run() override { 
    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);
    cl_int status;

    auto* x_buf = gather_param_->X->template data<float, cl::Buffer>();
    auto* index_buf = gather_param_->Index->template data<float, cl::Buffer>();
    auto* out_buf = gather_param_->Out->template mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
    auto index_dims = gather_param_->Index->dims();
    auto x_dims = gather_param_->X->dims();
    
    if (gather_param_->Axis != nullptr) {
        // with axis!
        VLOG(4) << "gather_param_->Axis is not nullptr!!!";
        auto* axis_data = gather_param_->Axis->data<int32_t>();
        int axis_index = axis_data[0];
        int input_size = x_dims.production();
        int index_size = index_dims.production();
        int inner_dim_size = 1;
        int outer_dim_size = 1;
        for (int i = 0; i < axis_index; i++) {
          inner_dim_size *= x_dims[i];
        }
        for (int i = axis_index + 1; i < x_dims.size(); i++) {
          outer_dim_size *= x_dims[i];
        }
        global_work_size_ =
            cl::NDRange{static_cast<cl::size_type>(inner_dim_size),
                        static_cast<cl::size_type>(index_size),
                        static_cast<cl::size_type>(outer_dim_size)};
        kernel_func_names_.push_back("gather_with_axis");
        context.cl_context()->AddKernel(kernel_func_names_[0],
                                        kernel_func_paths_[0],
                                        build_options_,
                                        time_stamp_);
        STL::stringstream kernel_key;
        kernel_key.str("");
        kernel_key << kernel_func_names_[0] << build_options_ << time_stamp_;
        auto kernel = context.cl_context()->GetKernel(kernel_key.str());

        status = kernel.setArg(0, *x_buf);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(1, *out_buf);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(2, *index_buf);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(3, input_size);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(4, index_size);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(5, inner_dim_size);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(6, outer_dim_size);
        CL_CHECK_FATAL(status);
        VLOG(4) << "input_size: " << input_size;
        VLOG(4) << "index_size: " << index_size;
        VLOG(4) << "inner_dim_size: " << inner_dim_size;
        VLOG(4) << "outer_dim_size: " << outer_dim_size;
        status = EnqueueNDRangeKernel(context,
                                        kernel,
                                        cl::NullRange,
                                        global_work_size_,
                                        cl::NullRange,
                                        nullptr,
                                        event_);
        CL_CHECK_FATAL(status);
    } else {
        // no axis!
        VLOG(4) << "gather_param_->Axis is nullptr!!!";
        int batch_size = index_dims[0];
        int slice_size = 1;
        for (size_t i = 1; i < x_dims.size(); ++i) {
            slice_size *= x_dims[i];
        }
        if (x_dims.size() == 4) { // n,c,h,w
            kernel_func_names_.push_back("gather_without_axis_nchw");
            context.cl_context()->AddKernel(kernel_func_names_[0],
                                            kernel_func_paths_[0],
                                            build_options_,
                                            time_stamp_);
            STL::stringstream kernel_key;
            kernel_key.str("");
            kernel_key << kernel_func_names_[0] << build_options_ << time_stamp_;
            auto kernel = context.cl_context()->GetKernel(kernel_key.str());
            global_work_size_ =
                cl::NDRange{static_cast<cl::size_type>(x_dims[2]),
                            static_cast<cl::size_type>(x_dims[3]),
                            static_cast<cl::size_type>(x_dims[1])};

            status = kernel.setArg(0, *x_buf);
            CL_CHECK_FATAL(status);
            status = kernel.setArg(1, *out_buf);
            CL_CHECK_FATAL(status);
            status = kernel.setArg(2, *index_buf);
            CL_CHECK_FATAL(status);
            status = kernel.setArg(3, slice_size);
            CL_CHECK_FATAL(status);
            status = kernel.setArg(4, batch_size);
            CL_CHECK_FATAL(status);
            status = kernel.setArg(5, (int)x_dims[2]); // h 
            CL_CHECK_FATAL(status);
            status = kernel.setArg(6, (int)x_dims[3]); // w
            CL_CHECK_FATAL(status);

            status = EnqueueNDRangeKernel(context,
                                            kernel,
                                            cl::NullRange,
                                            global_work_size_,
                                            cl::NullRange,
                                            nullptr,
                                            event_);
            CL_CHECK_FATAL(status);
        } else if (x_dims.size() == 3) { // c,h,w
            kernel_func_names_.push_back("gather_without_axis_chw");
            context.cl_context()->AddKernel(kernel_func_names_[0],
                                            kernel_func_paths_[0],
                                            build_options_,
                                            time_stamp_);
            STL::stringstream kernel_key;
            kernel_key.str("");
            kernel_key << kernel_func_names_[0] << build_options_ << time_stamp_;
            auto kernel = context.cl_context()->GetKernel(kernel_key.str());
            global_work_size_ =
                cl::NDRange{static_cast<cl::size_type>(x_dims[1]),
                            static_cast<cl::size_type>(x_dims[2]),
                            static_cast<cl::size_type>(batch_size)};

            status = kernel.setArg(0, *x_buf);
            CL_CHECK_FATAL(status);
            status = kernel.setArg(1, *out_buf);
            CL_CHECK_FATAL(status);
            status = kernel.setArg(2, *index_buf);
            CL_CHECK_FATAL(status);
            status = kernel.setArg(3, slice_size);
            CL_CHECK_FATAL(status);
            status = kernel.setArg(4, batch_size);
            CL_CHECK_FATAL(status);
            status = kernel.setArg(5, (int)x_dims[1]); // h 
            CL_CHECK_FATAL(status);
            status = kernel.setArg(6, (int)x_dims[2]); // w
            CL_CHECK_FATAL(status);

            status = EnqueueNDRangeKernel(context,
                                            kernel,
                                            cl::NullRange,
                                            global_work_size_,
                                            cl::NullRange,
                                            nullptr,
                                            event_);
            CL_CHECK_FATAL(status);
        } else if (x_dims.size() == 2) { // h,w
            kernel_func_names_.push_back("gather_without_axis_hw");
            context.cl_context()->AddKernel(kernel_func_names_[0],
                                            kernel_func_paths_[0],
                                            build_options_,
                                            time_stamp_);
            STL::stringstream kernel_key;
            kernel_key.str("");
            kernel_key << kernel_func_names_[0] << build_options_ << time_stamp_;
            auto kernel = context.cl_context()->GetKernel(kernel_key.str());
            global_work_size_ =
                cl::NDRange{static_cast<cl::size_type>(batch_size),
                            static_cast<cl::size_type>(x_dims[1])};

            status = kernel.setArg(0, *x_buf);
            CL_CHECK_FATAL(status);
            status = kernel.setArg(1, *out_buf);
            CL_CHECK_FATAL(status);
            status = kernel.setArg(2, *index_buf);
            CL_CHECK_FATAL(status);
            status = kernel.setArg(3, batch_size);
            CL_CHECK_FATAL(status);
            status = kernel.setArg(4, (int)x_dims[1]); // w
            CL_CHECK_FATAL(status);

            status = EnqueueNDRangeKernel(context,
                                            kernel,
                                            cl::NullRange,
                                            global_work_size_,
                                            cl::NullRange,
                                            nullptr,
                                            event_);
            CL_CHECK_FATAL(status);
        }
    }
  }

#ifdef LITE_WITH_PROFILE
  void SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = kernel_func_names_[0];
    ch->cl_event =
        event_;  // `event_` defined in `kernel.h`, valid after kernel::Run
    ch->global_work_size = ch->NDRangeToStr(global_work_size_);
  }
#endif

protected:
  std::vector<std::string> kernel_func_names_{};
  std::vector<std::string> kernel_func_paths_{};
  std::string build_options_{"-DCL_DTYPE_float"};
  std::string time_stamp_{GetTimeStamp()};

  param_t* gather_param_{nullptr};
  bool first_epoch_for_reinit_{true};
  DDim last_x_dims_;
  cl::NDRange global_work_size_ = cl::NDRange{
        static_cast<size_t>(1), static_cast<size_t>(1), static_cast<size_t>(1)};
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

namespace ocl = paddle::lite::kernels::opencl;

REGISTER_LITE_KERNEL(gather,
                     kOpenCL,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::opencl::GatherBufferCompute,
                     def)
    .BindInput("X",
                {LiteType::GetTensorTy(TARGET(kOpenCL), PRECISION(kFloat) )})
    .BindInput("Index",
                {LiteType::GetTensorTy(TARGET(kOpenCL), PRECISION(kFloat))})
    .BindInput("Axis",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL), PRECISION(kFloat) )})
    .Finalize();
