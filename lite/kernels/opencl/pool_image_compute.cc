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

#include <vector>

#include "lite/backends/opencl/cl_half.h"
#include "lite/backends/opencl/cl_include.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/opencl/image_helper.h"
#include "lite/operators/op_params.h"
#include "lite/utils/replace_stl/stream.h"
#include "lite/utils/string.h"
#ifdef LITE_WITH_PROFILE
#include "lite/core/profile/profiler.h"
#endif
#include "lite/backends/opencl/cl_utility.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

class PoolComputeImage2D : public KernelLite<TARGET(kOpenCL),
                                             PRECISION(kFP16),
                                             DATALAYOUT(kImageDefault)> {
 public:
  using param_t = operators::PoolParam;

  std::string doc() const override { return "Pool using cl::Image2D, kFP16"; }

  void PrepareForRun() override {
    const auto& param = *param_.get_mutable<param_t>();
    const auto& in_dims = param.x->dims();
    const auto& out_dims = param.output->dims();
    const bool global_pooling = param.global_pooling;
    const bool exclusive = param.exclusive;
    std::vector<int> ksize = param.ksize;
    std::vector<int> paddings = *param.paddings;
    if (exclusive) {
      build_options_ += " -DEXCLUSIVE";
    }
    if (global_pooling) {
      build_options_ += " -DGLOBAL";
      for (size_t i = 0; i < ksize.size(); ++i) {
        paddings[2 * i] = 0;
        paddings[2 * i + 1] = 0;
        ksize[i] = static_cast<int>(in_dims[i + 2]);
      }
    }
    if (param.pooling_type == "avg") {
      build_options_ += " -DPOOL_AVG";
    }

    run_local_work_ =
        out_dims[0] * UP_DIV(out_dims[1], 4) * out_dims[2] * out_dims[3] <
            low_op_parallelism_thre_ &&
        ksize[0] * ksize[1] >= high_op_intensity_thre_;
    if (run_local_work_) {
      kernel_func_name_ += "_local";
    }

    VLOG(1) << "kernel_func_name_:" << kernel_func_name_;
    auto& context = ctx_->As<OpenCLContext>();
    context.cl_context()->AddKernel(
        kernel_func_name_, "image/pool_kernel.cl", build_options_, time_stamp_);

    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
    kernel_ = context.cl_context()->GetKernel(kernel_key.str());
  }

#ifdef LITE_WITH_PROFILE
  void SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = kernel_func_name_;
    ch->cl_event =
        event_;  // `event_` defined in `kernel.h`, valid after kernel::Run
  }
#endif

  void ReInitWhenNeeded() override {
    const auto& param = *param_.get_mutable<param_t>();
    auto& x_dims = param.x->dims();

    if ((!first_epoch_for_reinit_ && x_dims != last_x_dims_) ||
        first_epoch_for_reinit_) {
      last_x_dims_ = x_dims;
      first_epoch_for_reinit_ = false;
      const auto& out_dims = param.output->dims();

      x_img_ = DATA_GPU(param.x);
      auto out_image_shape = InitImageDimInfoWith(out_dims);
#ifdef LITE_WITH_LOG
      VLOG(4) << "out_image_shape = " << out_image_shape["width"] << " "
              << out_image_shape["height"];
#endif
      out_img_ = MUTABLE_DATA_GPU(param.output,
                                  out_image_shape["width"],
                                  out_image_shape["height"],
                                  nullptr);
      auto& context = ctx_->As<OpenCLContext>();
      CHECK(context.cl_context() != nullptr);

      const auto& in_dims = param.x->dims();
      const std::string pooling_type = param.pooling_type;
      const bool global_pooling = param.global_pooling;
      std::vector<int> paddings = *param.paddings;
      std::vector<int> strides = param.strides;
      std::vector<int> ksize = param.ksize;

#ifdef LITE_WITH_LOG
      VLOG(4) << "global_pooling: " << global_pooling;
      VLOG(4) << "pooling_type: " << pooling_type;
      VLOG(4) << "paddings : " << paddings[0] << "  " << paddings[1] << "  "
              << paddings[2] << "  " << paddings[3] << "  ";
#endif

      if (global_pooling) {
        for (size_t i = 0; i < ksize.size(); ++i) {
          paddings[2 * i] = 0;
          paddings[2 * i + 1] = 0;
          ksize[i] = static_cast<int>(in_dims[i + 2]);
        }
      }

#ifdef LITE_WITH_LOG
      VLOG(4) << "in_dims : [" << in_dims.size() << "]" << in_dims[0] << "  "
              << in_dims[1] << "  " << in_dims[2] << "  " << in_dims[3];
      VLOG(4) << "out_dims : [" << out_dims.size() << "]" << out_dims[0] << "  "
              << out_dims[1] << "  " << out_dims[2] << "  " << out_dims[3];
      VLOG(4) << "paddings fixed : " << paddings[0] << "  " << paddings[1]
              << "  " << paddings[2] << "  " << paddings[3] << "  ";
      VLOG(4) << "strides : [" << strides.size() << "]" << strides[0] << "  "
              << strides[1];
      VLOG(4) << "ksize : [" << ksize.size() << "]" << ksize[0] << "  "
              << ksize[1] << "  " << ksize[2] << "  " << ksize[3];
      VLOG(4) << "paddings : [" << paddings.size() << "]" << paddings[0] << "  "
              << paddings[1] << "  " << paddings[2] << "  " << paddings[3];
#endif

      bool pads_equal = ((abs(paddings[0] - paddings[1]) < 2) &&
                         (abs(paddings[2] - paddings[3]) < 2));
      if (!pads_equal) {
        LOG(FATAL)
            << "padding requires pad_left == pad_right, pad_top == pad_bottom";
      }

      const int out_c_blks = UP_DIV(out_dims[1], 4);
      uint32_t workgroup_size = 0;

      int type_size =
          (CLRuntime::Global()->get_precision() == lite_api::CL_PRECISION_FP16)
              ? sizeof(uint16_t)
              : sizeof(float);
      if (param.global_pooling && param.pooling_type == "avg") {
        type_size = sizeof(float);
      }
      uint32_t local_mem_size =
          CLRuntime::Global()->GetDeviceInfo()["CL_DEVICE_LOCAL_MEM_SIZE_KB"] *
          1024;
      uint32_t workgroupsize_max =
          CLRuntime::Global()->GetDeviceInfo()["CL_DEVICE_MAX_WORK_GROUP_SIZE"];

      uint32_t compute_intensity = ksize[0] * ksize[1];
      run_local_work_ = out_dims[0] * out_c_blks * out_dims[2] * out_dims[3] <
                            low_op_parallelism_thre_ &&
                        compute_intensity >= high_op_intensity_thre_;
      if (run_local_work_) {
        workgroup_size =
            std::min(static_cast<uint32_t>(local_mem_size / (4 * type_size)),
                     workgroupsize_max);
        workgroup_size =
            std::min(static_cast<uint32_t>(compute_intensity), workgroup_size);
        uint32_t temp_size = 1;
        while ((temp_size <<= 1) <= workgroup_size) {
        }
        workgroup_size = temp_size >> 1;

        int workgroup_w_size = 1, workgroup_h_size;
        while ((workgroup_w_size <<= 1) <= ksize[0] &&
               workgroup_w_size <= workgroup_size) {
        }
        workgroup_w_size >>= 1;
        workgroup_h_size = workgroup_size / workgroup_w_size;

        global_work_size_ =
            cl::NDRange{static_cast<cl::size_type>(out_c_blks * workgroup_size),
                        static_cast<cl::size_type>(out_dims[3]),
                        static_cast<cl::size_type>(out_dims[0] * out_dims[2])};
        local_work_size_ =
            cl::NDRange{static_cast<cl::size_type>(workgroup_size), 1, 1};
        LOG(INFO) << "GWS: " << global_work_size_[0] << ", "
                  << global_work_size_[1] << ", " << global_work_size_[2];
        LOG(INFO) << "LWS: " << local_work_size_[0] << ", "
                  << local_work_size_[1] << ", " << local_work_size_[2];

        int local_block_size_shape[2] = {workgroup_w_size, workgroup_h_size};
        int local_block_count_shape[2] = {UP_DIV(ksize[0], workgroup_w_size),
                                          UP_DIV(ksize[1], workgroup_h_size)};

        int idx = 12;
        kernel_.setArg(idx++, static_cast<int32_t>(workgroup_size));
        kernel_.setArg(
            idx++, sizeof(local_block_size_shape), local_block_size_shape);
        kernel_.setArg(
            idx++, sizeof(local_block_count_shape), local_block_count_shape);
        kernel_.setArg(idx++, workgroup_size * 4 * type_size, nullptr);
      } else {
        global_work_size_ =
            cl::NDRange(out_c_blks, out_dims[3], out_dims[0] * out_dims[2]);
        local_work_size_ = cl::NullRange;
      }

#ifdef LITE_WITH_LOG
      VLOG(4) << "global_work_size_ : [" << 3 << "]"
              << static_cast<int>(global_work_size_[0]) << "  "
              << static_cast<int>(global_work_size_[1]) << "  "
              << static_cast<int>(global_work_size_[2]) << "  ";
#endif
      cl_int status;
      int arg_idx = 2;
      status = kernel_.setArg(arg_idx, static_cast<const int>(in_dims[2]));
      CL_CHECK_FATAL(status);
      status = kernel_.setArg(++arg_idx, static_cast<const int>(in_dims[3]));
      CL_CHECK_FATAL(status);
      status = kernel_.setArg(++arg_idx, static_cast<const int>(out_dims[2]));
      CL_CHECK_FATAL(status);
      status = kernel_.setArg(++arg_idx, static_cast<const int>(out_dims[3]));
      CL_CHECK_FATAL(status);
      status = kernel_.setArg(++arg_idx, static_cast<const int>(ksize[0]));
      CL_CHECK_FATAL(status);
      status = kernel_.setArg(++arg_idx, static_cast<const int>(ksize[1]));
      CL_CHECK_FATAL(status);
      status = kernel_.setArg(++arg_idx, static_cast<const int>(strides[0]));
      CL_CHECK_FATAL(status);
      status = kernel_.setArg(++arg_idx, static_cast<const int>(strides[1]));
      CL_CHECK_FATAL(status);
      status = kernel_.setArg(++arg_idx, static_cast<const int>(paddings[2]));
      CL_CHECK_FATAL(status);
      status = kernel_.setArg(++arg_idx, static_cast<const int>(paddings[0]));
      CL_CHECK_FATAL(status);
    }
  }

  void Run() override {
    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);

    cl_int status;
    int arg_idx = 0;
    status = kernel_.setArg(arg_idx++, *x_img_);
    CL_CHECK_FATAL(status);
    status = kernel_.setArg(arg_idx++, *out_img_);
    CL_CHECK_FATAL(status);

    cl::Event myevent;
    status = EnqueueNDRangeKernel(context,
                                  kernel_,
                                  cl::NullRange,
                                  global_work_size_,
                                  local_work_size_,
                                  nullptr,
                                  myevent);
    float time = CLRuntime::Global()->GetCommandTime(myevent);
    LOG(INFO) << "POOL TIME: " << time;
    CL_CHECK_FATAL(status);
  }

 private:
  bool first_epoch_for_reinit_{true};
  DDim last_x_dims_;
  std::string kernel_func_name_{"pool"};
  std::string build_options_{""};
  std::string time_stamp_{GetTimeStamp()};
  cl::Kernel kernel_;
  cl::Image2D* x_img_{nullptr};
  cl::Image2D* out_img_{nullptr};
  cl::NDRange global_work_size_;
  cl::NDRange local_work_size_;
  bool run_local_work_{false};
  const uint32_t low_op_parallelism_thre_{256};
  const uint32_t high_op_intensity_thre_{128};
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(pool2d,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     paddle::lite::kernels::opencl::PoolComputeImage2D,
                     image2d)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();
