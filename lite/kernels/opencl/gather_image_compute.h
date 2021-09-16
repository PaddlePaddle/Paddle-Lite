#ifndef PADDLE_PADDLE_LITE_LITE_KERNELS_OPENCL_GATHER_IMAGE_COMPUTE_H
#define PADDLE_PADDLE_LITE_LITE_KERNELS_OPENCL_GATHER_IMAGE_COMPUTE_H

#endif //PADDLE_PADDLE_LITE_LITE_KERNELS_OPENCL_GATHER_IMAGE_COMPUTE_H
#pragma once

#include <memory>
#include <string>
#include <vector>
#include "lite/backends/opencl/cl_half.h"
#include "lite/core/kernel.h"
#include "lite/kernels/opencl/image_helper.h"
#include "lite/operators/op_params.h"
//#include "lite/utils/cp_logging.h"
#include "lite/utils/log/cp_logging.h"
#ifdef LITE_WITH_PROFILE
#include "lite/core/profile/profiler.h"
#endif
#include "lite/backends/opencl/cl_utility.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

class GatherImageCompute
    : public KernelLite<TARGET(kOpenCL),
                        PRECISION(kFP16),
                        DATALAYOUT(kImageDefault)> {
 public:
  using param_t = operators::GatherParam;

  void PrepareForRun() override;

  void ReInitWhenNeeded() override;

  void GetGlobalWorkSize();

  void Run() override;

#ifdef LITE_WITH_PROFILE
  void SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = kernel_func_name_;
    ch->cl_event =
        event_;  // `event_` defined in `kernel.h`, valid after kernel::Run
  }
#endif

  std::string doc() const override {
    return "ElementwiseAdd using cl::Image2D, kFP16";
  }

 protected:
  param_t* ga_param_{nullptr};
  DDim last_x_dims_;
  DDim x_img_shape_ = DDim(std::vector<DDim::value_type>(
      {static_cast<DDim::value_type>(1), static_cast<DDim::value_type>(1)}));
  DDim index_img_shape_ = DDim(std::vector<DDim::value_type>(
      {static_cast<DDim::value_type>(1), static_cast<DDim::value_type>(1)}));
  DDim out_img_shape_ = DDim(std::vector<DDim::value_type>(
      {static_cast<DDim::value_type>(1), static_cast<DDim::value_type>(1)}));
  int axis_=0;
  bool axis_change=false;
  std::string kernel_func_name_{"gather"};
  std::string build_options_{""};
  std::string time_stamp_{GetTimeStamp()};
  bool first_epoch_for_reinit_{true};
  cl::Kernel kernel_;
  cl::NDRange global_work_size_ = cl::NDRange{
      static_cast<size_t>(1), static_cast<size_t>(1), static_cast<size_t>(1)};
/*  std::unique_ptr<Tensor> y_weights_image_{
      nullptr}; */ // when param->Y from model weights
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle