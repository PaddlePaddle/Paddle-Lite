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

#undef LITE_WITH_LOG
namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

class YoloBoxComputeBuffer
    : public KernelLite<TARGET(kOpenCL), PRECISION(kAny), DATALAYOUT(kNCHW)> {
 public:
  using param_t = operators::YoloBoxParam;

  std::string doc() const override { return "YoloBox using cl::Buffer, kAny"; }

  void PrepareForRun() override {
    yolo_box_param_ = param_.get_mutable<param_t>();

    // others
    class_num_ = yolo_box_param_->class_num;
    conf_thresh_ = yolo_box_param_->conf_thresh;
    clip_bbox_ = static_cast<int>(yolo_box_param_->clip_bbox);
    scale_x_y_ = yolo_box_param_->scale_x_y;
    bias_ = -0.5 * (scale_x_y_ - 1.);
    // X: input
    lite::Tensor* X = yolo_box_param_->X;
    x_n_ = X->dims()[0];
    x_c_ = X->dims()[1];
    x_h_ = X->dims()[2];
    x_w_ = X->dims()[3];
    x_stride_ = x_h_ * x_w_;

    // anchors
    std::vector<int> anchors = yolo_box_param_->anchors;
    anchor_num_ = anchors.size() / 2;
    anchor_stride_ = (class_num_ + 5) * x_stride_;  // x_stride_ init should be
    const DDim anchors_dim = DDim(
        std::vector<DDim::value_type>{static_cast<int64_t>(anchors.size())});
    anchors_gpu_t_ = std::unique_ptr<Tensor>(new Tensor);
    anchors_gpu_t_->Resize(anchors_dim);
    anchors_gpu_data_ =
        anchors_gpu_t_->mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
    TargetWrapperCL::MemcpySync(anchors_gpu_data_,
                                anchors.data(),
                                anchors_gpu_t_->memory_size(),
                                IoDirection::HtoD);

    // ImgSize: input
    lite::Tensor* ImgSize = yolo_box_param_->ImgSize;
    imgsize_gpu_t_ = std::unique_ptr<Tensor>(new Tensor);
    imgsize_gpu_t_->Resize(ImgSize->dims());
    imgsize_gpu_data_ =
        imgsize_gpu_t_->mutable_data<int, cl::Buffer>(TARGET(kOpenCL));
    TargetWrapperCL::MemcpySync(imgsize_gpu_data_,
                                ImgSize->data<int>(),
                                ImgSize->memory_size(),
                                IoDirection::HtoD);
  }

  void ReInitWhenNeeded() override {
    const auto x_dims = yolo_box_param_->X->dims();
    if ((!first_epoch_for_reinit_ && x_dims != last_x_dims_) ||
        first_epoch_for_reinit_) {
      last_x_dims_ = x_dims;
      first_epoch_for_reinit_ = false;

      // X: input
      lite::Tensor* X = yolo_box_param_->X;
      x_n_ = X->dims()[0];
      x_c_ = X->dims()[1];
      x_h_ = X->dims()[2];
      x_w_ = X->dims()[3];
      x_stride_ = x_h_ * x_w_;
      x_size_ = yolo_box_param_->downsample_ratio * x_h_;
      x_data_ = X->mutable_data<float, cl::Buffer>();

      // Boxes: output
      lite::Tensor* Boxes = yolo_box_param_->Boxes;
      boxes_data_ = Boxes->mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
      box_num_ = Boxes->dims()[1];

      // Scores: output
      lite::Tensor* Scores = yolo_box_param_->Scores;
      scores_data_ = Scores->mutable_data<float, cl::Buffer>(TARGET(kOpenCL));

      VLOG(1) << "kernel_func_name_:" << kernel_func_name_;
      auto& context = ctx_->As<OpenCLContext>();
      context.cl_context()->AddKernel(kernel_func_name_,
                                      "buffer/yolo_box_kernel.cl",
                                      build_options_,
                                      time_stamp_);
      STL::stringstream kernel_key;
      kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
      kernel_ = context.cl_context()->GetKernel(kernel_key.str());

      GetGlobalWorkSize();
    }
  }

  void GetGlobalWorkSize() {
    global_work_size_ = {static_cast<cl::size_type>(x_h_),
                         static_cast<cl::size_type>(x_w_),
                         static_cast<cl::size_type>(anchor_num_)};
  }

  void Run() override {
#ifdef LITE_WITH_LOG
    LOG(INFO) << "x_n_:" << x_n_;
    LOG(INFO) << "x_c_:" << x_c_;
    LOG(INFO) << "x_h_:" << x_h_;
    LOG(INFO) << "x_w_:" << x_w_;
    LOG(INFO) << "x_stride_:" << x_stride_;
    LOG(INFO) << "x_size_:" << x_size_;
    LOG(INFO) << "box_num_:" << box_num_;
    LOG(INFO) << "anchor_num_:" << anchor_num_;
    LOG(INFO) << "anchor_stride_:" << anchor_stride_;
    LOG(INFO) << "class_num_:" << class_num_;
    LOG(INFO) << "clip_bbox_:" << clip_bbox_;
    LOG(INFO) << "conf_thresh_:" << conf_thresh_;
    LOG(INFO) << "scale_x_y_:" << scale_x_y_;
    LOG(INFO) << "bias_:" << bias_;
#endif

    auto kernel = kernel_;
    cl_int status;
    status = kernel.setArg(0, *x_data_);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(1, x_n_);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(2, x_c_);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(3, x_h_);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(4, x_w_);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(5, x_stride_);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(6, x_size_);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(7, *imgsize_gpu_data_);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(8, /* imgsize_num = */ x_n_);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(9, *boxes_data_);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(10, box_num_);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(11, *scores_data_);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(12, *anchors_gpu_data_);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(13, anchor_num_);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(14, anchor_stride_);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(15, class_num_);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(16, clip_bbox_);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(17, conf_thresh_);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(18, scale_x_y_);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(19, bias_);
    CL_CHECK_FATAL(status);

    auto& context = ctx_->As<OpenCLContext>();
    status = EnqueueNDRangeKernel(context,
                                  kernel,
                                  cl::NullRange,
                                  global_work_size_,
                                  cl::NullRange,
                                  nullptr,
                                  event_);
    CL_CHECK_FATAL(status);
  }

#ifdef LITE_WITH_PROFILE
  void SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = kernel_func_name_;
    ch->cl_event =
        event_;  // `event_` defined in `kernel.h`, valid after kernel::Run
  }
#endif

 private:
  std::string kernel_func_name_{"yolo_box"};
  std::string build_options_{"-DCL_DTYPE_float"};
  std::string time_stamp_{GetTimeStamp()};
  bool first_epoch_for_reinit_{true};
  DDim last_x_dims_;

  param_t* yolo_box_param_{nullptr};
  std::unique_ptr<Tensor> imgsize_gpu_t_{nullptr};
  std::unique_ptr<Tensor> anchors_gpu_t_{nullptr};

  int class_num_{-1};
  float conf_thresh_{0.5};
  int clip_bbox_{0};
  float scale_x_y_{1};
  float bias_{0};

  int x_n_{-1};
  int x_c_{-1};
  int x_h_{-1};
  int x_w_{-1};
  int box_num_{-1};
  int anchor_num_{-1};
  int x_size_{-1};

  int x_stride_{-1};
  int anchor_stride_{-1};

  cl::Buffer* x_data_;
  cl::Buffer* imgsize_gpu_data_;
  cl::Buffer* boxes_data_;
  cl::Buffer* scores_data_;
  cl::Buffer* anchors_gpu_data_;

  cl::NDRange global_work_size_;
  cl::Kernel kernel_;
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(yolo_box,
                     kOpenCL,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::opencl::YoloBoxComputeBuffer,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kOpenCL), PRECISION(kAny))})
    .BindInput("ImgSize",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindOutput("Boxes", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .BindOutput("Scores", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .Finalize();
#define LITE_WITH_LOG
