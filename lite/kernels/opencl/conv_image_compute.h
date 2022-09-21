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

#pragma once

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "lite/backends/opencl/cl_half.h"
#include "lite/backends/opencl/cl_include.h"
#include "lite/core/kernel.h"
#include "lite/core/tensor.h"
#include "lite/kernels/opencl/image_helper.h"
#include "lite/operators/op_params.h"
#ifdef LITE_WITH_PROFILE
#include "lite/core/profile/profiler.h"
#endif
#include "lite/backends/opencl/cl_utility.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

class ConvImageCompute : public KernelLite<TARGET(kOpenCL),
                                           PRECISION(kFP16),
                                           DATALAYOUT(kImageDefault)> {
 public:
  using param_t = operators::ConvParam;
  using kernel_t = void (ConvImageCompute::*)();

  void PrepareForRun() override;
  void ReInitWhenNeeded() override;
  void Run() override;

#ifdef LITE_WITH_PROFILE
  void SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = kernel_func_names_[0];
    ch->global_work_size = ch->NDRangeToStr(global_work_size_);
    ch->local_work_size = ch->NDRangeToStr(local_work_size_);
    ch->cl_event =
        event_;  // `event_` defined in `kernel.h`, valid after kernel::Run
  }
#endif

 private:
  void PrintConvInfo();
  void SetGlobalWorkSize();
  void SetLocalWorkSize(size_t repeats = 4);
  std::string GenerateTunedKey();
  void Conv2d1x1opt();
  void Conv2d3x3();
  void Conv2dnxnopt();
  void Conv2dnxn();
  void DepthwiseConv2d3x3s1();
  void DepthwiseConv2d3x3();
  void DepthwiseConv2d();
  void Conv2dCommon();
  void Conv2dCommonMulGroup();
  void Conv2d1x1Mali();
  void Conv2d1x1FC();
  void OIHW2OHWIO4I4(
      void* src, void* dst, size_t O, size_t I, size_t H, size_t W);
  // Change the travelsal order of the weight matrix in the following way:
  // The matrix is segmented to blocks of 4x4. If (any) dimension of the matrix
  // size is not divisible by 4, then pad with zeros. Each block is stored
  // contigously. The 16 elements within a block are ordered as 4 elements of
  // the first column, 4 elems of the second, etc. Blocks then traversed as
  // columns first, rows last. As an example, an 8x8 matrix would be traversed
  // as below.
  //
  //  |  0  4  8 12 32 36 40 44 |
  //  |  1  5  9 13 33 37 41 45 |
  //  |  2  6 10 14 34 38 42 46 |
  //  |  3  7 11 15 35 39 43 47 |
  //  | 16 20 24 28 48 52 56 60 |
  //  | 17 21 25 29 49 53 57 61 |
  //  | 18 22 26 30 50 54 58 62 |
  //  | 19 23 27 31 51 55 59 63 |
  //
  // The benefit of doing this is that reading contigous 16 elements gives a 4x4
  // block of the matrix, where the first 4 elements is the first row of the
  // block, second 4 elements is the second row of the block, etc. Subsequent
  // blocks contain elements of the same 4 columns.
  void OI2IOO4I4(void* src, void* dst, size_t O, size_t I);
  void NCHW2IMG4(void* src, void* dst, size_t oc, size_t ic, size_t index);
  void AssignDataFromCPUToGPU(const Tensor* tensor_cpu_p, Tensor* tensor_gpu_p);
  bool UseFcReplaceConv();

  param_t* conv_param_{nullptr};

  kernel_t impl_;
  std::vector<std::string> kernel_func_names_{};
  std::vector<std::string> kernel_func_paths_{};
  std::vector<std::string> build_options_{};
  std::string time_stamp_{GetTimeStamp()};

  std::unique_ptr<Tensor> filter_gpu_image_{nullptr};
  std::unique_ptr<Tensor> filter_gpu_image0_{nullptr};
  std::unique_ptr<Tensor> filter_gpu_image1_{nullptr};
  std::unique_ptr<Tensor> filter_gpu_image2_{nullptr};
  std::unique_ptr<Tensor> filter_gpu_image3_{nullptr};
  std::unique_ptr<Tensor> bias_gpu_image_{nullptr};
  std::unique_ptr<Tensor> alpha_gpu_image_{nullptr};
  std::unique_ptr<Tensor> tensor_hold_filter_image_{nullptr};
  std::unique_ptr<Tensor> tensor_hold_bias_image_{nullptr};
  std::unique_ptr<Tensor> filter_gpu_buffer_{nullptr};
  std::unique_ptr<Tensor> bias_gpu_buffer_{nullptr};
  std::unique_ptr<Tensor> wino_m_gpu_image_{nullptr};
  std::unique_ptr<Tensor> wino_v_gpu_image_{nullptr};
  std::unique_ptr<Tensor> mulgroups_in_fill_gpu_image_{nullptr};
  std::unique_ptr<Tensor> mulgroups_out_cut_gpu_image_{nullptr};

  cl::NDRange global_work_size_ = cl::NDRange{
      static_cast<size_t>(1), static_cast<size_t>(1), static_cast<size_t>(1)};
  cl::NDRange global_work_size_wino1_ =
      cl::NDRange{static_cast<size_t>(1), static_cast<size_t>(1)};
  cl::NDRange global_work_size_wino2_ =
      cl::NDRange{static_cast<size_t>(1), static_cast<size_t>(1)};
  cl::NDRange global_work_size_fill0_ =
      cl::NDRange{static_cast<size_t>(1), static_cast<size_t>(1)};
  cl::NDRange global_work_size_cut0_ =
      cl::NDRange{static_cast<size_t>(1), static_cast<size_t>(1)};
  // opencl kernel args
  int c_blk_ = 1;
  int w_blk_ = 1;
  int nh_blk_ = 1;

  const cl::Image2D* input_image_p_{nullptr};
  const cl::Image2D* second_input_image_p_{nullptr};
  const cl::Image2D* filter_image_p_{nullptr};
  const cl::Image2D* filter_image_p0_{nullptr};
  const cl::Image2D* filter_image_p1_{nullptr};
  const cl::Image2D* filter_image_p2_{nullptr};
  const cl::Image2D* filter_image_p3_{nullptr};
  const cl::Image2D* bias_image_p_{nullptr};
  const cl::Image2D* alpha_image_p_{nullptr};
  const cl::Image2D* output_image_p_{nullptr};
  const cl::Image2D* wino_v_image_p_{nullptr};
  const cl::Image2D* wino_m_image_p_{nullptr};
  const cl::Image2D* mulgroups_input_fill0_image_p_{nullptr};
  const cl::Image2D* mulgroups_output_cut0_image_p_{nullptr};

  std::unique_ptr<Tensor> w_gpu_t_{nullptr};
  std::unique_ptr<Tensor> bias_gpu_t_{nullptr};

  const cl::Buffer* filter_buffer_p_{nullptr};
  const cl::Buffer* bias_buffer_p_{nullptr};

  int stride_h_{-1};
  int stride_w_{-1};

  int dilation_h_{-1};
  int dilation_w_{-1};

  int pad_up_{-1};
  int pad_down_{-1};
  int pad_left_{-1};
  int pad_right_{-1};

  int offset_{-1};
  int offset_w_{-1};
  int offset_h_{-1};
  int groups_{-1};
  std::string fuse_eltwise_op_type_;
  bool relu_fused_{false};
  bool sigmoid_fused_{false};
  bool has_bias_{false};
  bool is_mali_{false};
  bool is_wino_{false};
  bool is_conv_mulgroup_{false};

  int input_tensor_n_{-1};
  int input_tensor_c_{-1};
  int input_tensor_h_{-1};
  int input_tensor_w_{-1};
  int input_image_h_{-1};
  int input_image_w_{-1};
  int input_c_block_{-1};

  int output_tensor_n_{-1};
  int output_tensor_c_{-1};
  int output_tensor_h_{-1};
  int output_tensor_w_{-1};
  int output_image_h_{-1};
  int output_image_w_{-1};

  int filter_tensor_n_{-1};
  int filter_tensor_c_{-1};
  int filter_tensor_h_{-1};
  int filter_tensor_w_{-1};
  int filter_image_h_{-1};
  int filter_image_w_{-1};

  int bias_image_h_{-1};
  int bias_image_w_{-1};

  int default_c_blk_ = 1;
  int default_w_blk_ = 1;
  int default_nh_blk_ = 1;
  // =================

  DDim last_input_dims_{};
  bool is_first_epoch_for_run_{true};

  cl::Kernel kernel_;
  cl::Kernel kernel_inner_product_;
  cl::Kernel kernel_output_trans_;
  cl::Kernel kernel_input_fill0_;
  cl::Kernel kernel_output_cut0_;
  cl_int status_;
  cl::NDRange local_work_size_ = cl::NDRange{
      static_cast<size_t>(1), static_cast<size_t>(1), static_cast<size_t>(1)};
  cl::NDRange local_work_size_wino1_ = cl::NDRange{
      static_cast<size_t>(1), static_cast<size_t>(1), static_cast<size_t>(1)};
  cl::NDRange local_work_size_wino2_ = cl::NDRange{
      static_cast<size_t>(1), static_cast<size_t>(1), static_cast<size_t>(1)};
  cl::NDRange local_work_size_fill0_ = cl::NDRange{
      static_cast<size_t>(1), static_cast<size_t>(1), static_cast<size_t>(1)};
  cl::NDRange local_work_size_cut0_ = cl::NDRange{
      static_cast<size_t>(1), static_cast<size_t>(1), static_cast<size_t>(1)};
  bool use_lws_{true};
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
