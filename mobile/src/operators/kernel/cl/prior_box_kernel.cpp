/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef PRIORBOX_OP

#include "operators/kernel/prior_box_kernel.h"
#include "framework/cl/cl_tensor.h"
namespace paddle_mobile {
namespace operators {

template <>
bool PriorBoxKernel<GPU_CL, float>::Init(PriorBoxParam<GPU_CL> *param) {
  this->cl_helper_.AddKernel("prior_box", "prior_box_kernel.cl");
  return true;
}

template <>
void PriorBoxKernel<GPU_CL, float>::Compute(
    const PriorBoxParam<GPU_CL> &param) {
  const auto *input_ = param.Input();
  const auto &input_dims = input_->dims();

  const auto &input_image_dims = param.InputImage()->dims();

  const auto &min_sizes = param.MinSizes();
  const auto &max_sizes = param.MaxSizes();
  const auto &variances = param.Variances();
  const auto &input_aspect_ratio = param.AspectRatios();
  const bool &flip = param.Flip();
  const bool &clip = param.Clip();
  int isclip = 0;
  if (clip) {
    isclip = 1;
  }
  const float &step_w = param.StepW();
  const float &step_h = param.StepH();
  const float &offset = param.Offset();
  const int C = param.OutputBoxes()->dims()[1];

  auto output_boxes = param.OutputBoxes()->GetCLImage();
  auto output_variances = param.OutputVariances()->GetCLImage();

  std::vector<float> aspect_ratios;
  ExpandAspectRatios(input_aspect_ratio, flip, &aspect_ratios);

  auto img_width = input_image_dims[3];
  auto img_height = input_image_dims[2];

  auto feature_width = input_dims[3];
  auto feature_height = input_dims[2];

  float step_width, step_height;
  /// 300 / 19
  if (step_w == 0 || step_h == 0) {
    step_width = static_cast<float>(img_width) / feature_width;
    step_height = static_cast<float>(img_height) / feature_height;
  } else {
    step_width = step_w;
    step_height = step_h;
  }

  int num_priors = aspect_ratios.size() * min_sizes.size();
  if (!max_sizes.empty()) {
    num_priors += max_sizes.size();
  }

  float *box_width = static_cast<float *>(
      paddle_mobile::memory::Alloc(sizeof(float) * num_priors));
  float *box_height = static_cast<float *>(
      paddle_mobile::memory::Alloc(sizeof(float) * num_priors));
  float *variancesptr =
      static_cast<float *>(paddle_mobile::memory::Alloc(sizeof(float) * 4));
  int idx = 0;
  for (size_t s = 0; s < min_sizes.size(); ++s) {
    auto min_size = min_sizes[s];
    if (param.MinMaxAspectRatiosOrder()) {
      box_width[idx] = box_height[idx] = min_size / 2.;
      idx++;
      if (max_sizes.size() > 0) {
        auto max_size = max_sizes[s];
        box_width[idx] = box_height[idx] = sqrt(min_size * max_size) / 2.;
        idx++;
      }
      for (float ar : aspect_ratios) {
        if (fabs(ar - 1.) < 1e-6) {
          continue;
        }
        box_width[idx] = min_size * sqrt(ar) / 2.;
        box_height[idx] = min_size / sqrt(ar) / 2.;
        idx++;
      }

    } else {
      for (float ar : aspect_ratios) {
        box_width[idx] = min_size * sqrt(ar) / 2.;
        box_height[idx] = min_size / sqrt(ar) / 2.;
        idx++;
      }
      if (!max_sizes.empty()) {
        auto max_size = max_sizes[s];
        box_width[idx] = box_height[idx] = sqrt(min_size * max_size) / 2.;
        idx++;
      }
    }
  }
  for (int i = 0; i < variances.size(); i++) {
    variancesptr[i] = variances[i];
  }
  cl_int status;
  auto kernel = this->cl_helper_.KernelAt(0);
  auto default_work_size =
      this->cl_helper_.DefaultWorkSize(*param.OutputBoxes());
  auto c_block = default_work_size[0];
  auto w = default_work_size[1];
  auto nh = default_work_size[2];

  std::vector<int64_t> box_shape({num_priors});
  framework::DDim ddim = framework::make_ddim(box_shape);

  framework::CLTensor box_width_cl_tensor(this->cl_helper_.CLContext(),
                                          this->cl_helper_.CLCommandQueue());
  box_width_cl_tensor.Resize(ddim);
  cl_mem box_width_Buffer =
      box_width_cl_tensor.mutable_with_data<float>(box_width);

  framework::CLTensor box_height_cl_tensor(this->cl_helper_.CLContext(),
                                           this->cl_helper_.CLCommandQueue());
  box_height_cl_tensor.Resize(ddim);
  cl_mem box_height_Buffer =
      box_height_cl_tensor.mutable_with_data<float>(box_height);

  framework::CLTensor variances_cl_tensor(this->cl_helper_.CLContext(),
                                          this->cl_helper_.CLCommandQueue());

  std::vector<int64_t> variances_shape({4});
  framework::DDim vddim = framework::make_ddim(variances_shape);

  variances_cl_tensor.Resize(vddim);
  cl_mem variances_Buffer =
      variances_cl_tensor.mutable_with_data<float>(variancesptr);

  //            DLOG << "c_block:" << c_block;
  //            DLOG << "w:" << w;
  //            DLOG << "nh:" << nh;
  //            DLOG << "step_width:" << step_width;
  //            DLOG << "step_height:" << step_height;
  //            DLOG << "offset:" << offset;
  //            DLOG << "img_width:" << img_width;
  //            DLOG << "img_height:" << img_height;
  //            DLOG << "num_priors:" << num_priors;
  //            DLOG << "C:" << C;
  //            DLOG << "isclip:" << isclip;
  //            printf("param.MinMaxAspectRatiosOrder() =
  //            %d\n",param.MinMaxAspectRatiosOrder()); for (int i = 0; i <
  //            num_priors; i++) {
  //                DLOG << box_width[i];
  //                DLOG << box_height[i];
  //            }
  status = clSetKernelArg(kernel, 0, sizeof(int), &c_block);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 1, sizeof(int), &w);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 2, sizeof(int), &nh);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 3, sizeof(cl_mem), &box_width_Buffer);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 4, sizeof(cl_mem), &box_height_Buffer);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 5, sizeof(cl_mem), &variances_Buffer);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 6, sizeof(cl_mem), &output_boxes);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 7, sizeof(cl_mem), &output_variances);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 8, sizeof(float), &step_width);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 9, sizeof(float), &step_height);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 10, sizeof(float), &offset);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 11, sizeof(int), &img_width);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 12, sizeof(int), &img_height);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 13, sizeof(int), &num_priors);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 14, sizeof(int), &C);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 15, sizeof(int), &isclip);
  CL_CHECK_ERRORS(status);
  size_t global_work_size[2] = {c_block, nh};
  status = clEnqueueNDRangeKernel(this->cl_helper_.CLCommandQueue(), kernel, 2,
                                  NULL, global_work_size, NULL, 0, NULL, NULL);
  CL_CHECK_ERRORS(status);

  paddle_mobile::memory::Free(box_width);
  paddle_mobile::memory::Free(box_height);
  paddle_mobile::memory::Free(variancesptr);
}
template class PriorBoxKernel<GPU_CL, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
