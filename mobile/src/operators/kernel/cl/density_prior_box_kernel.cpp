/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef DENSITY_PRIORBOX_OP

#include <operators/kernel/prior_box_kernel.h>
#include "framework/cl/cl_tensor.h"
namespace paddle_mobile {
namespace operators {

template <>
bool DensityPriorBoxKernel<GPU_CL, float>::Init(
    paddle_mobile::operators::DensityPriorBoxParam<paddle_mobile::GPU_CL>
        *param) {
  this->cl_helper_.AddKernel("density_prior_box",
                             "density_prior_box_kernel.cl");
  return true;
}

template <>
void DensityPriorBoxKernel<GPU_CL, float>::Compute(
    const paddle_mobile::operators::DensityPriorBoxParam<paddle_mobile::GPU_CL>
        &param) {
  auto kernel = this->cl_helper_.KernelAt(0);
  const auto *input = param.Input();
  const auto input_dims = input->dims();
  const auto input_image_dims = param.InputImage()->dims();

  auto output_boxes = param.OutputBoxes()->GetCLImage();
  auto output_var = param.OutputVariances()->GetCLImage();

  float step_w = param.StepW();
  float step_h = param.StepH();
  float offset = param.Offset();
  vector<float> fixed_sizes = param.FixedSizes();
  vector<float> fixed_ratios = param.FixedRatios();
  vector<int> densities = param.Densities();
  vector<float> variances = param.Variances();

  // feature map
  auto input_heigh = input_dims[2];
  auto input_width = input_dims[3];

  auto image_heigh = input_image_dims[2];
  auto image_width = input_image_dims[3];

  const int C = param.OutputBoxes()->dims()[1];

  if (step_w == 0 || step_h == 0) {
    step_h = static_cast<float>(image_heigh) / input_heigh;
    step_w = static_cast<float>(image_width) / input_width;
  }
  int num_density = 0;
  for (int l = 0; l < densities.size(); ++l) {
    num_density += densities[l] * densities[l] * fixed_ratios.size();
  }

  param.OutputBoxes()->Resize({input_heigh, input_width, num_density, 4});
  int step_average = static_cast<int>((step_w + step_h) * 0.5);
  int densities_and_fixedsize_size = densities.size();
  int fix_ratio_size = fixed_ratios.size();

  auto default_work = this->cl_helper_.DefaultWorkSize(*param.OutputBoxes());

  float *densities_data[densities.size() + fixed_sizes.size() + fix_ratio_size];

  int status;

  for (int i = 0; i < densities.size(); ++i) {
    float density = densities[i];
    densities_data[i] = &density;
  }

  for (int k = 0; k < fixed_sizes.size(); ++k) {
    densities_data[k + densities.size()] = &fixed_sizes[k];
  }

  for (int j = 0; j < fixed_ratios.size(); ++j) {
    float sqrt_ratios = sqrt(fixed_ratios[j]);
    densities_data[j + densities.size() + fixed_sizes.size()] = &sqrt_ratios;
  }

  cl_mem densities_memobj = clCreateBuffer(
      this->cl_helper_.CLContext(), CL_MEM_READ_WRITE,
      sizeof(float) * (densities.size() * 2 + fix_ratio_size), NULL, &status);
  status = clEnqueueWriteBuffer(
      this->cl_helper_.CLCommandQueue(), densities_memobj, CL_FALSE, 0,
      (densities.size() * 2 + fix_ratio_size) * sizeof(float), densities_data,
      0, NULL, NULL);
  CL_CHECK_ERRORS(status);

  float variances0 = variances[0];
  float variances1 = variances[1];
  float variances2 = variances[2];
  float variances3 = variances[3];

  status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &output_boxes);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_var);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &densities_memobj);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 3, sizeof(float), &step_h);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 4, sizeof(float), &step_w);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 5, sizeof(int), &variances0);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 6, sizeof(int), &variances1);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 7, sizeof(int), &variances2);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 8, sizeof(int), &variances3);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 9, sizeof(float), &offset);
  CL_CHECK_ERRORS(status);
  status =
      clSetKernelArg(kernel, 10, sizeof(int), &densities_and_fixedsize_size);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 11, sizeof(int), &image_width);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 12, sizeof(int), &image_heigh);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 13, sizeof(int), &C);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 14, sizeof(int), &num_density);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 15, sizeof(int), &step_average);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 16, sizeof(int), &input_width);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 17, sizeof(int), &default_work[0]);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 18, sizeof(int), &fix_ratio_size);
  CL_CHECK_ERRORS(status);
  status = clEnqueueNDRangeKernel(this->cl_helper_.CLCommandQueue(), kernel,
                                  default_work.size(), NULL,
                                  default_work.data(), NULL, 0, NULL, NULL);
  CL_CHECK_ERRORS(status);
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
