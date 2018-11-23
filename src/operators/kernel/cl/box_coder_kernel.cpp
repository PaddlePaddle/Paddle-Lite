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

#ifdef BOXCODER_OP

#include "operators/kernel/box_coder_kernel.h"

namespace paddle_mobile {
namespace operators {

template <>
bool BoxCoderKernel<GPU_CL, float>::Init(BoxCoderParam<GPU_CL>* param) {
  if (param->CodeType() == "decode_center_size") {
    this->cl_helper_.AddKernel("box_decoder", "box_coder_kernel.cl");
  }
  return true;
}

template <>
void BoxCoderKernel<GPU_CL, float>::Compute(
    const BoxCoderParam<GPU_CL>& param) {
  auto kernel = this->cl_helper_.KernelAt(0);
  auto default_work_size = this->cl_helper_.DefaultWorkSize(*param.OutputBox());
  const auto* input_priorbox = param.InputPriorBox();
  const auto* input_priorboxvar = param.InputPriorBoxVar();
  const auto* input_targetbox = param.InputTargetBox();
  const auto& code_type = param.CodeType();
  if (code_type == "decode_center_size") {
    auto prior_box_image = input_priorbox->GetCLImage();
    auto prior_box_var_image = input_priorboxvar->GetCLImage();
    auto target_box_image = input_targetbox->GetCLImage();
    auto output_image = param.OutputBox()->GetCLImage();
    auto& outputDim = param.OutputBox()->dims();
    int new_dims[4] = {1, 1, 1, 1};
    for (int i = 0; i < outputDim.size(); i++) {
      new_dims[4 - outputDim.size() + i] = outputDim[i];
    }
    int out_C = new_dims[1];
    int out_H = new_dims[2];
    DLOG << "out_C=" << out_C;
    DLOG << "out_H=" << out_H;
    DLOG << "default_work_size=" << default_work_size;
    cl_int status;
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &prior_box_image);
    CL_CHECK_ERRORS(status);
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &prior_box_var_image);
    CL_CHECK_ERRORS(status);
    status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &target_box_image);
    CL_CHECK_ERRORS(status);
    status = clSetKernelArg(kernel, 3, sizeof(cl_mem), &output_image);
    CL_CHECK_ERRORS(status);
    status = clSetKernelArg(kernel, 4, sizeof(int), &out_C);
    CL_CHECK_ERRORS(status);
    status = clSetKernelArg(kernel, 5, sizeof(int), &out_H);
    CL_CHECK_ERRORS(status);
    size_t global_work_size[2] = {default_work_size[0], default_work_size[2]};
    status =
        clEnqueueNDRangeKernel(this->cl_helper_.CLCommandQueue(), kernel, 2,
                               NULL, global_work_size, NULL, 0, NULL, NULL);
    CL_CHECK_ERRORS(status);
  }
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
