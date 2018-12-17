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

#ifdef FUSION_FC_OP

#include "operators/kernel/fusion_fc_kernel.h"
#include "operators/math/math_function.h"

namespace paddle_mobile {
namespace operators {

template <>
bool FusionFcKernel<GPU_CL, float>::Init(FusionFcParam<GPU_CL> *param) {
  param->InputY()->InitNormalCLImage(cl_helper_.CLContext(),
                                     this->cl_helper_.CLCommandQueue());
  param->InputZ()->InitNormalCLImage(cl_helper_.CLContext(),
                                     this->cl_helper_.CLCommandQueue());
  this->cl_helper_.AddKernel("fetch", "fetch_kernel.cl");
  this->cl_helper_.AddKernel("feed", "feed_kernel.cl");
  return true;
}

template <typename P>
void FusionFcCompute(const FusionFcParam<GPU_CL> &param, cl_context context,
                     cl_command_queue commandQueue, cl_kernel kernel0,
                     cl_kernel kernel1) {
  auto *input_x_image = param.InputX();
  auto *input_y_image = param.InputY();
  auto *input_z_image = param.InputZ();

  int axis = param.Axis();
  auto *out_image = param.Out();

  Tensor *input_x = new Tensor();
  input_x->Resize(input_x_image->dims());
  input_x->mutable_data<float>();
  framework::CLImageToTensor(input_x_image, input_x, context, commandQueue,
                             kernel0);

  Tensor *input_y = new Tensor();
  input_y->Resize(input_y_image->dims());
  input_y->mutable_data<float>();
  framework::CLImageToTensor(input_y_image, input_y, context, commandQueue,
                             kernel0);

  Tensor *input_z = new Tensor();
  input_z->Resize(input_z_image->dims());
  input_z->mutable_data<float>();
  framework::CLImageToTensor(input_z_image, input_z, context, commandQueue,
                             kernel0);
  auto *input_z_data = input_z->data<float>();

  DLOG << *input_x;
  DLOG << *input_y;
  DLOG << *input_z;

  Tensor *out = new Tensor();
  out->Resize(out_image->dims());
  out->mutable_data<float>();
  auto *out_data = out->mutable_data<float>();

  const Tensor x_matrix =
      input_x->dims().size() > 2
          ? framework::ReshapeToMatrix(*input_x, param.XNumColDims())
          : *input_x;
  const Tensor y_matrix =
      input_y->dims().size() > 2
          ? framework::ReshapeToMatrix(*input_y, param.YNumColDims())
          : *input_y;
  auto out_dim = out->dims();
  if (out_dim.size() != 2) {
    out->Resize({x_matrix.dims()[0], y_matrix.dims()[1]});
  }
  PADDLE_MOBILE_ENFORCE(out_dim.size() == 2, " out_dim.size must be 2.");
  PADDLE_MOBILE_ENFORCE(input_z->dims().size() == 1, "inpu_z size must be 1");
  PADDLE_MOBILE_ENFORCE(out_dim[1] == input_z->dims()[0],
                        " out_dim.size must be 2.");
  axis = (axis == -1 ? out_dim.size() - input_z->dims().size() : axis);
  PADDLE_MOBILE_ENFORCE(axis == 1, " to fit broadcast, axis = 1. ");

  int64_t classes = input_z->numel();
  for (int i = 0; i < out_dim[0]; i++) {
    memory::Copy(out_data + i * classes, input_z_data, sizeof(float) * classes);
  }

  math::MatMul<float, float>(x_matrix, false, y_matrix, false,
                             static_cast<float>(1), out, static_cast<float>(1),
                             false);

  out_image->InitEmptyImage(context, commandQueue, out->dims());
  framework::TensorToCLImage(out, out_image, context, commandQueue, kernel1);

  delete (input_x);
  delete (input_y);
  delete (input_z);
  delete (out);
  PADDLE_MOBILE_ENFORCE(out_dim.size() == 2, " out_dim.size must be 2.");
}

template <>
void FusionFcKernel<GPU_CL, float>::Compute(
    const FusionFcParam<GPU_CL> &param) {
  auto kernel0 = this->cl_helper_.KernelAt(0);
  auto kernel1 = this->cl_helper_.KernelAt(1);
  FusionFcCompute<float>(param, this->cl_helper_.CLContext(),
                         this->cl_helper_.CLCommandQueue(), kernel0, kernel1);
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
