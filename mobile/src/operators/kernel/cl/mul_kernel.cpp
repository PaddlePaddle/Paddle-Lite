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

#ifdef MUL_OP

#include "operators/kernel/mul_kernel.h"

namespace paddle_mobile {
namespace operators {

template <>
bool MulKernel<GPU_CL, float>::Init(MulParam<GPU_CL> *param) {
  this->cl_helper_.AddKernel("fetch", "fetch_kernel.cl");
  this->cl_helper_.AddKernel("feed", "feed_kernel.cl");
  return true;
}

template <typename Dtype>
void MulCompute(const MulParam<GPU_CL> &param, cl_context context,
                cl_command_queue commandQueue, cl_kernel kernel0,
                cl_kernel kernel1) {
  auto input_x = param.InputX();
  Tensor *input_x_tensor = new Tensor();
  input_x_tensor->Resize(input_x->dims());
  input_x_tensor->mutable_data<float>();

  framework::CLImageToTensor(input_x, input_x_tensor, context, commandQueue,
                             kernel0);

  auto input_y = param.InputY();
  Tensor input_y_tensor(input_y->data<float>(), input_y->dims());

  const Tensor x_matrix =
      input_x_tensor->dims().size() > 2
          ? framework::ReshapeToMatrix(*input_x_tensor, param.XNumColDims())
          : *input_x_tensor;
  const Tensor y_matrix =
      input_y_tensor.dims().size() > 2
          ? framework::ReshapeToMatrix(input_y_tensor, param.YNumColDims())
          : input_y_tensor;

  auto out_dim = param.Out()->dims();
  if (out_dim.size() != 2) {
    param.Out()->Resize({x_matrix.dims()[0], y_matrix.dims()[1]});
  }

  auto output = param.Out();
  Tensor *output_tensor = new Tensor();
  output_tensor->Resize(output->dims());
  output_tensor->mutable_data<float>();
  math::MatMul<float, float>(x_matrix, false, y_matrix, false,
                             static_cast<float>(1), output_tensor,
                             static_cast<float>(0));

  output->InitEmptyImage(context, commandQueue, output_tensor->dims());
  framework::TensorToCLImage(output_tensor, output, context, commandQueue,
                             kernel1);

  delete (input_x_tensor);
  delete (output_tensor);
}

template <>
void MulKernel<GPU_CL, float>::Compute(const MulParam<GPU_CL> &param) {
  auto kernel0 = this->cl_helper_.KernelAt(0);
  auto kernel1 = this->cl_helper_.KernelAt(1);

  MulCompute<float>(param, this->cl_helper_.CLContext(),
                    this->cl_helper_.CLCommandQueue(), kernel0, kernel1);
}

template class MulKernel<GPU_CL, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
