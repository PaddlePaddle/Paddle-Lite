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
#ifdef TRANSPOSE2_OP

#include "operators/kernel/transpose2_kernel.h"

namespace paddle_mobile {
namespace operators {

template <typename P>
void TransposeCompute(const Transpose2Param<FPGA>& param) {
  const auto* input_x = param.InputX();
  const auto input_x_dims = input_x->dims();
  auto* out = param.Out();
  const auto axis = param.Axis();
  const auto* input_x_data = input_x->data<half>();
  auto* out_data = out->mutable_data<half>();

  size_t ndim = axis.size();
  std::vector<int> xdim(ndim);
  std::vector<int> xstride(ndim);
  std::vector<int> xout(ndim);
  for (int i = 0; i < ndim; i++) {
    int j = ndim - 1 - i;
    xdim[j] = input_x_dims[axis[i]];
    xstride[j] = 1;
    for (int k = axis[i] + 1; k < ndim; k++) {
      xstride[j] *= input_x_dims[k];
    }
    xout[j] = xstride[j] * xdim[j];
  }

  auto numel = input_x->numel();
  size_t pind = 0;
  std::vector<int> ind(ndim);
  for (int i = 0; i < numel; i++) {
    out_data[i] = input_x_data[pind];
    ind[0]++;
    pind += xstride[0];
    for (int j = 0; j < ndim - 1; j++) {
      if (ind[j] == xdim[j]) {
        ind[j + 1]++;
        ind[j] = 0;
        pind += xstride[j + 1];
        pind -= xout[j];
      } else {
        break;
      }
    }
  }
}

template <>
bool Transpose2Kernel<FPGA, float>::Init(Transpose2Param<FPGA>* param) {
  auto input = param->InputX();
  auto output = param->Out();
  auto axis = param->Axis();
  auto dim = input->dims();
  output->ShareDataWith(*input);

  auto dim_v = vectorize(dim);

  for (int i = 0; i < axis.size(); i++) {
    dim_v[i] = dim[axis[i]];
  }
  output->Resize(framework::make_ddim(dim_v));
  output->mutable_data<half>();

  DLOG << "input: " << input;
  DLOG << "output: " << output;

  if (param->InputX()->dims().size() == 4) {
    param->Out()->ShareDataWith(*param->InputX());
  }
  return true;
}

template <>
void Transpose2Kernel<FPGA, float>::Compute(
    const Transpose2Param<FPGA>& param) {
  // Transpose2Compute<float>(param);
  auto input = param.InputX();
  auto output = param.Out();

  if (param.InputX()->dims().size() != 4) {
    TransposeCompute<float>(param);
    auto out = param.Out();
    // out->set_data_aligned(param.InputX()->data_aligned());
    auto out_data = out->data<half>();
  }
  auto xShape = param.OutputXShape();
  xShape->Resize({0, 0});  // TODO(chonwhite) fix it;

  std::cout << "Out scale:" << param.Out()->zynqmpTensor()->scale()[0]
            << std::endl;
}

template class Transpose2Kernel<FPGA, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
