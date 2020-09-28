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

#include <string>
#include <vector>

#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/core/type_system.h"
#include "lite/kernels/fpga/transpose_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace fpga {

using float16 = zynqmp::float16;

void transposeCompute(operators::TransposeParam param) {
  // copy from;
  const auto* input_x = param.x;
  const auto input_x_dims = input_x->dims();
  input_x->ZynqTensor()->invalidate();
  input_x->ZynqTensor()->unalignImage();

  const auto* input_x_data = input_x->data<float16>();

  auto* out = param.output;
  const auto axis = param.axis;

  auto* out_data = out->mutable_data<float16>();

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

// Transpose
void TransposeCompute::Run() {
  auto& param = this->Param<param_t>();
  param.output->mutable_data<zynqmp::float16>();
  param.x->ZynqTensor()->unalignImage();
  if (param.x->dims().size() != 4) {
    transposeCompute(param);
    param.output->ZynqTensor()->setAligned(param.x->ZynqTensor()->aligned());
  } else {
    param.output->ZynqTensor()->copyFrom(param.x->ZynqTensor());
  }
}

// Transpose2
void Transpose2Compute::Run() {
  auto& param = this->Param<param_t>();
  param.output->mutable_data<float16>();
  // param.x->ZynqTensor()->syncToCPU();
  param.x->ZynqTensor()->unalignImage();
  param.x->ZynqTensor()->flush();
  param.x->ZynqTensor()->invalidate();

  if (param.x->dims().size() != 4) {
    transposeCompute(param);
    param.output->ZynqTensor()->setAligned(param.x->ZynqTensor()->aligned());
  } else {
    param.output->ZynqTensor()->copyFrom(param.x->ZynqTensor());
  }

  param.output->ZynqTensor()->flush();
}

}  // namespace fpga
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

// Transpose
REGISTER_LITE_KERNEL(transpose,
                     kFPGA,
                     kFP16,
                     kNHWC,
                     paddle::lite::kernels::fpga::TransposeCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kFPGA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNHWC))})
    .Finalize();

// Transpose2
REGISTER_LITE_KERNEL(transpose2,
                     kFPGA,
                     kFP16,
                     kNHWC,
                     paddle::lite::kernels::fpga::Transpose2Compute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kFPGA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNHWC))})
    .BindOutput("XShape", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
