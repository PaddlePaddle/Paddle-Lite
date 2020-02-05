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

#include "lite/kernels/fpga/mul_compute.h"
#include <vector>
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"

#include "lite/backends/fpga/KD/debugger.hpp"

namespace paddle {
namespace lite {
namespace kernels {
namespace fpga {

using float16 = zynqmp::float16;

void MulCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();

  // ====================================================
  zynqmp::FullyConnectedParam& fc_param = pe_.param();

  param.output->mutable_data<float16>();

  fc_param.input = param.x->ZynqTensor();
  fc_param.output = param.output->ZynqTensor();
  fc_param.filter = param.y->ZynqTensor();

  fc_param.bias = &bias_;

  int channel = fc_param.filter->shape().channel();

  zynqmp::Shape bias_shape(zynqmp::N, {channel});

  float* bias_data =
      fc_param.bias->mutableData<float>(zynqmp::FP32, bias_shape);
  memset(bias_data, 0, channel * sizeof(float));
  bias_.flush();

  pe_.init();
  pe_.apply();
}

void mul(MulCompute* k) {
  auto& param = k->Param<operators::MulParam>();
  int num = param.x->dims()[0];
  int channel = param.x->dims()[1];

  int fn = param.y->dims()[1];

  float16* out_data = param.output->mutable_data<float16>();
  int g_index = 0;
  for (int n = 0; n < 1; n++) {
    for (int on = 0; on < fn; on++) {
      float sum = 0;
      int si = 0;
      for (int c = 0; c < channel; c++) {
        float value = zynqmp::half_to_float(param.x->data<float16>()[si]);
        int index = c * fn + on;
        float weight = param.y->data<float>()[index];
        sum += value * weight;
        si++;
      }
      out_data[g_index] = zynqmp::float_to_half(sum);
      g_index++;
    }
  }
}

void MulCompute::Run() {
  pe_.dispatch();
#ifdef FPGA_PRINT_TENSOR
  zynqmp::FullyConnectedParam& fc_param = pe_.param();
  Debugger::get_instance().registerOutput("mul", fc_param.output);
#endif
}

}  // namespace fpga
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    mul, kFPGA, kFP16, kNHWC, paddle::lite::kernels::fpga::MulCompute, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kFPGA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNHWC))})
    .Finalize();
