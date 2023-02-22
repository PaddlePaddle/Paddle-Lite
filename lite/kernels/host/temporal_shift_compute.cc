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

#include "lite/kernels/host/temporal_shift_compute.h"
#include <string>
#include "lite/backends/host/math/temporal_shift.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <>
void TemporalShiftCompute<PRECISION(kFloat), PRECISION(kFloat)>::Run() {
  auto& param = Param<operators::TemporalShiftParam>();
  const lite::Tensor* input = param.X;
  lite::Tensor* output = param.Out;
  int t = param.seg_num;
  float shift_ratio = param.shift_ratio;
  DataLayoutType data_layout;
  if (param.data_format == "NCHW") {
    data_layout = DATALAYOUT(kNCHW);
  } else if (param.data_format == "NHWC") {
    data_layout = DATALAYOUT(kNHWC);
  } else {
    LOG(FATAL) << "Unknown datalayout";
  }

  auto input_dims = input->dims();
  const int nt = input_dims[0];
  const int c =
      data_layout == DATALAYOUT(kNCHW) ? input_dims[1] : input_dims[3];
  const int h =
      data_layout == DATALAYOUT(kNCHW) ? input_dims[2] : input_dims[1];
  const int w =
      data_layout == DATALAYOUT(kNCHW) ? input_dims[3] : input_dims[2];

  const int hw = h * w;
  const int chw = c * hw;
  const int tchw = t * chw;
  const int ntchw = nt * chw;

  const int c1 = static_cast<int>(c * shift_ratio);
  const int c2 = static_cast<int>(c * 2 * shift_ratio);

  DDim out_dims;
  if (data_layout == DATALAYOUT(kNCHW)) {
    out_dims.ConstructFrom({nt, c, h, w});
  } else {
    out_dims.ConstructFrom({nt, h, w, c});
  }

  const float* input_data = input->data<float>();
  output->Resize(out_dims);
  float* output_data = output->mutable_data<float>();

  if (data_layout == DATALAYOUT(kNCHW)) {
    lite::host::math::temporalshiftNCHW_func(
        input_data, output_data, ntchw, tchw, chw, hw, t, c1, c2);
  } else {
    lite::host::math::temporalshiftNHWC_func(
        input_data, output_data, ntchw, tchw, chw, t, c, c1, c2);
  }
  return;
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
typedef paddle::lite::kernels::host::TemporalShiftCompute<PRECISION(kFloat),
                                                          PRECISION(kFloat)>
    TSfp32;

REGISTER_LITE_KERNEL(temporal_shift, kHost, kFloat, kNCHW, TSfp32, fp32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .Finalize();
