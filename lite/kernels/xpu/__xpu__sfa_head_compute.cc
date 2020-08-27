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

#include "lite/kernels/xpu/__xpu__sfa_head_compute.h"
#include <string>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void XPUSfaHeadCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();
  std::string vis_type = param.op_type;
  auto input = param.input;

  const int batch = static_cast<int>(input->dims()[0]);
  const int m = static_cast<int>(input->dims()[1]);
  const int n = static_cast<int>(input->dims()[2]);
  if (vis_type == "meanstd") {
    int r = xdnn::vis_meanstd(ctx.GetRawContext(),
                              param.input->data<float>(),
                              param.output->mutable_data<float>(TARGET(kXPU)),
                              batch,
                              m,
                              n);
    CHECK_EQ(r, 0) << "XPU kernel error";
    (void)param.output->mutable_data<float>();
  } else if (vis_type == "moment") {
    int r = xdnn::vis_moment(ctx.GetRawContext(),
                             param.input->data<float>(),
                             param.output->mutable_data<float>(TARGET(kXPU)),
                             batch,
                             m,
                             n);
    CHECK_EQ(r, 0) << "XPU kernel error";
  } else {
    LOG(FATAL) << "vis xpu op not supported type " << vis_type.c_str();
  }
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(__xpu__sfa_head,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::XPUSfaHeadCompute,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
