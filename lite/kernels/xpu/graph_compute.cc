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

#include "lite/kernels/xpu/graph_compute.h"
#include <sys/time.h>
#include <time.h>
#include <string>
#include <vector>
#include "lite/backends/xpu/runtime.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void GraphCompute::PrepareForRun() {
  // auto& ctx = this->ctx_->template As<XPUContext>();
  auto& param = this->Param<param_t>();
  CHECK(param.weight);
  CHECK(lite::xpu::LoadModel(*param.weight, &model_runtime_));
  CHECK(model_runtime_ != nullptr);
}

void GraphCompute::Run() {
  auto& param = this->Param<param_t>();
  auto GetCurrentUS = []() -> double {
    struct timeval time;
    gettimeofday(&time, NULL);
    return 1e+6 * time.tv_sec + time.tv_usec;
  };
  auto start_time = GetCurrentUS();
  for (int i = 0; i < param.inputs.size(); i++) {
    auto input_var_name = param.inputs[i].first;
    auto input_tensor = param.inputs[i].second;
    LOG(INFO) << "input dims[" << i << " " << input_var_name
              << "]: " << input_tensor->dims();
    auto input_tensor_data = input_tensor->data<float>();
    for (int j = 0; j < input_tensor->dims().production(); j++) {
      VLOG(3) << input_tensor_data[j];
    }
    auto input_ndarray = xtcl::xNDArray::Empty(
        input_tensor->dims().Vectorize(), {kDLFloat, 32, 1}, {kDLCPU, 0});
    auto input_ndarray_data =
        static_cast<float*>(input_ndarray.ToDLPack()->dl_tensor.data);
    std::memcpy(input_ndarray_data,
                input_tensor_data,
                sizeof(float) * input_tensor->dims().production());
    model_runtime_->SetInputZeroCopy(input_var_name,
                                     &input_ndarray.ToDLPack()->dl_tensor);
  }
  model_runtime_->Run();
  for (int i = 0; i < param.outputs.size(); i++) {
    auto output_ndarray = model_runtime_->GetOutput(i);
    auto output_var_name = param.outputs[i].first;
    auto output_tensor = param.outputs[i].second;
    output_tensor->Resize(output_ndarray.Shape());
    LOG(INFO) << "output dims[" << i << " " << output_var_name
              << "]: " << output_tensor->dims();
    auto output_ndarray_data =
        static_cast<float*>(output_ndarray.ToDLPack()->dl_tensor.data);
    auto output_tensor_data = output_tensor->mutable_data<float>();
    std::memcpy(output_tensor_data,
                output_ndarray_data,
                sizeof(float) * output_tensor->dims().production());
    for (int j = 0; j < output_tensor->dims().production(); j++) {
      VLOG(3) << output_tensor_data[j];
    }
  }
  LOG(INFO) << "[XPU] Process cost " << GetCurrentUS() - start_time << " us";
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(graph_op,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::GraphCompute,
                     def)
    .BindInput("Inputs", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Weight", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Outputs", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
