// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/xpu/expand_compute.h"
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <typename T>    
void ExpandCompute<T>::Run() {
  auto& ctx = this->ctx_->As<XPUContext>();
  auto& param = this->Param<operators::ExpandParam>();
  const auto* x = param.X;
  auto* out = param.Out;

  std::vector<int> expand_times;
  if (param.ExpandTimes != nullptr) {
    auto expand_times_data = param.ExpandTimes->template data<int>();
    for (int64_t i = 0; i < param.ExpandTimes->numel(); i++) {
      expand_times.push_back(expand_times_data[i]);
    }
  } else if (!param.expand_times_tensor.empty()) {
    for (size_t i = 0; i < param.expand_times_tensor.size(); i++) {
      expand_times.push_back(
          param.expand_times_tensor[i]->template data<int>()[0]);
    }
  } else {
    expand_times = param.expand_times;
  }

  // std::cout << "expand_times: " << std::endl;
  // for(size_t i = 0; i < expand_times.size(); i++)
  //     std::cout << expand_times[i] << "  ";
  // std::cout << std::endl;
  std::vector<int> vec_in_dims;
  //int dims = expand_times.size();
  DDim in_shape = x->dims();
  //std::cout << "xshape: " << std::endl;
  for (int i = 0; i < in_shape.size(); ++i) {
      //vec_in_dims.push_back(static_cast<int64_t>(in_shape[i]));
    vec_in_dims.push_back(static_cast<T>(in_shape[i]));
    //std::cout << vec_in_dims[i] << "  ";
  }
  //std::cout << std::endl;

  //std::cout << "yshape: " << std::endl;
  std::vector<int> vec_out_dims(vec_in_dims.size());
  for (size_t i = 0; i < vec_in_dims.size(); i++) {
    vec_out_dims[i] = vec_in_dims[i] * expand_times[i];
    //std::cout << vec_out_dims[i] << "  ";
  }
  //std::cout << std::endl;

  int r = xdnn::broadcast<T>(ctx.GetRawContext(),
                             x->template data<T>(),
                             out->template mutable_data<T>(TARGET(kXPU)),
                             vec_in_dims,
                             vec_out_dims);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using expand_xpu_float = paddle::lite::kernels::xpu::ExpandCompute<float>;
REGISTER_LITE_KERNEL(expand, kXPU, kFloat, kAny, expand_xpu_float, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindInput("ExpandTimes",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindInput("expand_times_tensor",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kAny))})
    .Finalize();


using expand_xpu_int32 = paddle::lite::kernels::xpu::ExpandCompute<int32_t>;
REGISTER_LITE_KERNEL(expand, kXPU, kFloat, kAny, expand_xpu_int32, def_int32)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindInput("ExpandTimes",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindInput("expand_times_tensor",
               {LiteType::GetTensorTy(TARGET(kXPU),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kXPU),
                                       PRECISION(kInt32),
                                       DATALAYOUT(kAny))})
    .Finalize();
