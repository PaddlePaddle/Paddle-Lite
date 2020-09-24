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
#include "lite/kernels/arm/gather_compute.h"
#include <vector>
#include "lite/backends/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

template <typename IndexType, typename DataType>
void GatherFunc(const operators::GatherParam& param) {
  auto src_dims = param.X->dims();
  auto index_size = param.Index->dims()[0];
  auto* p_src = param.X->data<DataType>();
  const IndexType* p_index = param.Index->data<IndexType>();
  auto* p_output = param.Out->mutable_data<DataType>();

  int slice_size = 1;
  for (size_t i = 1; i < src_dims.size(); ++i) {
    slice_size *= src_dims[i];
  }
  for (int i = 0; i < index_size; ++i) {
    IndexType index_ = p_index[i];
    memcpy(p_output + i * slice_size,
           p_src + index_ * slice_size,
           slice_size * sizeof(DataType));
  }
}

template <typename IndexType>
void GatherCompute<IndexType>::Run() {
  auto& param = this->template Param<operators::GatherParam>();

  switch (param.X->precision()) {
    case PRECISION(kFloat):
      GatherFunc<IndexType, float>(param);
      break;
    case PRECISION(kInt8):
      GatherFunc<IndexType, int8_t>(param);
      break;
    case PRECISION(kInt16):
      GatherFunc<IndexType, int16_t>(param);
      break;
    case PRECISION(kInt32):
      GatherFunc<IndexType, int32_t>(param);
      break;
    case PRECISION(kInt64):
      GatherFunc<IndexType, int64_t>(param);
      break;
    default:
      LOG(FATAL) << "Gather does not implement for the "
                 << "input type:" << static_cast<int>(param.X->precision());
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(gather,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::GatherCompute<int32_t>,
                     int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kAny))})
    .BindInput("Index",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kAny))})
    .Finalize();

REGISTER_LITE_KERNEL(gather,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::GatherCompute<int64_t>,
                     int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kAny))})
    .BindInput("Index",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kAny))})
    .Finalize();
