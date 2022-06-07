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

#include "lite/kernels/host/meshgrid_compute.h"
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <typename T, PrecisionType PType>
void MeshgridCompute<T, PType>::Run() {
  auto& param = this->template Param<operators::MeshgridParam>();
  std::vector<lite::Tensor*>& ins = param.X;
  std::vector<lite::Tensor*>& outs = param.Out;
  int64_t size = ins.size();
  std::vector<int64_t> shape(size);
  for (int64_t i = 0; i < size; ++i) {
    switch (ins[i]->dims().size()) {
      case 0:
        shape[i] = 1;
        break;
      case 1:
        shape[i] = ins[i]->dims()[0];
        break;
      default:
        LOG(FATAL) << "Meshgrid Op expected scalar or 1D tensor in the input "
                      "tensor list";
        break;
    }
  }

  DDim out_dims;
  out_dims.ConstructFrom(shape);

  for (int64_t i = 0; i < size; ++i) {
    T* dst = outs[i]->template mutable_data<T>();
    outs[i]->Resize(out_dims);
    Tensor reshape_ins_tensor;
    reshape_ins_tensor.ShareDataWith(*ins[i]);
    std::vector<int64_t> view_shape(size, 1);
    view_shape[i] = shape[i];
    DDim in_dims_reshape;
    in_dims_reshape.ConstructFrom(view_shape);
    reshape_ins_tensor.Resize(in_dims_reshape);
    const T* src = reshape_ins_tensor.data<T>();
    std::vector<int> bcast_dims(size);
    for (int64_t j = 0; j < size; j++) {
      bcast_dims[j] = shape[j];
    }
    bcast_dims[i] = 1;
    int inner_num = 1;
    int idx = size - 1;
    int outer_num = in_dims_reshape.count(0, idx);
    inner_num *= in_dims_reshape[idx];
    for (int j = 0; j < outer_num; ++j) {
      for (int k = 0; k < bcast_dims[idx]; ++k) {
        memcpy(dst + (j * bcast_dims[idx] + k) * inner_num,
               src + j * inner_num,
               sizeof(T) * inner_num);
      }
    }
    inner_num *= bcast_dims[idx];
    for (int idx = size - 2; idx >= 0; --idx) {
      int outer_num = in_dims_reshape.count(0, idx);
      inner_num *= in_dims_reshape[idx];
      for (int j = outer_num - 1; j >= 0; --j) {
        for (int k = bcast_dims[idx] - 1; k >= 0; --k) {
          memcpy(dst + (j * bcast_dims[idx] + k) * inner_num,
                 dst + j * inner_num,
                 sizeof(T) * inner_num);
        }
      }
      inner_num *= bcast_dims[idx];
    }
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using meshgrid_float =
    paddle::lite::kernels::host::MeshgridCompute<float, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(meshgrid, kHost, kFloat, kAny, meshgrid_float, float32)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kAny))})
    .Finalize();

using meshgrid_int32 =
    paddle::lite::kernels::host::MeshgridCompute<int, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(meshgrid, kHost, kFloat, kAny, meshgrid_int32, int32)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kInt32),
                                       DATALAYOUT(kAny))})
    .Finalize();

using meshgrid_int64 =
    paddle::lite::kernels::host::MeshgridCompute<int64_t, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(meshgrid, kHost, kFloat, kAny, meshgrid_int64, int64)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kInt64),
                                       DATALAYOUT(kAny))})
    .Finalize();
