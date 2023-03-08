// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/host/pad_compute.h"
#include <algorithm>
#include <string>
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <typename DataType>
void pad_inner_1(const DataType* input,
                 DataType* output,
                 DataType val,
                 int outer_num,
                 int inner_num,
                 int pad_l,
                 int pad_r) {  // outter/inner is outputdims
  int num = inner_num - pad_l - pad_r;
  for (int i = 0; i < outer_num; i++) {
    for (int j = 0; j < inner_num;) {
      if (j < pad_l) {
        std::fill_n(output + i * inner_num + j, pad_l, val);
        j += pad_l;
      } else if (j >= inner_num - pad_r) {
        std::fill_n(output + i * inner_num + j, pad_r, val);
        j += pad_r;
      } else {
        memcpy(output + i * inner_num + j,
               input + i * num + j - pad_l,
               num * sizeof(DataType));
        j += num;
      }
    }
  }
}

template <typename DataType>
void pad_inner_2(const DataType* input,
                 DataType* output,
                 DataType val,
                 DDim in_dims,
                 DDim out_dims,
                 std::vector<int> paddings,
                 DataType* workspace) {
  int outer_num = in_dims[0];
  int inner_num = out_dims[1];
  int pad_num = 1;
  int pad_idx = 2;
  pad_inner_1(input,
              workspace,
              val,
              outer_num,
              inner_num,
              paddings[pad_idx],
              paddings[pad_idx + 1]);
  outer_num = 1;
  inner_num = out_dims[0] * out_dims[1];
  pad_num = out_dims[1];
  pad_idx = 0;
  pad_inner_1(workspace,
              output,
              val,
              outer_num,
              inner_num,
              paddings[pad_idx] * pad_num,
              paddings[pad_idx + 1] * pad_num);
}

template <typename DataType>
void pad_inner_3(const DataType* input,
                 DataType* output,
                 DataType val,
                 DDim in_dims,
                 DDim out_dims,
                 std::vector<int> paddings,
                 DataType* workspace) {
  int outer_num = in_dims[0] * in_dims[1];
  int inner_num = out_dims[2];
  int pad_num = 1;
  int pad_idx = 4;
  pad_inner_1(input,
              output,
              val,
              outer_num,
              inner_num,
              paddings[pad_idx],
              paddings[pad_idx + 1]);
  outer_num = in_dims[0];
  inner_num = out_dims[1] * out_dims[2];
  pad_num = out_dims[2];
  pad_idx = 2;
  pad_inner_1(output,
              workspace,
              val,
              outer_num,
              inner_num,
              paddings[pad_idx] * pad_num,
              paddings[pad_idx + 1] * pad_num);
  outer_num = 1;
  inner_num = out_dims[0] * out_dims[1] * out_dims[2];
  pad_num = out_dims[1] * out_dims[2];
  pad_idx = 0;
  pad_inner_1(workspace,
              output,
              val,
              outer_num,
              inner_num,
              paddings[pad_idx] * pad_num,
              paddings[pad_idx + 1] * pad_num);
}

template <typename DataType>
void pad_inner_4(const DataType* input,
                 DataType* output,
                 DataType val,
                 DDim in_dims,
                 DDim out_dims,
                 std::vector<int> paddings,
                 DataType* workspace) {
  int outer_num = in_dims[0] * in_dims[1] * in_dims[2];
  int inner_num = out_dims[3];
  int pad_num = 1;
  pad_inner_1(
      input, workspace, val, outer_num, inner_num, paddings[6], paddings[7]);
  outer_num = in_dims[0] * in_dims[1];
  inner_num = out_dims[2] * out_dims[3];
  pad_num = out_dims[3];
  pad_inner_1(workspace,
              output,
              val,
              outer_num,
              inner_num,
              paddings[4] * pad_num,
              paddings[5] * pad_num);
  outer_num = in_dims[0];
  inner_num = out_dims[1] * out_dims[2] * out_dims[3];
  pad_num = out_dims[2] * out_dims[3];
  pad_inner_1(output,
              workspace,
              val,
              outer_num,
              inner_num,
              paddings[2] * pad_num,
              paddings[3] * pad_num);
  outer_num = 1;
  inner_num = out_dims[0] * out_dims[1] * out_dims[2] * out_dims[3];
  pad_num = out_dims[1] * out_dims[2] * out_dims[3];
  pad_inner_1(workspace,
              output,
              val,
              outer_num,
              inner_num,
              paddings[0] * pad_num,
              paddings[1] * pad_num);
}

template <typename DataType>
void pad_inner_5(const DataType* input,
                 DataType* output,
                 DataType val,
                 DDim in_dims,
                 DDim out_dims,
                 std::vector<int> paddings,
                 DataType* workspace) {
  int outer_num = in_dims[0] * in_dims[1] * in_dims[2] * in_dims[3];
  int inner_num = out_dims[4];
  int pad_num = 1;
  int pad_idx = 8;
  pad_inner_1(input,
              output,
              val,
              outer_num,
              inner_num,
              paddings[pad_idx] * pad_num,
              paddings[pad_idx + 1] * pad_num);
  outer_num = in_dims[0] * in_dims[1] * in_dims[2];
  inner_num = out_dims[3] * out_dims[4];
  pad_num = out_dims[4];
  pad_idx = 6;
  pad_inner_1(output,
              workspace,
              val,
              outer_num,
              inner_num,
              paddings[pad_idx] * pad_num,
              paddings[pad_idx + 1] * pad_num);
  outer_num = in_dims[0] * in_dims[1];
  inner_num = out_dims[2] * out_dims[3] * out_dims[4];
  pad_num = out_dims[3] * out_dims[4];
  pad_idx = 4;
  pad_inner_1(workspace,
              output,
              val,
              outer_num,
              inner_num,
              paddings[pad_idx] * pad_num,
              paddings[pad_idx + 1] * pad_num);
  outer_num = in_dims[0];
  inner_num = out_dims[1] * out_dims[2] * out_dims[3] * out_dims[4];
  pad_num = out_dims[2] * out_dims[3] * out_dims[4];
  pad_idx = 2;
  pad_inner_1(output,
              workspace,
              val,
              outer_num,
              inner_num,
              paddings[pad_idx] * pad_num,
              paddings[pad_idx + 1] * pad_num);
  outer_num = 1;
  inner_num =
      out_dims[0] * out_dims[1] * out_dims[2] * out_dims[3] * out_dims[4];
  pad_num = out_dims[1] * out_dims[2] * out_dims[3] * out_dims[4];
  pad_idx = 0;
  pad_inner_1(workspace,
              output,
              val,
              outer_num,
              inner_num,
              paddings[pad_idx] * pad_num,
              paddings[pad_idx + 1] * pad_num);
}

template <typename DataType>
void pad_inner_6(const DataType* input,
                 DataType* output,
                 DataType val,
                 DDim in_dims,
                 DDim out_dims,
                 std::vector<int> paddings,
                 DataType* workspace) {
  int outer_num =
      in_dims[0] * in_dims[1] * in_dims[2] * in_dims[3] * in_dims[4];
  int inner_num = out_dims[5];
  int pad_num = 1;
  int pad_idx = 10;
  pad_inner_1(input,
              workspace,
              val,
              outer_num,
              inner_num,
              paddings[pad_idx],
              paddings[pad_idx + 1]);
  outer_num = in_dims[0] * in_dims[1] * in_dims[2] * in_dims[3];
  inner_num = out_dims[4] * out_dims[5];
  pad_num = out_dims[5];
  pad_idx = 8;
  pad_inner_1(workspace,
              output,
              val,
              outer_num,
              inner_num,
              paddings[pad_idx] * pad_num,
              paddings[pad_idx + 1] * pad_num);
  outer_num = in_dims[0] * in_dims[1] * in_dims[2];
  inner_num = out_dims[3] * out_dims[4] * out_dims[5];
  pad_num = out_dims[4] * out_dims[5];
  pad_idx = 6;
  pad_inner_1(output,
              workspace,
              val,
              outer_num,
              inner_num,
              paddings[pad_idx] * pad_num,
              paddings[pad_idx + 1] * pad_num);
  outer_num = in_dims[0] * in_dims[1];
  inner_num = out_dims[2] * out_dims[3] * out_dims[4] * out_dims[5];
  pad_num = out_dims[3] * out_dims[4] * out_dims[5];
  pad_idx = 4;
  pad_inner_1(workspace,
              output,
              val,
              outer_num,
              inner_num,
              paddings[pad_idx] * pad_num,
              paddings[pad_idx + 1] * pad_num);
  outer_num = in_dims[0];
  inner_num =
      out_dims[1] * out_dims[2] * out_dims[3] * out_dims[4] * out_dims[5];
  pad_num = out_dims[2] * out_dims[3] * out_dims[4] * out_dims[5];
  pad_idx = 2;
  pad_inner_1(output,
              workspace,
              val,
              outer_num,
              inner_num,
              paddings[pad_idx] * pad_num,
              paddings[pad_idx + 1] * pad_num);
  outer_num = 1;
  inner_num = out_dims[0] * out_dims[1] * out_dims[2] * out_dims[3] *
              out_dims[4] * out_dims[5];
  pad_num = out_dims[1] * out_dims[2] * out_dims[3] * out_dims[4] * out_dims[5];
  pad_idx = 0;
  pad_inner_1(workspace,
              output,
              val,
              outer_num,
              inner_num,
              paddings[pad_idx] * pad_num,
              paddings[pad_idx + 1] * pad_num);
}

template <typename DataType>
void pad_compute_constant(const lite::Tensor& input,
                          const std::vector<int>& paddings,
                          lite::Tensor* output,
                          float pad_val,
                          DataType* workspace) {
  DDim in_dims = input.dims();
  DDim out_dims = output->dims();
  const DataType* in_data = input.template data<DataType>();
  DataType* out_data = output->template mutable_data<DataType>();
  DataType pad_value = static_cast<DataType>(pad_val);
  switch (in_dims.size()) {
    case 1:
      pad_inner_1<DataType>(in_data,
                            out_data,
                            pad_value,
                            1,
                            out_dims[0],
                            paddings[0],
                            paddings[1]);
      break;
    case 2:
      pad_inner_2<DataType>(
          in_data, out_data, pad_value, in_dims, out_dims, paddings, workspace);
      break;
    case 3:
      pad_inner_3<DataType>(
          in_data, out_data, pad_value, in_dims, out_dims, paddings, workspace);
      break;
    case 4:
      pad_inner_4<DataType>(
          in_data, out_data, pad_value, in_dims, out_dims, paddings, workspace);
      break;
    case 5:
      pad_inner_5<DataType>(
          in_data, out_data, pad_value, in_dims, out_dims, paddings, workspace);
      break;
    case 6:
      pad_inner_6<DataType>(
          in_data, out_data, pad_value, in_dims, out_dims, paddings, workspace);
      break;
    default:
      LOG(FATAL) << "Pad Only supports input_dims{1-6}, but receive "
                 << in_dims.size();
      break;
  }
}

template <typename Dtype>
void PadCompute<Dtype>::Run() {
  auto& param = Param<operators::PadParam>();
  auto x_dims = param.X->dims();
  int x_dims_size = x_dims.size();
  if (param.paddings.size() == 2 * x_dims_size) {
    std::vector<int64_t> out_dims(x_dims_size, 0);
    for (int i = 0; i < x_dims_size; i++)
      out_dims[i] =
          x_dims[i] + param.paddings[i * 2] + param.paddings[i * 2 + 1];
    param.Out->Resize(out_dims);
    param.Out->template mutable_data<Dtype>();
    Dtype* workspace = reinterpret_cast<Dtype*>(
        TargetMalloc(TARGET(kHost), sizeof(Dtype) * param.Out->numel()));
    pad_compute_constant<Dtype>(
        *(param.X), param.paddings, param.Out, param.pad_value, workspace);
    TargetFree(TARGET(kHost), workspace);
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using pad_fp32_compute = paddle::lite::kernels::host::PadCompute<float>;
REGISTER_LITE_KERNEL(pad, kHost, kFloat, kAny, pad_fp32_compute, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kAny))})
    .Finalize();

using pad_int32_compute = paddle::lite::kernels::host::PadCompute<int>;
REGISTER_LITE_KERNEL(pad, kHost, kFloat, kAny, pad_int32_compute, int32)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kInt32),
                                       DATALAYOUT(kAny))})
    .Finalize();

using pad_int64_compute = paddle::lite::kernels::host::PadCompute<int64_t>;
REGISTER_LITE_KERNEL(pad, kHost, kFloat, kAny, pad_int64_compute, int64)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kInt64),
                                       DATALAYOUT(kAny))})
    .Finalize();
