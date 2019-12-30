/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <algorithm>
#include <vector>
#include "lite/core/op_registry.h"
#include "lite/kernels/cuda/concat_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {
using Tensor = lite::Tensor;

template <typename Dtype>
__global__ void Concat(const int num,
                       const Dtype* in_data,
                       const int num_concats,
                       const int concat_size,
                       const int top_concat_axis,
                       const int bottom_concat_axis,
                       const int offset_concat_axis,
                       Dtype* out_data) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < num) {
    const int total_concat_size = concat_size * bottom_concat_axis;
    const int concat_num = index / total_concat_size;
    const int concat_index = index % total_concat_size;
    const int top_index =
        concat_index +
        (concat_num * top_concat_axis + offset_concat_axis) * concat_size;
    out_data[top_index] = in_data[index];
  }
}

template <typename Dtype>
void ConcatCompute<Dtype>::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<CUDAContext>();
  auto stream = ctx.exec_stream();

  std::vector<Tensor*> input = param.x;
  Tensor* output = param.output;
  auto* output_data = output->mutable_data<Dtype>(TARGET(kCUDA));
  int axis = param.axis;
  Tensor* axis_tensor = param.axis_tensor;
  if (axis_tensor != nullptr) {
    const int* axis_tensor_data = axis_tensor->data<int>();
    axis = axis_tensor_data[0];
  }
  int inner_size = 1;
  int outer_size = 1;
  auto input_dims = input[0]->dims();
  for (int i = 0; i < axis; i++) {
    outer_size *= input_dims[i];
  }

  for (int i = axis + 1; i < input_dims.size(); i++) {
    inner_size *= input_dims[i];
  }

  int all_concat_axis = param.output->dims()[axis];
  int in_num = input.size();
  int offset_concat_axis = 0;

  for (int i = 0; i < in_num; i++) {
    auto* input_data = input[i]->data<Dtype>();
    int input_concat_axis = input[i]->dims()[axis];
    int input_concat_size = input_concat_axis * inner_size;
    int num = input_concat_size * outer_size;
    int threads = 1024;
    int blocks = (num + threads - 1) / threads;
    Concat<<<blocks, threads, 0, stream>>>(num,
                                           input_data,
                                           outer_size,
                                           inner_size,
                                           all_concat_axis,
                                           input_concat_axis,
                                           offset_concat_axis,
                                           output_data);
    offset_concat_axis += input_concat_axis;
  }
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(concat,
                     kCUDA,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::cuda::ConcatCompute<float>,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindInput("AxisTensor",
               {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .Finalize();
