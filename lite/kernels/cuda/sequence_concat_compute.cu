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

#include <vector>
#include "lite/core/op_registry.h"
#include "lite/core/target_wrapper.h"
#include "lite/kernels/cuda/sequence_concat_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

template <typename dtype>
__global__ void concat_impl_cuda(const int nthreads,
                                 const dtype* in_data,
                                 const int num_concats,
                                 const int concat_size,
                                 const int top_concat_axis,
                                 const int bottom_concat_axis,
                                 const int offset_concat_axis,
                                 dtype* out_data) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
       index += blockDim.x * gridDim.x) {
    const int total_concat_size = concat_size * bottom_concat_axis;
    const int concat_num = index / total_concat_size;
    const int concat_index = index % total_concat_size;
    const int top_index =
        concat_index +
        (concat_num * top_concat_axis + offset_concat_axis) * concat_size;
    out_data[top_index] = in_data[index];
  }
}

template <typename dtype>
__global__ void concat_impl_2d_impl(const int inner_size,
                                    const int num_concats,
                                    const dtype* in_data,
                                    const int concat_size,
                                    const int out_concat_axis,
                                    const int offset_concat_axis,
                                    dtype* out_data) {
  int idx_inner = threadIdx.x + blockIdx.x * blockDim.x;
  int idx_outer = threadIdx.y + blockIdx.y * blockDim.y;

  if (idx_inner < inner_size && idx_outer < num_concats) {
    int idx_input = idx_outer * inner_size + idx_inner;
    int idx_output =
        (idx_outer * out_concat_axis + offset_concat_axis) * concat_size +
        idx_inner;
    out_data[idx_output] = in_data[idx_input];
  }
}

void SequenceConcatCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<CUDAContext>();
  auto stream = ctx.exec_stream();

  const int BLOCK_SIZE = 32;
  const int axis = 1;
  int num_concats = param.X[0]->dims().count(0, axis);
  int concat_input_size =
      param.X[0]->dims().count(axis + 1, param.X[0]->dims().size());

  int input_size = param.X.size();
  std::vector<std::vector<int64_t>> shapes_in(input_size);
  for (int i = 0; i < input_size; ++i) {
    shapes_in[i] = param.X[i]->dims().Vectorize();
  }
  std::vector<int64_t> shape_out = shapes_in[0];

  // compute output shape
  for (int i = 1; i < input_size; ++i) {
    for (int j = 0; j < shapes_in[i].size(); ++j) {
      if (j == axis) {
        continue;
      } else if (shapes_in[i][j] != -1) {
        CHECK_EQ(shape_out[j], shapes_in[i][j])
            << "All inputs must have the same shape, except at concat_axis.";
      }
    }
    shape_out[axis] += shapes_in[i][axis];
  }

  param.Out->Resize(shape_out);
  float* out_data = param.Out->mutable_data<float>(TARGET(kCUDA));
  int offset_concat_axis = 0;
  const int out_concat_axis = shape_out[axis];

  for (int i = 0; i < input_size; ++i) {
    std::vector<int64_t> in_shape = param.X[i]->dims().Vectorize();
    const auto* in_data = param.X[i]->data<float>();
    const int in_concat_axis = in_shape[axis];
    const int in_concat_size = in_concat_axis * concat_input_size;
    const int nthreads = in_concat_size * num_concats;
    float ratio = static_cast<float>(in_concat_size) / num_concats;
    bool is_balance = (ratio > 0.1 && ratio < 10);
    if (is_balance) {
      int block_x = BLOCK_SIZE;
      int block_y = BLOCK_SIZE;
      int grid_x = (in_concat_size + block_x - 1) / block_x;
      int grid_y = (num_concats + block_y - 1) / block_y;
      dim3 block(block_x, block_y);
      dim3 grid(grid_x, grid_y);
      concat_impl_2d_impl<float><<<grid, block, 0, stream>>>(in_concat_size,
                                                             num_concats,
                                                             in_data,
                                                             concat_input_size,
                                                             out_concat_axis,
                                                             offset_concat_axis,
                                                             out_data);
    } else {
      int grid = (nthreads + BLOCK_SIZE - 1) / BLOCK_SIZE;
      concat_impl_cuda<float><<<grid, BLOCK_SIZE, 0, stream>>>(
          nthreads,
          in_data,
          num_concats,
          concat_input_size,
          out_concat_axis,
          in_concat_axis,
          offset_concat_axis,
          out_data);
    }
    offset_concat_axis += in_concat_axis;
  }
  param.Out->set_lod(param.X[0]->lod());
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(sequence_concat,
                     kCUDA,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::cuda::SequenceConcatCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .Finalize();
