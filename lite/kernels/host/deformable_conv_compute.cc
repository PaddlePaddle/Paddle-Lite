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

#include "lite/kernels/host/deformable_conv_compute.h"

#include <vector>

#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"
#include "lite/kernels/host/deformable_conv_op.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

// todo: use blas if necessary
/**
 * naive row majored mat mul
 */
template <class T>
void MatMul(const Tensor& mat_a,
            const Tensor& mat_b,
            T alpha,
            Tensor* mat_out,
            T beta) {
  auto dim_a = mat_a.dims();
  auto dim_b = mat_b.dims();
  auto dim_out = mat_out->dims();

  int M = dim_out[0];
  int N = dim_out[1];
  int K = dim_a[1];
  auto* pA = mat_a.data<T>();
  auto* pB = mat_b.data<T>();
  auto* pC = mat_out->mutable_data<T>();
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      T sum = 0;
      for (int k = 0; k < K; ++k) {
        sum += pA[i * K + k] * pB[k * N + j];
      }
      pC[i * N + j] = sum * alpha + beta;
    }
  }
}

/**
 * @note this function is modified from paddle fluid
 * paddle commit id: f4c750d721a1226738bea382f6c0cf725cca8481
 *
 * check "paddle/fluid/operators/deformable_conv_op.h"
 * if necessary
 */
template <>
void DeformableConvComputeHost<PRECISION(kFloat), PRECISION(kFloat)>::Run() {
  const auto& param = this->Param<operators::DeformableConvParam>();

  // this implementation only support v2
  // to support v1, you could follow
  // "paddle/fluid/operators/deformable_conv_v1_op.h"

  const auto* input = param.x;
  const auto* offset = param.offset;
  const auto* mask = param.mask;
  const auto& filter = *param.conv_param.filter;
  auto* output = param.output;

  const int groups = param.conv_param.groups;
  const int deformable_groups = param.deformable_groups;
  const int im2col_step = param.im2col_step;
  const std::vector<int>& strides = param.conv_param.strides;
  const std::vector<int>& paddings = *param.conv_param.paddings;
  const std::vector<int>& dilations = *param.conv_param.dilations;

  const int batch_size = static_cast<int>(input->dims()[0]);

  std::vector<int64_t> filter_shape_vec(filter.dims().Vectorize());
  std::vector<int64_t> output_shape_vec(output->dims().Vectorize());

  // col_shape_vec: {c_i * k_h * k_w, im2col_step, o_h, o_w}
  std::vector<int64_t> col_buffer_shape_vec(filter_shape_vec.size());
  col_buffer_shape_vec[0] =
      input->dims()[1] * filter.dims()[2] * filter.dims()[3];
  col_buffer_shape_vec[1] = im2col_step;
  for (size_t j = 0; j < filter_shape_vec.size() - 2; ++j) {
    col_buffer_shape_vec[j + 2] = output_shape_vec[j + 2];
  }
  DDim col_shape(col_buffer_shape_vec);
  std::vector<int64_t> output_buffer_shape_vec(1);
  output_buffer_shape_vec[0] = batch_size * output_shape_vec[1] *
                               output_shape_vec[2] * output_shape_vec[3];
  DDim output_shape(output_buffer_shape_vec);
  Tensor col_buffer;
  Tensor output_buffer;
  col_buffer.Resize(col_shape);
  col_buffer.mutable_data<float>();
  output_buffer.Resize(output_shape);
  output_buffer.mutable_data<float>();
  int64_t M = output_shape_vec[1] / groups;
  int64_t N = im2col_step * output_shape_vec[2] * output_shape_vec[3];
  int64_t K =
      input->dims()[1] * filter_shape_vec[2] * filter_shape_vec[3] / groups;

  Tensor weight_3d;
  weight_3d.ShareDataWith(filter);
  weight_3d.Resize(DDim({groups, M, K}));
  Tensor col_buffer_3d;
  col_buffer_3d.ShareDataWith(col_buffer);
  col_buffer_3d.Resize(DDim({groups, K, N}));
  Tensor output_4d;
  output_4d.ShareDataWith(output_buffer);
  output_4d.Resize(DDim({batch_size / im2col_step, groups, M, N}));
  output_4d.mutable_data<float>();
  DDim input_shape = input->dims().Slice(1, input->dims().size());
  std::vector<int64_t> input_shape_vec = input_shape.Vectorize();
  int input_dim = input->numel() / input->dims()[0];
  int input_offset_dim = offset->numel() / offset->dims()[0];
  int input_mask_dim = mask->numel() / mask->dims()[0];
  const float* input_ptr = input->data<float>();
  const float* offset_ptr = offset->data<float>();
  const float* mask_ptr = mask->data<float>();
  col_buffer.mutable_data<float>();
  float* col_buffer_ptr = col_buffer.mutable_data<float>();
  for (int i = 0; i < batch_size / im2col_step; ++i) {
    ModulatedDeformableIm2colCPU<float>(
        input_ptr + i * im2col_step * input_dim,
        offset_ptr + i * im2col_step * input_offset_dim,
        mask_ptr + i * im2col_step * input_mask_dim,
        input_shape_vec,
        col_buffer_shape_vec,
        filter_shape_vec,
        paddings,
        strides,
        dilations,
        deformable_groups,
        col_buffer_ptr);
    Tensor output_3d = output_4d.Slice<float>(i, i + 1);
    output_3d.Resize(DDim(output_4d.dims()).Slice(1, output_4d.dims().size()));
    // get the product of pixel and weight
    for (int g = 0; g < groups; ++g) {
      Tensor weight_3d_slice = weight_3d.Slice<float>(g, g + 1);
      weight_3d_slice.Resize(
          DDim(weight_3d.dims()).Slice(1, weight_3d.dims().size()));
      Tensor col_buffer_3d_slice = col_buffer_3d.Slice<float>(g, g + 1);
      col_buffer_3d_slice.Resize(
          DDim(col_buffer_3d.dims()).Slice(1, col_buffer_3d.dims().size()));
      Tensor output_3d_slice = output_3d.Slice<float>(g, g + 1);
      output_3d_slice.Resize(
          DDim(output_3d.dims()).Slice(1, output_3d.dims().size()));
      MatMul<float>(
          weight_3d_slice, col_buffer_3d_slice, 1.0f, &output_3d_slice, 0.0f);
    }
  }
  output->ShareDataWith(output_buffer);
  output->Resize(DDim(output_shape_vec));
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using DeformableConvFp32Host =
    paddle::lite::kernels::host::DeformableConvComputeHost<PRECISION(kFloat),
                                                           PRECISION(kFloat)>;

REGISTER_LITE_KERNEL(
    deformable_conv, kHost, kFloat, kNCHW, DeformableConvFp32Host, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Mask", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Offset", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
