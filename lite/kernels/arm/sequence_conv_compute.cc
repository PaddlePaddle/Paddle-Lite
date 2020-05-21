/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "lite/kernels/arm/sequence_conv_compute.h"
#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>
#include "lite/backends/arm/math/conv_impl.h"
#include "lite/backends/arm/math/sgemm.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/core/type_system.h"
#include "lite/operators/op_params.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

template <typename Dtype>
void local_naive_transpose(const Dtype* din, Dtype* dout, int m, int n) {
  int k = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      dout[k++] = din[j * n + i];
    }
  }
}

void SequenceConvCompute::PrepareForRun() {}

void SequenceConvCompute::Run() {
  // param.X is in shape: [sequence_len, hidden_dim];
  // param.Filter is in shape: [kernel_size * hidden_dim, kernel_num]
  // param.contextLength : kernel_size
  // param.contextStart: for padding idx
  // param.Out is in shape [new_sequence_len, kernel_num]
  auto& param = this->Param<operators::SequenceConvParam>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  const auto* in_data = param.X->data<float>();
  const auto* filter_data = param.Filter->data<float>();
  float* out_data = param.Out->mutable_data<float>();
  int pad_start = param.contextStart;
  int kernel_size = param.contextLength;
  int kernel_num = param.Filter->dims()[1];
  int up_pad = std::max(0, -pad_start);
  int down_pad = std::max(0, pad_start + kernel_size - 1);
  auto hidden_dim = static_cast<int64_t>(param.X->dims()[1]);
  auto sequence_len = static_cast<int64_t>(param.X->dims()[0]);
  auto lod = param.X->lod();

  // Im2Col
  lite::Tensor col;
  lite::Tensor tmp;
  col.Resize({sequence_len, kernel_size * hidden_dim});
  auto* col_data = col.mutable_data<float>();
  auto lod_level_0 = lod[0];
  int input_row_begin, input_row_end;
  for (int i = 0; i < static_cast<int>(lod_level_0.size()) - 1; i++) {
    if (lod_level_0[i] == lod_level_0[i + 1]) continue;
    input_row_begin = (pad_start > 0)
                          ? static_cast<int>(lod_level_0[i]) + pad_start
                          : static_cast<int>(lod_level_0[i]);
    input_row_end = static_cast<int>(lod_level_0[i + 1]);

    if (input_row_begin < input_row_end) {
      // do im2col
      auto* sub_in_data = in_data + input_row_begin * hidden_dim;
      auto* sub_col_data =
          col_data + input_row_begin * kernel_size * hidden_dim;
      tmp.Resize({kernel_size * hidden_dim, input_row_end - input_row_begin});
      auto* tmp_data = tmp.mutable_data<float>();
      // Image Col: [input_channels, filter_height, filter_width, output_height,
      // output_width]
      // sequence Col: [1, kernel_size, hidden_dim, sequence_len, 1]
      paddle::lite::arm::math::im2col(
          sub_in_data,
          1,
          sequence_len,
          hidden_dim,  // C H W -> 1, seq_len, hidden_dim
          kernel_size,
          hidden_dim,  // kernel_h, kernel_w
          up_pad,
          down_pad,
          0,
          0,  // pad_top, pad_bottom, pad_left, pad_right
          1,
          1,
          1,
          1,  // stride_h, stride_w, dilation_h, dilation_w
          tmp_data);
      local_naive_transpose(tmp_data,
                            sub_col_data,
                            kernel_size * hidden_dim,
                            input_row_end - input_row_begin);
    }
  }

  // SGDMM C := alpha * A * B + beta * C
  // matmul: col * filter_data
  // [sequence_len, kernel_size * hidden_dim] * [kernel_size * hidden_dim,
  // kernel_num]
  // = [sequence_len, kernel_num]
  paddle::lite::operators::ActivationParam act_param;
  paddle::lite::arm::math::sgemm(false,
                                 false,                     // is_transB,
                                 sequence_len,              // M
                                 kernel_num,                // N
                                 kernel_size * hidden_dim,  // K
                                 1.0f,                      // alpha
                                 col_data,                  // A
                                 kernel_size * hidden_dim,  // lda: k
                                 filter_data,               // B
                                 kernel_num,                // ldb: n
                                 0.f,                       // beta
                                 out_data,                  // C
                                 kernel_num,                // ldc: n
                                 NULL,                      // bias
                                 false,                     // is_bias
                                 act_param,                 // act_param
                                 &ctx);                     // ctx
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(sequence_conv,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::SequenceConvCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
