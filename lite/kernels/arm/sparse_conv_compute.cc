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
#include "lite/backends/arm/math/conv_block_utils.h"
#include "lite/backends/arm/math/conv_impl.h"
#include "lite/backends/arm/math/sgemm.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/core/type_system.h"
#include "lite/operators/op_params.h"
#ifdef ENABLE_ARM_FP16
#include "lite/backends/arm/math/fp16/funcs_fp16.h"
#endif

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

template <typename Dtype>
void data_padding(const Dtype* data_in,
                  int up_pad,
                  int down_pad,
                  int width,
                  int hidden_dim,
                  int kernel_size,
                  Dtype* out,
                  int stride) {
  int len = hidden_dim;
  for (int i = 0; i <= (width + up_pad + down_pad) - kernel_size; i += stride) {
    for (int k = 0; k < kernel_size; k++) {
      int start = i + k * stride - up_pad;
      if (start < 0) {
        len = hidden_dim;
        while (len-- > 0) *(out++) = 0;
      } else if (start < width) {
        int in_s = start * hidden_dim;
        len = hidden_dim;
        while (len-- > 0) *(out++) = data_in[in_s++];
      } else {
        len = hidden_dim;
        while (len-- > 0) *(out++) = 0;
      }
    }
  }
}
template <>
void SequenceConvCompute<PRECISION(kFloat),
                         PRECISION(kFloat)>::PrepareForRun() {}

template <>
void SequenceConvCompute<PRECISION(kFloat), PRECISION(kFloat)>::Run() {
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
  memset(out_data, 0, sizeof(float) * param.Out->dims().production());
  int pad_start = param.contextStart;
  int kernel_size = param.contextLength;
  int stride = param.contextStride;
  int kernel_num = param.Filter->dims()[1];
  int up_pad = std::max(0, -pad_start);
  int down_pad = std::max(0, pad_start + kernel_size - 1);
  auto hidden_dim = static_cast<int64_t>(param.X->dims()[1]);
  auto sequence_len = static_cast<int64_t>(param.X->dims()[0]);
  auto lod = param.X->lod();
  lite::Tensor col;
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
      auto* sub_in_data = in_data + input_row_begin * hidden_dim;
      auto* sub_col_data =
          col_data + input_row_begin * kernel_size * hidden_dim;
      data_padding<float>(sub_in_data,
                          up_pad,
                          down_pad,
                          input_row_end - input_row_begin,
                          hidden_dim,
                          kernel_size,
                          sub_col_data,
                          stride);
    }
  }
  // SGDMM C := alpha * A * B + beta * C
  // matmul: col * filter_data
  // [sequence_len, kernel_size * hidden_dim] * [kernel_size * hidden_dim,
  // kernel_num]
  // = [sequence_len, kernel_num]
  auto m =
      static_cast<int>(lod_level_0[static_cast<int>(lod_level_0.size()) - 1]);
  paddle::lite::operators::ActivationParam act_param;
  paddle::lite::arm::math::sgemm(false,
                                 false,                     // is_transB,
                                 m,                         // M
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

#ifdef ENABLE_ARM_FP16
template <>
void SequenceConvCompute<PRECISION(kFP16), PRECISION(kFP16)>::PrepareForRun() {
  // when running op python unit_test, the weight dtype is float
  auto& param = this->Param<operators::SequenceConvParam>();
  auto filter_tensor = param.Filter;
  if (filter_tensor->precision() != PRECISION(kFP16)) {
    Tensor tmp_tensor;
    tmp_tensor.CopyDataFrom(*filter_tensor);
    filter_tensor->clear();
    filter_tensor->set_precision(PRECISION(kFP16));
    float16_t* fp_data = filter_tensor->mutable_data<float16_t>();
    const float* in_data = tmp_tensor.data<float>();
    lite::arm::math::fp16::fp32_to_fp16(
        in_data, fp_data, filter_tensor->numel());
  }
}

template <>
void SequenceConvCompute<PRECISION(kFP16), PRECISION(kFP16)>::Run() {
  // param.X is in shape: [sequence_len, hidden_dim];
  // param.Filter is in shape: [kernel_size * hidden_dim, kernel_num]
  // param.contextLength : kernel_size
  // param.contextStart: for padding idx
  // param.Out is in shape [new_sequence_len, kernel_num]
  auto& param = this->Param<operators::SequenceConvParam>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  const auto* in_data = param.X->data<float16_t>();
  const auto* filter_data = param.Filter->data<float16_t>();
  float16_t* out_data = param.Out->mutable_data<float16_t>();
  memset(out_data, 0, sizeof(float16_t) * param.Out->dims().production());
  int pad_start = param.contextStart;
  int kernel_size = param.contextLength;
  int stride = param.contextStride;
  int kernel_num = param.Filter->dims()[1];
  int up_pad = std::max(0, -pad_start);
  int down_pad = std::max(0, pad_start + kernel_size - 1);
  auto hidden_dim = static_cast<int64_t>(param.X->dims()[1]);
  auto sequence_len = static_cast<int64_t>(param.X->dims()[0]);
  auto lod = param.X->lod();
  lite::Tensor col;
  col.Resize({sequence_len, kernel_size * hidden_dim});
  auto* col_data = col.mutable_data<float16_t>();
  auto lod_level_0 = lod[0];
  int input_row_begin, input_row_end;
  for (int i = 0; i < static_cast<int>(lod_level_0.size()) - 1; i++) {
    if (lod_level_0[i] == lod_level_0[i + 1]) continue;
    input_row_begin = (pad_start > 0)
                          ? static_cast<int>(lod_level_0[i]) + pad_start
                          : static_cast<int>(lod_level_0[i]);
    input_row_end = static_cast<int>(lod_level_0[i + 1]);

    if (input_row_begin < input_row_end) {
      auto* sub_in_data = in_data + input_row_begin * hidden_dim;
      auto* sub_col_data =
          col_data + input_row_begin * kernel_size * hidden_dim;
      data_padding<float16_t>(sub_in_data,
                              up_pad,
                              down_pad,
                              input_row_end - input_row_begin,
                              hidden_dim,
                              kernel_size,
                              sub_col_data,
                              stride);
    }
  }
  // SGDMM C := alpha * A * B + beta * C
  // matmul: col * filter_data
  // [sequence_len, kernel_size * hidden_dim] * [kernel_size * hidden_dim,
  // kernel_num]
  // = [sequence_len, kernel_num]
  auto m =
      static_cast<int>(lod_level_0[static_cast<int>(lod_level_0.size()) - 1]);
  paddle::lite::operators::ActivationParam act_param;
  paddle::lite::arm::math::fp16::sgemm_fp16(false,
                                            false,       // is_transB,
                                            m,           // M
                                            kernel_num,  // N
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
                                            false,      // is_bias
                                            act_param,  // act_param
                                            &ctx);      // ctx
}
#endif

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
typedef paddle::lite::kernels::arm::SequenceConvCompute<PRECISION(kFloat),
                                                        PRECISION(kFloat)>
    SeqConvFp32;
#ifdef ENABLE_ARM_FP16
typedef paddle::lite::kernels::arm::SequenceConvCompute<PRECISION(kFP16),
                                                        PRECISION(kFP16)>
    SeqConvFp16;

REGISTER_LITE_KERNEL(sequence_conv, kARM, kFP16, kNCHW, SeqConvFp16, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindInput("Filter",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .Finalize();
#endif  // ENABLE_ARM_FP16

REGISTER_LITE_KERNEL(sequence_conv, kARM, kFloat, kNCHW, SeqConvFp32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindInput("Filter",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .Finalize();
