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

#include "lite/kernels/x86/conv_depthwise.h"
#include "lite/backends/x86/math/conv_depthwise_pack4.h"
#include "lite/backends/x86/math/conv_depthwise_pack8.h"
#include "lite/backends/x86/math/conv_utils.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

template <>
void DepthwiseConv<float>::Run() {
  auto& param = this->Param<param_t>();
  CHECK(this->ctx_);

  auto input_dims = param.x->dims();
  CHECK_EQ(input_dims.size(), 4UL);
  int batch_size = param.x->dims()[0];
  int input_channel = param.x->dims()[1];

  const int pack_size =
      input_channel % 8 == 0 ? 8 : input_channel % 4 == 0 ? 4 : 1;
  const int pack_num = input_channel / pack_size;

  if (pack_size == 8) {
    // lite::x86::math::pack8_m256(param.x, &input_pack_, pack_num, false);
    // lite::x86::math::padding8_m256(
    //    &input_pack_, &input_padding_, *(param.paddings));
    lite::x86::math::pack_padding8_m256(
        param.x, &input_padding_, pack_num, *(param.paddings));
  } else if (pack_size == 4) {
    lite::x86::math::pack4_m128(param.x, &input_pack_, pack_num, false);
    lite::x86::math::padding4_m128(
        &input_pack_, &input_padding_, *(param.paddings));
  } else {
    lite::x86::math::padding1_float(
        param.x, &input_padding_, *(param.paddings));
  }

  // filter [oc, ic/groups=1, kh, kw]
  auto filter_dims = param.filter->dims();
  CHECK_EQ(filter_dims.size(), 4UL);
  int kernel_h = param.filter->dims()[2];
  int kernel_w = param.filter->dims()[3];

  // filter [oc, 1, ih, iw] & pack_size=8 => [oc/8, ih, iw, 8]
  // filter [oc, 1, ih, iw] & pack_size=4 => [ic/4, ih, iw, 4]
  if (pack_size == 8) {
    lite::x86::math::pack8_m256(param.filter, &filter_pack_, pack_num, true);
  } else if (pack_size == 4) {
    lite::x86::math::pack4_m128(param.filter, &filter_pack_, pack_num, true);
  }

  // attributes
  const int stride_h = param.strides[0];
  const int stride_w = param.strides[1];
  const int dilation_h = (*param.dilations)[0];
  const int dilation_w = (*param.dilations)[1];

  // act type
  auto act_param = param.activation_param;
  bool has_act = act_param.has_active;
  auto act_type = act_param.active_type;

  // output [bs, oc, oh, ow]
  CHECK_EQ(param.output->dims().size(), 4UL);
  const int in_h = input_padding_.dims()[2], in_w = input_padding_.dims()[3];
  const int kernel_extend_h = dilation_h * (kernel_h - 1) + 1;
  const int kernel_extend_w = dilation_w * (kernel_w - 1) + 1;
  int output_height = (in_h - kernel_extend_h) / stride_h + 1;
  int output_width = (in_w - kernel_extend_w) / stride_w + 1;
  // output_trans [bs, oc/8, oh, ow, 8]
  // output_trans [bs, oc/4, oh, ow, 4]
  output_pack_.Resize(
      {batch_size, pack_num, output_height, output_width, pack_size});

  if (pack_size == 8) {
    if (kernel_h == 3 && kernel_w == 3 && stride_h == 1 && stride_w == 1 &&
        dilation_h == 1 && dilation_w == 1) {
      lite::x86::math::conv_depthwise_3x3s1_m256(&input_padding_,
                                                 &output_pack_,
                                                 &filter_pack_,
                                                 param.bias,
                                                 has_act,
                                                 act_type);
#ifdef LITE_WITH_PROFILE
      kernel_func_name_ = "conv_depthwise_3x3s1_m256";
#endif
    } else if (kernel_h == 3 && kernel_w == 3 && stride_h == 2 &&
               stride_w == 2 && dilation_h == 1 && dilation_w == 1) {
      lite::x86::math::conv_depthwise_3x3s2_m256(&input_padding_,
                                                 &output_pack_,
                                                 &filter_pack_,
                                                 param.bias,
                                                 has_act,
                                                 act_type);
#ifdef LITE_WITH_PROFILE
      kernel_func_name_ = "conv_depthwise_3x3s2_m256";
#endif
    } else {
      lite::x86::math::conv_depthwise_m256(&input_padding_,
                                           &output_pack_,
                                           &filter_pack_,
                                           param.bias,
                                           stride_h,
                                           stride_w,
                                           dilation_h,
                                           dilation_w,
                                           has_act,
                                           act_type);
#ifdef LITE_WITH_PROFILE
      kernel_func_name_ = "conv_depthwise_m256";
#endif
    }
  } else if (pack_size == 4) {
    lite::x86::math::conv_depthwise_m128(&input_padding_,
                                         &output_pack_,
                                         &filter_pack_,
                                         param.bias,
                                         stride_h,
                                         stride_w,
                                         dilation_h,
                                         dilation_w,
                                         has_act,
                                         act_type);
#ifdef LITE_WITH_PROFILE
    kernel_func_name_ = "conv_depthwise_m128";
#endif
  }

  // [bs, oh, ow, oc] => [bs, oc, oh, ow]
  if (pack_size == 8) {
    lite::x86::math::unpack8_m256(&output_pack_, param.output);
  } else if (pack_size == 4) {
    lite::x86::math::unpack4_m128(&output_pack_, param.output);
  }
}

#ifdef LITE_WITH_PROFILE
template <>
void DepthwiseConv<float>::SetProfileRuntimeKernelInfo(
    paddle::lite::profile::OpCharacter* ch) {
  ch->kernel_func_name = kernel_func_name_;
}
#endif

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
