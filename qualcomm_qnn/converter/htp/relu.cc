// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "driver/qualcomm_qnn/converter/htp/relu.h"
#include <algorithm>
#include <limits>

namespace nnadapter {
namespace qualcomm_qnn {
namespace htp {

/* execute functions for ops */
template <typename T_Ttype>
int ReluImpl(T_Ttype &out, const T_Ttype &in) {  // NOLINT
  debuglog("relu execute... dims=(%zdx%zdx%zdx%zd)",
           in.dim(0),
           in.dim(1),
           in.dim(2),
           in.dim(3));
  debuglog("in=%p out=%p", &in, &out);
  out.set_dims(in);
  for (Idx b = 0; b < in.dim(0); b++) {
    for (Idx h = 0; h < in.dim(1); h++) {
      for (Idx w = 0; w < in.dim(2); w++) {
        for (Idx d = 0; d < in.dim(3); d++) {
          float inval = in(b, h, w, d);
          out(b, h, w, d) = fmaxf(inval, 0.0f);
        }
      }
    }
  }
  return GraphStatus::Success;
}

template <typename T_TtypeI, typename T_TtypeX>
int ReluMinMaxImpl(T_TtypeI &out,  // NOLINT
                   const T_TtypeI &in,
                   const T_TtypeX &in_x,
                   const T_TtypeX &in_y) {
  debuglog("reluxy execute... dims=(%zdx%zdx%zdx%zd)",
           in.dim(0),
           in.dim(1),
           in.dim(2),
           in.dim(3));
  debuglog("in=%p out=%p", &in, &out);

  float x = in_x(0, 0, 0, 0);
  float y = in_y(0, 0, 0, 0);

  if (!(y > x)) {
    errlog("reluXY limit %f not > %f", x, y);
    return GraphStatus::ErrorFatal;
  }
  out.set_dims(in);

  bool no_scaling = false;

  const auto[b_in, h_in, w_in, d_in] = in.dims();

  if
    constexpr(!(std::is_same<Tensor, T_TtypeI>::value)) {  // NOLINT
      if (in.interface_scale() == out.interface_scale() &&
          in.interface_offset() == out.interface_offset()) {
        no_scaling = true;
      }
    }

  if (no_scaling) {
    static const float s_inf = std::numeric_limits<float>::infinity();
    static const int s_min_int = std::numeric_limits<int>::min();
    static const int s_max_int = std::numeric_limits<int>::max();

    if
      constexpr(std::is_base_of<LayoutCrouton_8, T_TtypeI>::value) {  //
        NOLINT
        const float out_step = out.interface_scale();
        const int out_zero_offset = out.interface_offset();
        size_t in_blocks = in.blocktab_len();
        auto in_block_tab = in.blocktab_ptr();
        auto out_block_tab = out.blocktab_ptr();
        int min_output = s_min_int;
        int max_output = s_max_int;
        if (x > -s_inf) {
          min_output = saturate_round<int>(x / out_step + out_zero_offset);
        }
        if (y < s_inf) {
          max_output = saturate_round<int>(y / out_step + out_zero_offset);
        }
        const int min_clip = std::max(static_cast<int>(min_output), 0);
        const int max_clip = std::min(static_cast<int>(max_output), 255);
        HVX_Vector vOmin = Q6_Vb_vsplat_R(min_clip);
        HVX_Vector vOmax = Q6_Vb_vsplat_R(max_clip);
        for (uint32_t i = 0; i < in_blocks; ++i) {
          auto in_vptr = (const HVX_Vector *)(in_block_tab[i]);
          auto out_vptr = reinterpret_cast<HVX_Vector *>(out_block_tab[i]);
          for (uint32_t j = 0; j < 16; ++j) {
            out_vptr[j] = Q6_Vub_vmin_VubVub(
                Q6_Vub_vmax_VubVub(in_vptr[j], vOmin), vOmax);
          }
        }
        return GraphStatus::Success;
      }
    else if  // NOLINT
      constexpr(std::is_base_of<LayoutFlat_8, T_TtypeI>::value) {
        uint8_t *outptr = &out.get_raw(0, 0, 0, 0);
        const uint8_t *inptr = &in.get_raw(0, 0, 0, 0);
        int32_t length = b_in * h_in * w_in * d_in;

        const float out_step = out.interface_scale();
        const int out_zero_offset = out.interface_offset();
        int min_output = s_min_int;
        int max_output = s_max_int;
        if (x > -s_inf) {
          min_output = saturate_round<int>(x / out_step + out_zero_offset);
        }
        if (y < s_inf) {
          max_output = saturate_round<int>(y / out_step + out_zero_offset);
        }
        const int min_clip = std::max(static_cast<int>(min_output), 0);
        const int max_clip = std::min(static_cast<int>(max_output), 255);
        HVX_Vector vOmin = Q6_Vb_vsplat_R(min_clip);
        HVX_Vector vOmax = Q6_Vb_vsplat_R(max_clip);

        int nvecs = length >> 7;
        int leftover = length & 127;

        bool use_unalign =
            (((size_t)inptr) & 0x7f) != 0 || (((size_t)outptr) & 0x7f) != 0;

        if (use_unalign) {
          for (int n = 0; n < nvecs; n++) {
            q6op_vstu_AV(outptr,
                         Q6_Vub_vmin_VubVub(
                             Q6_Vub_vmax_VubVub(vmemu(inptr), vOmin), vOmax));
            inptr += 128;
            outptr += 128;
          }
        } else {
          const HVX_Vector *vinptr = (const HVX_Vector *)inptr;
          HVX_Vector *voptr = reinterpret_cast<HVX_Vector *>(outptr);
          for (int n = 0; n < nvecs; n++) {
            *voptr++ =
                Q6_Vub_vmin_VubVub(Q6_Vub_vmax_VubVub(*vinptr++, vOmin), vOmax);
          }
          inptr = (const uint8_t *)vinptr;
          outptr = reinterpret_cast<uint8_t *>(voptr);
        }
        if (leftover) {
          q6op_vstu_variable_ARV(
              outptr,
              leftover,
              Q6_Vub_vmin_VubVub(Q6_Vub_vmax_VubVub(vmemu(inptr), vOmin),
                                 vOmax));
        }

        return GraphStatus::Success;
      }
  }

  warnlog("Reluxy using reference.... in: (%ld, %ld, %ld %ld) %s ",
          b_in,
          h_in,
          w_in,
          d_in,
          __PRETTY_FUNCTION__);

  // float fyiMax = 0.0f;
  for (Idx b = 0; b < b_in; b++) {
    for (Idx h = 0; h < h_in; h++) {
      for (Idx w = 0; w < w_in; w++) {
        for (Idx d = 0; d < d_in; d++) {
          float inval = in(b, h, w, d);
          out(b, h, w, d) = fminf(fmaxf(inval, x), y);
        }
      }
    }
  }
  return GraphStatus::Success;
}

inline size_t FlatToVlut(size_t index) {
  return ((index & 63) << 1) | ((index >> 6) & 1) | (index & -128);
}

void TablegenFuncImpl(TensorContiguous<Tdefs::QuantUint8> &out,  // NOLINT
                      const Tensor &in_step_size,
                      const Tensor &in_offset,
                      std::function<float(float)> const &f) {
  const float in_step_size_val = in_step_size(0, 0, 0, 0);
  const float in_offset_val = in_offset(0, 0, 0, 0);
  for (int i = 0; i < 256; i++) {
    /* Calculate what input equal to i would mean */
    float in_val = (i - in_offset_val) * in_step_size_val;
    out(0, 0, 0, FlatToVlut(i)) = f(in_val);
  }
}

GraphStatus ReluTablegenImpl(
    TensorContiguous<Tdefs::QuantUint8> &out,  // NOLINT
    const Tensor &in_step_size,
    const Tensor &in_offset,
    const Tensor &min,
    const Tensor &max) {
  float in_min = min(0, 0, 0, 0);
  float in_max = max(0, 0, 0, 0);
  TablegenFuncImpl(out, in_step_size, in_offset, [in_min, in_max](float x) {
    return fmaxf(in_min, fminf(x, in_max));
  });
  return GraphStatus::Success;
}

}  // namespace htp
}  // namespace qualcomm_qnn
}  // namespace nnadapter
