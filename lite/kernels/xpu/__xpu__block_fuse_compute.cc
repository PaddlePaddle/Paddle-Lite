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

#include "lite/kernels/xpu/__xpu__block_fuse_compute.h"
#include "lite/backends/xpu/math.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite_metal {
namespace kernels {
namespace xpu {

struct PackParam {
  int c;
  int f;
  int kh;
  int kw;
  std::vector<int> s;
  std::vector<int> p;
  std::vector<int> d;
  int g;
  std::vector<xdnn::Activation_t> acts;
  const void* w;
  const void* w_max;
  const void* bias;
};

template <typename T>
bool QuantBlockFilter(const float* cpu_w,
                      T* xpu_quant_w,
                      float* xpu_wmax,
                      const int64_t len) {
  return false;
}

template <>
bool QuantBlockFilter<int16_t>(const float* cpu_w,
                               int16_t* xpu_quant_w,
                               float* xpu_wmax,
                               const int64_t len) {
  float max_f = paddle::lite_metal::xpu::math::FindMaxAbs(cpu_w, len);
  std::vector<float> max_f_v(4, max_f);
  XPU_CALL(xpu_memcpy(xpu_wmax,
                      max_f_v.data(),
                      4 * sizeof(float),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));
  std::vector<int16_t> cpu_quant_w(len, 0);
  paddle::lite_metal::xpu::math::ConvertFP32ToInt16(
      cpu_w, cpu_quant_w.data(), max_f, len);
  XPU_CALL(xpu_memcpy(xpu_quant_w,
                      cpu_quant_w.data(),
                      len * sizeof(int16_t),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));
  return true;
}

template <>
bool QuantBlockFilter<int8_t>(const float* cpu_w,
                              int8_t* xpu_quant_w,
                              float* xpu_wmax,
                              const int64_t len) {
  float max_f = paddle::lite_metal::xpu::math::FindMaxAbs(cpu_w, len);
  std::vector<float> max_f_v(4, max_f);
  XPU_CALL(xpu_memcpy(xpu_wmax,
                      max_f_v.data(),
                      4 * sizeof(float),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));
  std::vector<int8_t> cpu_quant_w(len, 0);
  paddle::lite_metal::xpu::math::ConvertFP32ToInt8(
      cpu_w, cpu_quant_w.data(), max_f, len);
  XPU_CALL(xpu_memcpy(xpu_quant_w,
                      cpu_quant_w.data(),
                      len * sizeof(int8_t),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));
  return true;
}

std::vector<const float*> GenerateBiasVector(
    const float* encode_bias_ptr,
    const int encode_bias_len,
    const std::vector<int>& op_type,
    const std::vector<int>& filter_dims,
    const std::vector<int>& conv_bias) {
  std::vector<const float*> bias_v;
  int conv_idx = 0;
  int filter_idx = 0;
  for (int op_idx = 0; op_idx < op_type.size(); op_idx++) {
    if (op_type[op_idx] == 0 && conv_idx < conv_bias.size()) {
      if (conv_bias[conv_idx] > 0) {
        bias_v.push_back(encode_bias_ptr);
        encode_bias_ptr += filter_dims[filter_idx];
      } else {
        bias_v.push_back(nullptr);
      }
      filter_idx += 4;
      conv_idx += 1;
    } else if (op_type[op_idx] <= 3) {
      filter_idx += 2;
    } else if (op_type[op_idx] == 4 && conv_idx < conv_bias.size()) {
      if (conv_bias[conv_idx] > 0) {
        bias_v.push_back(encode_bias_ptr);
        encode_bias_ptr +=
            filter_dims[filter_idx + 1] / filter_dims[filter_idx] +
            filter_dims[filter_idx + 1];
      } else {
        bias_v.push_back(nullptr);
      }
      conv_idx += 1;
      filter_idx += 2;
    }
  }
  return bias_v;
}

template <typename T>
void GenerateOpWeight(const float* cpu_w,
                      T* xpu_w,
                      float* xpu_wmax,
                      const std::vector<int>& op_type,
                      const std::vector<int>& filter_dims) {
  int filter_idx = 0;
  for (int op_idx = 0; op_idx < op_type.size(); op_idx++) {
    if (op_type[op_idx] == 0) {
      int filter_len = filter_dims[filter_idx] * filter_dims[filter_idx + 1] *
                       filter_dims[filter_idx + 2] *
                       filter_dims[filter_idx + 3];
      QuantBlockFilter<T>(cpu_w, xpu_w, xpu_wmax, filter_len);
      cpu_w += filter_len;
      xpu_w += filter_len;
      xpu_wmax += 4;
      filter_idx += 4;
    } else if (op_type[op_idx] <= 3) {
      filter_idx += 2;
    } else if (op_type[op_idx] == 4) {
      int filter_len = filter_dims[filter_idx + 1] *
                       filter_dims[filter_idx + 1] / filter_dims[filter_idx];
      QuantBlockFilter<T>(cpu_w, xpu_w, xpu_wmax, filter_len);
      cpu_w += filter_len;
      xpu_w += filter_len;
      xpu_wmax += 4;
      QuantBlockFilter<T>(cpu_w, xpu_w, xpu_wmax, filter_len);
      cpu_w += filter_len;
      xpu_w += filter_len;
      xpu_wmax += 4;
      filter_idx += 2;
    }
  }
}

template <typename T>
std::vector<PackParam> GenerateOpAttr(const T* w_ptr,
                                      const float* wmax_ptr,
                                      const std::vector<const float*> bias,
                                      const std::vector<int>& op_type,
                                      const std::vector<int>& filter_dims,
                                      const std::vector<int>& strides,
                                      const std::vector<int>& paddings,
                                      const std::vector<int>& dilations,
                                      const std::vector<int> groups,
                                      const std::vector<int>& act_type,
                                      const std::vector<float>& act_param) {
  for (auto i = 0; i < dilations.size(); i++) {
  }
  std::vector<PackParam> res;
  std::vector<int>::const_iterator f_iter = filter_dims.begin();
  std::vector<int>::const_iterator s_iter = strides.begin();
  std::vector<int>::const_iterator p_iter = paddings.begin();
  std::vector<int>::const_iterator d_iter = dilations.begin();
  std::vector<int>::const_iterator g_iter = groups.begin();
  std::vector<int>::const_iterator act_iter = act_type.begin();
  std::vector<float>::const_iterator actp_iter = act_param.begin();
  std::vector<const float*>::const_iterator b_iter = bias.begin();
  for (int op_idx = 0; op_idx < op_type.size(); op_idx++) {
    if (op_type[op_idx] == 0) {
      xdnn::Activation_t act((xdnn::Activation_t::act_enum)(*act_iter));
      if (*act_iter == 5) {
        act.leaky_alpha = *actp_iter;
      } else if (*act_iter == 15) {
        act.hard_sigmoid_slope = *actp_iter;
      }
      res.push_back({*(f_iter + 1) * (*g_iter),
                     *f_iter,
                     *(f_iter + 2),
                     *(f_iter + 3),
                     {*s_iter, *(s_iter + 1)},
                     {*p_iter, *(p_iter + 1), *(p_iter + 2), *(p_iter + 3)},
                     {*d_iter, *(d_iter + 1)},
                     *g_iter,
                     {act},
                     w_ptr,
                     wmax_ptr,
                     *b_iter});

      w_ptr +=
          (*(f_iter)) * (*(f_iter + 1)) * (*(f_iter + 2)) * (*(f_iter + 3));
      wmax_ptr += 4;
      f_iter += 4;
      s_iter += 2;
      p_iter += 4;
      d_iter += 2;
      g_iter += 1;
      act_iter += 1;
      actp_iter += 1;
      b_iter += 1;
    } else if (op_type[op_idx] <= 3) {
      res.push_back({-1,
                     -1,
                     *f_iter,
                     *(f_iter + 1),
                     {*s_iter, *(s_iter + 1)},
                     {*p_iter, *(p_iter + 1), *(p_iter + 2), *(p_iter + 3)},
                     {},
                     -1,
                     {},
                     nullptr,
                     nullptr,
                     nullptr});
      f_iter += 2;
      s_iter += 2;
      p_iter += 4;
    } else if (op_type[op_idx] == 4) {
      std::vector<xdnn::Activation_t> cur_acts;
      for (int i = 0; i < 3; i++) {
        xdnn::Activation_t act((xdnn::Activation_t::act_enum)(*(act_iter + i)));
        if (*(act_iter + i) == 5) {
          act.leaky_alpha = *(actp_iter + i);
        } else if (*(act_iter + i) == 15) {
          act.hard_sigmoid_slope = *(actp_iter + i);
        }
        cur_acts.push_back(act);
      }
      res.push_back({*(f_iter + 1),
                     *f_iter,
                     -1,
                     -1,
                     {},
                     {},
                     {},
                     -1,
                     cur_acts,
                     w_ptr,
                     wmax_ptr,
                     *b_iter});
      w_ptr += (*(f_iter + 1)) * (*(f_iter + 1)) / (*f_iter) * 2;
      wmax_ptr += 8;
      f_iter += 2;
      act_iter += 3;
      actp_iter += 3;
      b_iter += 1;
    } else if (op_type[op_idx] == 10) {
      xdnn::Activation_t act((xdnn::Activation_t::act_enum)(*act_iter));
      if (*act_iter == 5) {
        act.leaky_alpha = *actp_iter;
      } else if (*act_iter == 15) {
        act.hard_sigmoid_slope = *actp_iter;
      }
      res.push_back(
          {-1, -1, -1, -1, {}, {}, {}, -1, {act}, nullptr, nullptr, nullptr});
      act_iter += 1;
      actp_iter += 1;
    } else if (op_type[op_idx] == 20) {
      res.push_back(
          {-1, -1, -1, -1, {}, {}, {}, -1, {}, nullptr, nullptr, nullptr});
    } else {
      LOG(FATAL) << "Unsupport layer type" << op_type[op_idx];
    }
  }
  return res;
}

template <typename TM, typename TW, PrecisionType PType>
void XPUBlockFuseCompute<TM, TW, PType>::PrepareForRun() {
  auto& param = this->template Param<param_t>();
  auto op_type = param.op_type;
  auto place_x = param.place_x;
  auto place_y = param.place_y;
  auto place_z = param.place_z;
  auto& block_lod = param.block_lod;
  // filter
  auto cpu_filter_ptr = param.filter->template data<float>();
  auto filter_len = param.filter->numel();
  // quant_filter
  quant_filter_guard =
      TargetWrapperXPU::MallocScratchPad(filter_len * sizeof(TW));
  quant_filter = reinterpret_cast<TW*>(quant_filter_guard->addr_);
  // max
  int64_t filter_max_len = 0;
  for (auto op_idx = 0; op_idx < op_type.size(); op_idx++) {
    if (op_type[op_idx] == 0) {
      filter_max_len += 4;
    } else if (op_type[op_idx] == 4) {
      filter_max_len += 8;
    }
  }
  filter_max_guard =
      TargetWrapperXPU::MallocScratchPad(filter_max_len * sizeof(float));
  filter_max = reinterpret_cast<float*>(filter_max_guard->addr_);
  GenerateOpWeight<TW>(
      cpu_filter_ptr, quant_filter, filter_max, op_type, param.filter_dims);
  // bias
  auto bias_ptr = param.has_bias ? param.bias->template data<float>() : nullptr;
  int encode_bias_len = param.has_bias ? param.bias->numel() : 0;
  auto bias_v = GenerateBiasVector(
      bias_ptr, encode_bias_len, op_type, param.filter_dims, param.conv_bias);
  auto pack_param = GenerateOpAttr(quant_filter,
                                   filter_max,
                                   bias_v,
                                   op_type,
                                   param.filter_dims,
                                   param.strides,
                                   *param.paddings,
                                   *param.dilations,
                                   param.groups,
                                   param.act_type,
                                   param.act_param);

  int op_cnt = 0;
  for (int block_idx = 0; block_idx < block_lod.size(); block_idx++) {
    int cur_block_op_num = block_lod[block_idx];
    xdnn::fusion_block<float, TW, TW, TM> cur_block;
    for (int op_idx = 0; op_idx < cur_block_op_num; op_idx++) {
      switch (op_type[op_cnt]) {
        case 0: {
          int r = cur_block.add_conv_layer(
              place_x[op_cnt],
              place_y[op_cnt],
              place_z[op_cnt],
              static_cast<const TW*>(pack_param[op_cnt].w),
              pack_param[op_cnt].c,
              pack_param[op_cnt].f,
              {pack_param[op_cnt].kh, pack_param[op_cnt].kw},
              pack_param[op_cnt].s,
              pack_param[op_cnt].p,
              pack_param[op_cnt].d,
              pack_param[op_cnt].g,
              static_cast<const float*>(pack_param[op_cnt].w_max),
              true,
              static_cast<const float*>(pack_param[op_cnt].bias),
              pack_param[op_cnt].acts[0]);
          CHECK_EQ(r, 0);
          break;
        }
        case 1: {
          int r = cur_block.add_avg_pool2d_layer(
              place_x[op_cnt],
              place_z[op_cnt],
              {pack_param[op_cnt].kh, pack_param[op_cnt].kw},
              pack_param[op_cnt].s,
              pack_param[op_cnt].p,
              true,
              true);
          CHECK_EQ(r, 0);
          break;
        }
        case 2: {
          int r = cur_block.add_avg_pool2d_layer(
              place_x[op_cnt],
              place_z[op_cnt],
              {pack_param[op_cnt].kh, pack_param[op_cnt].kw},
              pack_param[op_cnt].s,
              pack_param[op_cnt].p,
              false,
              true);
          CHECK_EQ(r, 0);
          break;
        }
        case 3: {
          int r = cur_block.add_max_pool2d_layer(
              place_x[op_cnt],
              place_z[op_cnt],
              {pack_param[op_cnt].kh, pack_param[op_cnt].kw},
              pack_param[op_cnt].s,
              pack_param[op_cnt].p,
              true);
          CHECK_EQ(r, 0);
          break;
        }
        case 4: {
          int r = cur_block.add_se_layer(
              place_x[op_cnt],
              place_y[op_cnt],
              place_z[op_cnt],
              static_cast<const TW*>(pack_param[op_cnt].w),
              pack_param[op_cnt].c,
              pack_param[op_cnt].f,
              static_cast<const float*>(pack_param[op_cnt].w_max),
              static_cast<const float*>(pack_param[op_cnt].bias),
              pack_param[op_cnt].acts[0],
              pack_param[op_cnt].acts[1],
              pack_param[op_cnt].acts[2],
              true);
          CHECK_EQ(r, 0);
          break;
        }
        case 10: {
          int r = cur_block.add_ew_layer(place_x[op_cnt],
                                         place_y[op_cnt],
                                         place_z[op_cnt],
                                         pack_param[op_cnt].acts[0]);
          CHECK_EQ(r, 0);
          break;
        }
        case 20: {
          int r = cur_block.add_concat_layer(
              place_x[op_cnt], place_y[op_cnt], place_z[op_cnt], 1);
          CHECK_EQ(r, 0);
          break;
        }
        default: { LOG(FATAL) << "Unsupport layer type" << op_type[op_cnt]; }
      }
      op_cnt += 1;
    }
    xpu_fusion_block.push_back(cur_block);
  }
}

template <typename TM, typename TW, PrecisionType PType>
void XPUBlockFuseCompute<TM, TW, PType>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto& input_dims = param.input->dims();
  int n = static_cast<int>(input_dims[0]);
  int c = static_cast<int>(input_dims[1]);
  int h = static_cast<int>(input_dims[2]);
  int w = static_cast<int>(input_dims[3]);

  std::vector<float*> feature_list(xpu_fusion_block.size() - 1, nullptr);
  int f = c, yh = h, yw = w;
  for (int block_idx = 0; block_idx < xpu_fusion_block.size(); block_idx++) {
    int cur_block_c = f;
    int cur_block_h = yh;
    int cur_block_w = yw;
    int r = xpu_fusion_block[block_idx].infer_shape(
        n, cur_block_c, cur_block_h, cur_block_w, &f, &yh, &yw);
    CHECK_EQ(r, 0);
  }
  auto output = param.output;
  output->Resize({n, f, yh, yw});
  const float* input_max =
      param.input_max ? (param.input_max->template data<float>()) : nullptr;
  float* output_max =
      param.output_max->template mutable_data<float>(TARGET(kXPU));

  int r = xdnn::run_fusion_block_list<float, TW, TW, TM>(
      ctx.GetRawContext(),
      param.input->template data<float>(),
      output->template mutable_data<float>(TARGET(kXPU)),
      input_max,
      output_max,
      n,
      c,
      h,
      w,
      xpu_fusion_block,
      feature_list);

  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

namespace xpu = paddle::lite_metal::kernels::xpu;
using XPUBlockFp32 =
    xpu::XPUBlockFuseCompute<float, int16_t, PRECISION(kFloat)>;

using XPUBlockFp16 =
    xpu::XPUBlockFuseCompute<float16, int16_t, PRECISION(kFP16)>;

using XPUBlockInt8 = xpu::XPUBlockFuseCompute<float, int8_t, PRECISION(kInt8)>;

REGISTER_LITE_KERNEL(
    __xpu__block_fuse_op, kXPU, kFloat, kNCHW, XPUBlockFp32, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("InputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("OutputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(
    __xpu__block_fuse_op, kXPU, kFP16, kNCHW, XPUBlockFp16, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("InputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("OutputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(
    __xpu__block_fuse_op, kXPU, kInt8, kNCHW, XPUBlockInt8, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("InputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("OutputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
