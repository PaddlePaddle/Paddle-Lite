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
namespace lite {
namespace kernels {
namespace xpu {

template <typename T>
bool QuantBlockFilter(const float* filter_on_host,
                      T* quant_res,
                      float max,
                      int64_t len) {
  return false;
}

template <>
bool QuantBlockFilter<int16_t>(const float* filter_on_host,
                               int16_t* quant_res,
                               float max,
                               int64_t len) {
  paddle::lite::xpu::math::ConvertFP32ToInt16(
      filter_on_host, quant_res, max, len);
  return true;
}

template <>
bool QuantBlockFilter<int8_t>(const float* filter_on_host,
                              int8_t* quant_res,
                              float max,
                              int64_t len) {
  paddle::lite::xpu::math::ConvertFP32ToInt8(
      filter_on_host, quant_res, max, len);
  return true;
}

template <typename TM, typename TW, PrecisionType PType>
void XPUBlockFuseCompute<TM, TW, PType>::PrepareForRun() {
  auto& param = this->template Param<param_t>();
  auto op_type = param.op_type;

  auto place_x = param.place_x;
  auto place_y = param.place_y;
  auto place_z = param.place_z;

  auto& filter_dims = param.filter_dims;
  auto& strides = param.strides;
  auto& paddings = param.paddings;
  auto& dilations = param.dilations;
  auto& groups = param.groups;
  auto& act_type = param.act_type;
  auto& act_param = param.act_param;
  auto& block_lod = param.block_lod;
  auto& conv_bias = param.conv_bias;
  size_t conv_num = 0;

  for (auto i = 0; i < op_type.size(); i++) {
    conv_num += op_type[i] == 0 ? 1 : 0;
  }
  // filter
  auto cpu_filter_ptr = param.filter->template data<float>();
  auto filter_len = param.filter->numel();
  // max
  filter_max_guard =
      TargetWrapperXPU::MallocScratchPad(4 * conv_num * sizeof(float), false);
  filter_max = reinterpret_cast<float*>(filter_max_guard->addr_);
  // quant_filter
  quant_filter_guard =
      TargetWrapperXPU::MallocScratchPad(filter_len * sizeof(TW), false);
  quant_filter = reinterpret_cast<TW*>(quant_filter_guard->addr_);
  // bias
  auto bias_ptr = param.has_bias ? param.bias->template data<float>() : nullptr;
  auto xpu_filter_max = filter_max;
  auto xpu_quant_filter = quant_filter;

  int op_count = 0;
  int f_count = 0, s_count = 0, p_count = 0, d_count = 0, g_count = 0;
  int act_count = 0, bias_count = 0;
  for (int block_idx = 0; block_idx < block_lod.size(); block_idx++) {
    int cur_block_op_num = block_lod[block_idx];
    xdnn::fusion_block<float, TW, TW, TM> cur_block;
    for (int op_idx = 0; op_idx < cur_block_op_num; op_idx++) {
      switch (op_type[op_count]) {
        case 0: {
          xdnn::Activation_t act(
              (xdnn::Activation_t::act_enum)act_type[act_count]);
          if (act_type[act_count] == 5) {
            act.leaky_alpha = act_param[act_count];
          } else if (act_type[act_count] == 15) {
            act.hard_sigmoid_slope = act_param[act_count];
          }
          int cur_filter_len = filter_dims[f_count] * filter_dims[f_count + 1] *
                               filter_dims[f_count + 2] *
                               filter_dims[f_count + 3];
          float max_f = paddle::lite::xpu::math::FindMaxAbs(cpu_filter_ptr,
                                                            cur_filter_len);
          std::vector<float> max_f_v(4, max_f);
          XPU_CALL(xpu_memcpy(xpu_filter_max,
                              max_f_v.data(),
                              4 * sizeof(float),
                              XPUMemcpyKind::XPU_HOST_TO_DEVICE));

          std::vector<TW> quant_filter_cpu(cur_filter_len, 0);
          bool ret = QuantBlockFilter<TW>(
              cpu_filter_ptr, quant_filter_cpu.data(), max_f, cur_filter_len);
          CHECK_EQ(ret, true);
          XPU_CALL(xpu_memcpy(xpu_quant_filter,
                              quant_filter_cpu.data(),
                              cur_filter_len * sizeof(TW),
                              XPUMemcpyKind::XPU_HOST_TO_DEVICE));

          int r = cur_block.add_conv_layer(
              place_x[op_count],
              place_y[op_count],
              place_z[op_count],
              xpu_quant_filter,
              filter_dims[f_count + 1] * groups[g_count],
              filter_dims[f_count],
              {filter_dims[f_count + 2], filter_dims[f_count + 3]},
              {strides[s_count], strides[s_count + 1]},
              {paddings[p_count],
               paddings[p_count + 1],
               paddings[p_count + 2],
               paddings[p_count + 3]},
              {dilations[d_count], dilations[d_count + 1]},
              groups[g_count],
              xpu_filter_max,
              true,
              bias_ptr,
              act);
          CHECK_EQ(r, 0);
          cpu_filter_ptr += cur_filter_len;
          xpu_quant_filter += cur_filter_len;
          xpu_filter_max += 4;
          if (conv_bias[bias_count] != 0) {
            bias_ptr = bias_ptr + filter_dims[f_count];
          }
          f_count += 4;
          s_count += 2;
          g_count += 1;
          p_count += 4;
          d_count += 2;
          op_count += 1;
          act_count += 1;
          bias_count += 1;
          break;
        }
        case 1: {
          int r = cur_block.add_avg_pool2d_layer(
              place_x[op_count],
              place_z[op_count],
              {filter_dims[f_count], filter_dims[f_count + 1]},
              {strides[s_count], strides[s_count + 1]},
              {paddings[p_count],
               paddings[p_count + 1],
               paddings[p_count + 2],
               paddings[p_count + 3]},
              true,
              true);
          CHECK_EQ(r, 0);
          f_count += 2;
          s_count += 2;
          p_count += 4;
          op_count += 1;
          break;
        }
        case 2: {
          int r = cur_block.add_avg_pool2d_layer(
              place_x[op_count],
              place_z[op_count],
              {filter_dims[f_count], filter_dims[f_count + 1]},
              {strides[s_count], strides[s_count + 1]},
              {paddings[p_count],
               paddings[p_count + 1],
               paddings[p_count + 2],
               paddings[p_count + 3]},
              false,
              true);
          CHECK_EQ(r, 0);
          f_count += 2;
          s_count += 2;
          p_count += 4;
          op_count += 1;
          break;
        }
        case 3: {
          int r = cur_block.add_max_pool2d_layer(
              place_x[op_count],
              place_z[op_count],
              {filter_dims[f_count], filter_dims[f_count + 1]},
              {strides[s_count], strides[s_count + 1]},
              {paddings[p_count],
               paddings[p_count + 1],
               paddings[p_count + 2],
               paddings[p_count + 3]},
              true);
          CHECK_EQ(r, 0);
          f_count += 2;
          s_count += 2;
          p_count += 4;
          op_count += 1;
          break;
        }
        case 20: {
          int r = cur_block.add_concat_layer(
              place_x[op_count], place_y[op_count], place_z[op_count], 1);
          CHECK_EQ(r, 0);
          op_count += 1;
          break;
        }
        default: { LOG(FATAL) << "Unsupport layer type" << op_type[op_count]; }
      }
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

namespace xpu = paddle::lite::kernels::xpu;
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
