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
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <typename T, PrecisionType PType>
void XPUBlockFuseCompute<T, PType>::PrepareForRun() {
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
  auto filter_ptr = param.filter->template data<int16_t>();
  auto bias_ptr = param.bias ? param.bias->template data<float>() : nullptr;
  auto max_filter_ptr = param.max_filter->template data<float>();

  int op_count = 0;
  int f_count = 0, s_count = 0, p_count = 0, d_count = 0, g_count = 0;
  int act_count = 0;
  for (int block_idx = 0; block_idx < block_lod.size(); block_idx++) {
    int cur_block_op_num = block_lod[block_idx];
    xdnn::fusion_block<float, int16_t, int16_t, T> cur_block;
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
          int r = cur_block.add_conv_layer(
              place_x[op_count],
              place_y[op_count],
              place_z[op_count],
              filter_ptr,
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
              max_filter_ptr,
              true,
              bias_ptr,
              act);
          CHECK_EQ(r, 0);
          filter_ptr = filter_ptr +
                       filter_dims[f_count] * filter_dims[f_count + 1] *
                           filter_dims[f_count + 2] * filter_dims[f_count + 3];
          max_filter_ptr = max_filter_ptr + 4;
          bias_ptr = bias_ptr + filter_dims[f_count];
          f_count += 4;
          s_count += 2;
          g_count += 1;
          p_count += 4;
          d_count += 2;
          op_count += 1;
          act_count += 1;
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

template <typename T, PrecisionType PType>
void XPUBlockFuseCompute<T, PType>::Run() {
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

  int r = xdnn::run_fusion_block_list<float, int16_t, int16_t, T>(
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
using XPUBlockFp32 = xpu::XPUBlockFuseCompute<float, PRECISION(kFloat)>;

using XPUBlockFp16 = xpu::XPUBlockFuseCompute<float16, PRECISION(kFP16)>;

REGISTER_LITE_KERNEL(
    __xpu__block_fuse_op, kXPU, kFloat, kNCHW, XPUBlockFp32, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("InputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("FilterMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("OutputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(
    __xpu__block_fuse_op, kXPU, kFP16, kNCHW, XPUBlockFp16, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("InputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("FilterMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("OutputMax", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
