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

#include "lite/kernels/arm/grid_sampler_compute.h"
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void GridSamplerCompute::PrepareForRun() {}

void GridSamplerCompute::Run() {
  auto& param = this->Param<param_t>();
  auto n = param.x->dims()[0];
  auto c = param.x->dims()[1];
  auto h = param.x->dims()[2];
  auto w = param.x->dims()[3];
  const float* in = param.x->data<float>();
  const float* grid = param.grid->data<float>();
  float* out = param.out->mutable_data<float>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  const size_t coor_size = n * h * w;
  const size_t workspace_size = coor_size * 12 * sizeof(float);

  ctx.ExtendWorkspace(workspace_size);
  int32_t* coor_p = ctx.workspace_data<int>();
  float* dis_p = reinterpret_cast<float*>(coor_p) + coor_size * 4;
  uint32_t* bound_p = reinterpret_cast<uint32_t*>(dis_p) + coor_size * 4;

  float x_max = static_cast<float>(w - 1);
  float y_max = static_cast<float>(h - 1);
  float32x4_t vxmax = vdupq_n_f32(x_max);
  float32x4_t vymax = vdupq_n_f32(y_max);
  float32x4_t vone = vdupq_n_f32(1.f);
  float32x4_t vzero = vdupq_n_f32(0.f);

  // compute coor, dis, bound
  int i = coor_size;
  for (; i > 3; i -= 4) {
    float32x4x2_t xy = vld2q_f32(grid);
    float32x4_t grid_x = vmulq_n_f32(vaddq_f32(xy.val[0], vone), 0.5 * x_max);
    float32x4_t grid_y = vmulq_n_f32(vaddq_f32(xy.val[1], vone), 0.5 * y_max);
    grid += 8;

    // compute xw, we, yn, ys
    int32x4x4_t vcoor;
    vcoor.val[0] = vcvtq_s32_f32(grid_x);
    vcoor.val[2] = vcvtq_s32_f32(grid_y);
    float32x4_t vxwf = vcvtq_f32_s32(vcoor.val[0]);
    float32x4_t vynf = vcvtq_f32_s32(vcoor.val[2]);
    float32x4_t vxef = vaddq_f32(vxwf, vone);
    float32x4_t vysf = vaddq_f32(vynf, vone);
    vcoor.val[1] = vcvtq_s32_f32(vxef);
    vcoor.val[3] = vcvtq_s32_f32(vysf);
    vst4q_s32(coor_p, vcoor);
    coor_p += 16;

    // compute dw, dn ,de, ds
    float32x4x4_t vdis;
    vdis.val[0] = vsubq_f32(grid_x, vxwf);
    vdis.val[2] = vsubq_f32(grid_y, vynf);
    vdis.val[1] = vsubq_f32(vxef, grid_x);
    vdis.val[3] = vsubq_f32(vysf, grid_y);
    vst4q_f32(dis_p, vdis);
    dis_p += 16;

    // compute bound
    uint32x4x4_t vbound;
    uint32x4_t logic_xw =
        vorrq_u32(vcltq_f32(vxwf, vzero), vcgtq_f32(vxwf, vxmax));
    uint32x4_t logic_xe =
        vorrq_u32(vcltq_f32(vxef, vzero), vcgtq_f32(vxef, vxmax));
    uint32x4_t logic_yn =
        vorrq_u32(vcltq_f32(vynf, vzero), vcgtq_f32(vynf, vymax));
    uint32x4_t logic_ys =
        vorrq_u32(vcltq_f32(vysf, vzero), vcgtq_f32(vysf, vymax));
    vbound.val[0] = vmvnq_u32(vorrq_u32(logic_xw, logic_yn));
    vbound.val[1] = vmvnq_u32(vorrq_u32(logic_xe, logic_yn));
    vbound.val[2] = vmvnq_u32(vorrq_u32(logic_xw, logic_ys));
    vbound.val[3] = vmvnq_u32(vorrq_u32(logic_xe, logic_ys));
    vst4q_u32(bound_p, vbound);
    bound_p += 16;
  }

  for (; i > 0; i--) {
    float x = grid[0];
    float y = grid[1];
    float grid_x = (x + 1) * 0.5 * x_max;
    float grid_y = (y + 1) * 0.5 * y_max;
    grid += 2;

    // compute xw, xe, yn, ys
    int32_t xw = static_cast<int32_t>(floor(grid_x));
    int32_t xe = xw + 1;
    int32_t yn = static_cast<int32_t>(floor(grid_y));
    int32_t ys = yn + 1;
    *coor_p++ = xw;
    *coor_p++ = xe;
    *coor_p++ = yn;
    *coor_p++ = ys;

    // compute dw, de, dn, ds
    float dw = grid_x - xw;
    float de = xe - grid_x;
    float dn = grid_y - yn;
    float ds = ys - grid_y;
    *dis_p++ = dw;
    *dis_p++ = de;
    *dis_p++ = dn;
    *dis_p++ = ds;

    // compute bound
    bool logic_xw = (xw < 0.f || xw > x_max);
    bool logic_xe = (xe < 0.f || xe > x_max);
    bool logic_yn = (yn < 0.f || yn > y_max);
    bool logic_ys = (ys < 0.f || ys > y_max);
    *bound_p++ = ((logic_xw || logic_yn) ? 0 : 0xffffffff);
    *bound_p++ = ((logic_xe || logic_yn) ? 0 : 0xffffffff);
    *bound_p++ = ((logic_xw || logic_ys) ? 0 : 0xffffffff);
    *bound_p++ = ((logic_xe || logic_ys) ? 0 : 0xffffffff);
  }

  size_t cube_size = c * h * w;
  size_t spatial_size = h * w;
  // compute output
  for (int i = 0; i < n; ++i) {
    const float* in_n = in + i * cube_size;
    float* out_n = out + i * cube_size;
    int32_t* coor_n = ctx.workspace_data<int>() + i * spatial_size * 4;
    float* dis_n = reinterpret_cast<float*>(coor_n) + coor_size * 4;
    uint32_t* bound_n = reinterpret_cast<uint32_t*>(dis_n) + coor_size * 4;
#pragma omp parallel for
    for (int j = 0; j < c; ++j) {
      int32_t* coor_ptr = coor_n;
      float* dis_ptr = dis_n;
      uint32_t* bound_ptr = bound_n;
      const float* in_c = in_n + j * spatial_size;
      float* out_c = out_n + j * spatial_size;
      for (int k = 0; k < spatial_size; k++) {
        int32x4_t vcoor = vld1q_s32(coor_ptr);
        float32x4_t vdis = vld1q_f32(dis_ptr);
        int32_t xw = vgetq_lane_s32(vcoor, 0);
        int32_t xe = vgetq_lane_s32(vcoor, 1);
        int32_t yn = vgetq_lane_s32(vcoor, 2);
        int32_t ys = vgetq_lane_s32(vcoor, 3);

        uint32x4_t vbound = vld1q_u32(bound_ptr);
        float dw = vgetq_lane_f32(vdis, 0);
        float de = vgetq_lane_f32(vdis, 1);
        float dn = vgetq_lane_f32(vdis, 2);
        float ds = vgetq_lane_f32(vdis, 3);

        uint32_t wnbound = vgetq_lane_u32(vbound, 0);
        uint32_t enbound = vgetq_lane_u32(vbound, 1);
        uint32_t wsbound = vgetq_lane_u32(vbound, 2);
        uint32_t esbound = vgetq_lane_u32(vbound, 3);

        float in_wn = wnbound ? in_c[yn * w + xw] : 0.f;
        float in_en = enbound ? in_c[yn * w + xe] : 0.f;
        float in_ws = wsbound ? in_c[ys * w + xw] : 0.f;
        float in_es = esbound ? in_c[ys * w + xe] : 0.f;

        coor_ptr += 4;
        dis_ptr += 4;
        bound_ptr += 4;
        *out_c++ =
            ds * (in_wn * de + in_en * dw) + dn * (in_ws * de + in_es * dw);
      }
    }
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(grid_sampler,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::GridSamplerCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Grid", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
