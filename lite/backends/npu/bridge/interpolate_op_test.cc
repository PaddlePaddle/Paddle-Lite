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

#include "lite/operators/interpolate_op.h"
#include <gtest/gtest.h>
#include <random>
#include "lite/backends/npu/bridge/registry.h"
#include "lite/backends/npu/bridge/test_helper.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace npu {
namespace bridge {

template <typename DType>
void bilinear_interp_ref(const std::shared_ptr<operators::InterpolateOp> op) {
  auto scope = op->scope();
  auto op_info = op->op_info();
  auto x = scope->FindVar(op_info->Input("X").front())->GetMutable<Tensor>();
  auto out =
      scope->FindVar(op_info->Output("Out").front())->GetMutable<Tensor>();
  auto x_dims = x->dims();
  int batch_size = x_dims[0];
  int channel_size = x_dims[1];
  auto x_h = x_dims[2];
  auto x_w = x_dims[3];
  CHECK_EQ(x_dims.size(), 4);
  auto scale = op_info->GetAttr<float>("scale");
  auto out_w = op_info->GetAttr<int>("out_w");
  auto out_h = op_info->GetAttr<int>("out_h");
  auto align_corners = op_info->GetAttr<bool>("align_corners");
  int align_mode = op_info->GetAttr<int>("align_mode");
  auto interp_method = op_info->GetAttr<std::string>("interp_method");

  // calc real out_h and out_w
  if (scale > 0) {
    out_h = static_cast<int>(x_h * scale);
    out_w = static_cast<int>(x_w * scale);
  }
  if (op_info->HasInput("OutSize")) {
    auto out_size_var_names = op_info->Input("OutSize");
    if (out_size_var_names.size() > 0) {
      auto out_size_var_name = out_size_var_names.front();
      auto out_size =
          scope->FindVar(out_size_var_name)->GetMutable<lite::Tensor>();
      auto out_size_dims = out_size->dims();
      CHECK_EQ(out_size_dims.size(), 1);
      CHECK_EQ(out_size_dims.production(), 2);
      auto out_size_data = out_size->mutable_data<int>();
      out_h = out_size_data[0];
      out_w = out_size_data[1];
    }
  }
  CHECK_GT(out_h, 0);
  CHECK_GT(out_w, 0);
  out->Resize({batch_size, channel_size, out_h, out_w});

  // copy from x if no change
  if (x_h == out_h && x_w == out_w) {
    out->CopyDataFrom(*x);
    return;
  }

  float ratio_h = 0.f;
  float ratio_w = 0.f;
  if (out_h > 1) {
    ratio_h = (align_corners) ? static_cast<float>(x_h - 1) / (out_h - 1)
                              : static_cast<float>(x_h) / out_h;
  }
  if (out_w > 1) {
    ratio_w = (align_corners) ? static_cast<float>(x_w - 1) / (out_w - 1)
                              : static_cast<float>(x_w) / out_w;
  }

  // naive bilinear interpolation
  auto x_data = x->mutable_data<DType>();
  auto out_data = out->mutable_data<DType>();
  bool align_flag = (align_mode == 0 && !align_corners);

  std::vector<int> vy_n, vy_s;
  std::vector<float> vd_n, vd_s;
  vy_n.reserve(out_h);
  vy_s.reserve(out_h);
  vd_n.reserve(out_h);
  vd_s.reserve(out_h);
  for (int k = 0; k < out_h; k++) {
    int yn = align_flag ? static_cast<int>(ratio_h * (k + 0.5) - 0.5)
                        : static_cast<int>(ratio_h * k);
    yn = (yn > 0) ? yn : 0;
    int ys = (yn + 1) < (x_h - 1) ? (yn + 1) : (x_h - 1);
    float idx_src_y = ratio_h * (k + 0.5) - 0.5;
    idx_src_y = (idx_src_y > 0) ? idx_src_y : 0;
    float dn = align_flag ? idx_src_y - yn : ratio_h * k - yn;
    float ds = 1.f - dn;
    {
      vy_n[k] = yn;
      vy_s[k] = ys;
      vd_n[k] = dn;
      vd_s[k] = ds;
    }
  }

  std::vector<int> vx_w, vx_e;
  std::vector<float> vd_w, vd_e;
  vx_w.reserve(out_w);
  vx_e.reserve(out_w);
  vd_w.reserve(out_w);
  vd_e.reserve(out_w);
  for (int l = 0; l < out_w; l++) {
    int xw = (align_mode == 0 && !align_corners)
                 ? static_cast<int>(ratio_w * (l + 0.5) - 0.5)
                 : static_cast<int>(ratio_w * l);
    xw = (xw > 0) ? xw : 0;
    int xe = (xw + 1) < (x_w - 1) ? (xw + 1) : (x_w - 1);
    float idx_src_x = ratio_w * (l + 0.5) - 0.5;
    idx_src_x = (idx_src_x > 0) ? idx_src_x : 0;
    float dw = align_flag ? idx_src_x - xw : ratio_w * l - xw;
    float de = 1.f - dw;
    {
      vx_w[l] = xw;
      vx_e[l] = xe;
      vd_w[l] = dw;
      vd_e[l] = de;
    }
  }

  std::vector<int64_t> x_strides(x_dims.size(), 1);
  for (int idx = x_strides.size() - 2; idx >= 0; idx--) {
    x_strides[idx] = x_strides[idx + 1] * x_dims[idx + 1];
  }
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < channel_size; j++) {
      for (int k = 0; k < out_h; k++) {
        for (int l = 0; l < out_w; l++) {
          DType x0 = x_data[i * x_strides[0] + j * x_strides[1] +
                            vy_n[k] * x_strides[2] + vx_w[l] * x_strides[3]];
          DType x1 = x_data[i * x_strides[0] + j * x_strides[1] +
                            vy_s[k] * x_strides[2] + vx_w[l] * x_strides[3]];
          DType x2 = x_data[i * x_strides[0] + j * x_strides[1] +
                            vy_n[k] * x_strides[2] + vx_e[l] * x_strides[3]];
          DType x3 = x_data[i * x_strides[0] + j * x_strides[1] +
                            vy_s[k] * x_strides[2] + vx_e[l] * x_strides[3]];
          *out_data = x0 * vd_s[k] * vd_e[l] + x1 * vd_n[k] * vd_e[l] +
                      x2 * vd_s[k] * vd_w[l] + x3 * vd_n[k] * vd_w[l];
          out_data++;
        }
      }
    }
  }
}

template <typename DType>
void nearest_interp_ref(const std::shared_ptr<operators::InterpolateOp> op) {
  auto scope = op->scope();
  auto op_info = op->op_info();
  auto x = scope->FindVar(op_info->Input("X").front())->GetMutable<Tensor>();
  auto out =
      scope->FindVar(op_info->Output("Out").front())->GetMutable<Tensor>();
  auto x_dims = x->dims();
  CHECK_EQ(x_dims.size(), 4);
  auto scale = op_info->GetAttr<float>("scale");
  auto out_w = op_info->GetAttr<int>("out_w");
  auto out_h = op_info->GetAttr<int>("out_h");
  auto align_corners = op_info->GetAttr<bool>("align_corners");
  // int align_mode = op_info->GetAttr<int>("align_mode");
  auto interp_method = op_info->GetAttr<std::string>("interp_method");
  CHECK_EQ(interp_method, "nearest");

  int x_h = x_dims[2];
  int x_w = x_dims[3];
  if (scale > 0) {
    out_h = static_cast<int>(x_h * scale);
    out_w = static_cast<int>(x_w * scale);
  }
  if (op_info->HasInput("OutSize")) {
    auto out_size_var_names = op_info->Input("OutSize");
    if (out_size_var_names.size() > 0) {
      auto out_size_var_name = out_size_var_names.front();
      auto out_size =
          scope->FindVar(out_size_var_name)->GetMutable<lite::Tensor>();
      CHECK_EQ(out_size->numel(), 2);
      auto out_size_data = out_size->mutable_data<int>();
      out_h = out_size_data[0];
      out_w = out_size_data[1];
    }
  }
  CHECK_GT(out_h, 0);
  CHECK_GT(out_w, 0);
  out->Resize({x_dims[0], x_dims[1], out_h, out_w});

  float ratio_h = 0.f;
  float ratio_w = 0.f;
  if (out_h > 1) {
    ratio_h = align_corners ? static_cast<float>(x_h - 1.0) / (out_h - 1.0)
                            : static_cast<float>(x_h) / out_h;
  }
  if (out_w > 1) {
    ratio_w = align_corners ? static_cast<float>(x_w - 1.0) / (out_w - 1.0)
                            : static_cast<float>(x_w) / out_w;
  }

  auto x_data = x->data<DType>();
  auto out_data = out->mutable_data<DType>();
  auto out_dims = out->dims();
  std::vector<int64_t> x_strides(x_dims.size(), 1);
  for (int idx = x_strides.size() - 2; idx >= 0; idx--) {
    x_strides[idx] = x_strides[idx + 1] * x_dims[idx + 1];
  }

  for (int n = 0; n < out_dims[0]; n++) {
    for (int c = 0; c < out_dims[1]; c++) {
      for (int h = 0; h < out_dims[2]; h++) {
        for (int w = 0; w < out_dims[3]; w++) {
          int in_i = ratio_h * h;
          int in_j = ratio_w * w;
          if (align_corners) {
            in_i = ratio_h * h + 0.5;
            in_j = ratio_w * w + 0.5;
          }
          *out_data = x_data[n * x_strides[0] + c * x_strides[1] +
                             in_i * x_strides[2] + in_j * x_strides[3]];
          out_data++;
        }
      }
    }
  }
}

void test_interpolate(int bs,
                      int ic,
                      int ih,
                      int iw,
                      int oh,
                      int ow,
                      float scale,
                      int out_size_h,
                      int out_size_w,
                      bool align_corners,
                      int align_mode,
                      std::string interp_method) {
  // prepare input&output variables
  Scope scope;
  std::string x_var_name("x");
  std::string out_size_var_name("out_size");
  std::string out_var_name("out");
  std::string out_ref_var_name("out_ref");
  auto x = scope.Var(x_var_name)->GetMutable<Tensor>();
  auto out_size = scope.Var(out_size_var_name)->GetMutable<Tensor>();
  auto out = scope.Var(out_var_name)->GetMutable<Tensor>();
  auto out_ref = scope.Var(out_ref_var_name)->GetMutable<Tensor>();
  x->Resize({bs, ic, ih, iw});
  out_size->Resize({2});

  // initialize input&output data
  FillTensor<float, int>(x);

  // initialize op desc
  cpp::OpDesc opdesc;
  opdesc.SetType(interp_method + "_interp");
  opdesc.SetInput("X", {x_var_name});
  opdesc.SetOutput("Out", {out_var_name});
  opdesc.SetAttr("out_h", oh);
  opdesc.SetAttr("out_w", ow);
  opdesc.SetAttr("scale", scale);
  opdesc.SetAttr("align_corners", static_cast<bool>(align_corners));
  opdesc.SetAttr("align_mode", static_cast<int>(align_mode));
  opdesc.SetAttr("interp_method", interp_method);
  if (out_size_h > 0 && out_size_w > 0) {
    auto out_size_dims = out_size->dims();
    CHECK_EQ(out_size_dims.size(), 1);
    CHECK_EQ(out_size_dims.production(), 2);
    auto out_size_data = out_size->mutable_data<int>();
    out_size_data[0] = out_size_h;
    out_size_data[1] = out_size_w;
    opdesc.SetInput("OutSize", {out_size_var_name});
  }

  // create op and execute reference implementation
  auto op = CreateOp<operators::InterpolateOp>(opdesc, &scope);
  if (interp_method == "bilinear") {
    bilinear_interp_ref<float>(op);
  } else {
    nearest_interp_ref<float>(op);
  }
  out_ref->CopyDataFrom(*out);

  // convert op to NPU model, then run it on NPU
  LauchOp(op, {x_var_name}, {out_var_name});

  // compare results
  auto out_dims = out->dims();
  auto out_ref_dims = out_ref->dims();
  CHECK_EQ(out_dims.size(), out_ref_dims.size());
  for (int i = 0; i < out_dims.size(); i++) {
    CHECK_EQ(out_dims[i], out_ref_dims[i]);
  }
  auto* out_data = out->mutable_data<float>();
  auto* out_ref_data = out_ref->mutable_data<float>();
  for (int i = 0; i < out->dims().production(); i++) {
    VLOG(5) << i;
    EXPECT_NEAR(out_data[i], out_ref_data[i], 1e-2f);
  }
}

TEST(NPUBridges, bilinear_interp) {
#if 1
  for (auto bs : {1, 3}) {
    for (auto ic : {3, 4}) {
      for (auto ih : {4, 5}) {
        for (auto iw : {3, 6}) {
          for (auto oh : {0, 3, 8}) {
            for (auto ow : {0, 4, 9}) {
              for (auto scale : {0.f, 0.5f, 0.6f, 2.0f, 2.2f}) {
                for (auto out_size_h : {0, 3, 11}) {
                  for (auto out_size_w : {0, 2, 12}) {
                    for (auto align_corners : {true, false}) {
                      for (auto align_mode : {0, 1}) {
                        for (auto interp_method : {"bilinear", "nearest"}) {
                          int act_oh = 0, act_ow = 0;
                          if (out_size_h > 0 && out_size_w > 0) {
                            act_oh = out_size_h;
                            act_ow = out_size_w;
                          } else if (scale > 1e-5) {
                            act_oh = static_cast<int>(ih * scale);
                            act_ow = static_cast<int>(iw * scale);
                          } else if (oh > 0 && ow > 0) {
                            act_oh = oh;
                            act_ow = ow;
                          }
                          if (act_oh <= 0 || act_ow <= 0) {
                            continue;
                          }
                          // TODO(hong19860320) multiple=(ih*iw)/(oh*ow)
                          // should
                          // not exceed 7.0 in NPU DDK, delete the following
                          // lines
                          // if the limination is removed.
                          const float largest_multiple = 7.0f;
                          float multiple =
                              static_cast<float>(ih * iw) / (act_oh * act_ow);
                          if (multiple > largest_multiple) {
                            continue;
                          }
                          if (align_mode == 0 && !align_corners) {
                            continue;
                          }
                          VLOG(3) << "bs: " << bs << " ic: " << ic
                                  << " ih: " << ih << " iw: " << iw
                                  << " oh: " << oh << " ow: " << ow
                                  << " scale: " << scale
                                  << " out_size: " << out_size_h << ","
                                  << out_size_w
                                  << " align_corners: " << align_corners
                                  << " align_mode: " << align_mode;
                          test_interpolate(bs,
                                           ic,
                                           ih,
                                           iw,
                                           oh,
                                           ow,
                                           scale,
                                           out_size_h,
                                           out_size_w,
                                           align_corners,
                                           align_mode,
                                           interp_method);
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
#else
  test_interpolate(1, 1, 4, 3, 0, 0, 1.f, 3, 6, false, 1, "nearest");
#endif
}

}  // namespace bridge
}  // namespace npu
}  // namespace lite
}  // namespace paddle

USE_LITE_OP(bilinear_interp);
USE_NPU_BRIDGE(bilinear_interp);

USE_LITE_OP(nearest_interp);
USE_NPU_BRIDGE(nearest_interp);
