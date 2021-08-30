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
#include <string>
#include "lite/core/device_info.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"
#include "lite/core/subgraph/subgraph_bridge_registry.h"
#include "lite/kernels/mlu/bridges/test_helper.h"
#include "lite/kernels/mlu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace mlu {

template <typename dtype>
void ResizeNearestAlign(const lite::Tensor* x,
                        lite::Tensor* out,
                        bool with_align) {
  auto x_dims = x->dims();
  int num = x_dims[0];
  int channels = x_dims[1];
  int hin = x_dims[2];
  int win = x_dims[3];
  int hout = out->dims()[2];
  int wout = out->dims()[3];
  dtype scale_w = (with_align) ? (static_cast<float>(win - 1) / (wout - 1))
                               : (static_cast<float>(win) / (wout));
  dtype scale_h = (with_align) ? (static_cast<float>(hin - 1) / (hout - 1))
                               : (static_cast<float>(hin) / (hout));
  const dtype* src = x->data<dtype>();
  dtype* dst = out->mutable_data<dtype>();
  int dst_stride_w = 1;
  int dst_stride_h = wout;
  int dst_stride_c = wout * hout;
  int dst_stride_batch = wout * hout * channels;
  int src_stride_w = 1;
  int src_stride_h = win;
  int src_stride_c = win * hin;
  int src_stride_batch = win * hin * channels;
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
      int src_index = n * src_stride_batch + c * src_stride_c;
      for (int h = 0; h < hout; ++h) {
        for (int w = 0; w < wout; ++w) {
          int fw = (with_align) ? static_cast<int>(scale_w * w + 0.5)
                                : static_cast<int>(scale_w * w);
          fw = (fw < 0) ? 0 : fw;
          int fh = (with_align) ? static_cast<int>(scale_h * h + 0.5)
                                : static_cast<int>(scale_h * h);
          fh = (fh < 0) ? 0 : fh;
          int w_start = static_cast<int>(fw);
          int h_start = static_cast<int>(fh);
          int dst_index = n * dst_stride_batch + c * dst_stride_c +
                          h * dst_stride_h + w * dst_stride_w;
          dst[dst_index] =
              src[src_index + w_start * src_stride_w + h_start * src_stride_h];
        }
      }
    }
  }
}

template <typename DType>
void BilinearInterpRef(const lite::Tensor* x,
                       lite::Tensor* out,
                       bool align_corners,
                       int align_mode) {
  auto x_dims = x->dims();
  int batch_size = x_dims[0];
  int channel_size = x_dims[1];
  auto x_h = x_dims[2];
  auto x_w = x_dims[3];
  CHECK_EQ(x_dims.size(), 4u);

  auto out_dims = out->dims();
  int out_h = out_dims[2];
  int out_w = out_dims[3];

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
  auto x_data = x->data<DType>();
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
    int xw = align_flag ? static_cast<int>(ratio_w * (l + 0.5) - 0.5)
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

class InterpComputeTester {
 protected:
  // common attributes for this op.
  std::string x_var_name = "X";
  std::string outsize_var_name = "OutSize";
  std::string out_var_name = "Out";
  std::string out_ref_var_name = "out_ref";
  DDim dims_{{1, 2, 3, 4}};

  Scope scope;
  std::string interp_method_ = "nearest";
  float scale_ = -1.f;
  int out_h_ = -1;
  int out_w_ = -1;
  bool align_corners_ = true;
  int align_mode_ = 1;
  bool use_outsize_ = false;

 public:
  InterpComputeTester(const std::string& alias,
                      DDim dims,
                      std::string interp_method = "nearest",
                      float scale = -1.f,
                      int out_h = -1,
                      int out_w = -1,
                      bool align_corners = true,
                      int align_mode = 1,
                      bool use_outsize = false)
      : dims_(dims),
        interp_method_(interp_method),
        scale_(scale),
        out_h_(out_h),
        out_w_(out_w),
        align_corners_(align_corners),
        align_mode_(align_mode),
        use_outsize_(use_outsize) {}

  void Execute(float abs_error) {
    cpp::OpDesc op_desc;
    auto* x = scope.Var(x_var_name)->GetMutable<Tensor>();
    auto* out = scope.Var(out_var_name)->GetMutable<Tensor>();
    auto* outsize = scope.Var(outsize_var_name)->GetMutable<Tensor>();
    auto* outref = scope.Var(out_ref_var_name)->GetMutable<Tensor>();
    int out_h = out_h_;
    int out_w = out_w_;
    if (scale_ > 0) {
      out_h = static_cast<int>(dims_[2] * scale_);
      out_w = static_cast<int>(dims_[3] * scale_);
    }
    x->Resize(dims_);
    /* printf("----output tensor dims: %ld, %d, %d, %ld\n", dims_[0], out_h,
     * out_w, dims_[1]); */
    std::vector<int64_t> out_shape_nchw = {dims_[0], dims_[1], out_h, out_w};
    outref->Resize(out_shape_nchw);
    outsize->Resize({2});

    FillTensor<float, float>(x, -1.f, 1.f);

    if (use_outsize_) {
      outsize->mutable_data<int>()[0] = out_h;
      outsize->mutable_data<int>()[1] = out_w;
      outsize->set_persistable(true);
    }

    if (interp_method_ == "nearest") {
      op_desc.SetType("nearest_interp");
    } else if (interp_method_ == "bilinear") {
      op_desc.SetType("bilinear_interp");
    } else {
      LOG(FATAL) << "unsupport";
    }
    op_desc.SetInput("X", {x_var_name});
    if (use_outsize_) {
      op_desc.SetInput("OutSize", {outsize_var_name});
    }
    op_desc.SetOutput("Out", {out_var_name});
    op_desc.SetAttr("scale", scale_);
    op_desc.SetAttr("out_h", out_h_);
    op_desc.SetAttr("out_w", out_w_);
    op_desc.SetAttr("align_corners", align_corners_);
    op_desc.SetAttr("align_mode", align_mode_);
    op_desc.SetAttr("interp_method", interp_method_);
    auto op = CreateOp<operators::InterpolateOp>(op_desc, &scope);

    if (interp_method_ == "nearest") {
      ResizeNearestAlign<float>(x, outref, align_corners_);
    } else if (interp_method_ == "bilinear") {
      BilinearInterpRef<float>(x, outref, align_corners_, align_mode_);
    }

    int in = dims_[0], ic = dims_[1], ih = dims_[2], iw = dims_[3];
    Tensor input_trans;
    input_trans.Resize(dims_);
    transpose(x->mutable_data<float>(),
              input_trans.mutable_data<float>(),
              {in, ic, ih, iw},
              {0, 2, 3, 1});
    x->CopyDataFrom(input_trans);
    if (use_outsize_) {
      LaunchOp(op, {x_var_name, outsize_var_name}, {out_var_name});
    } else {
      LaunchOp(op, {x_var_name}, {out_var_name});
    }

    auto* out_ref_data = outref->mutable_data<float>();

    Tensor output_trans;
    output_trans.Resize(out_shape_nchw);
    transpose(
        out->mutable_data<float>(),
        output_trans.mutable_data<float>(),
        {static_cast<int>(dims_[0]), out_h, out_w, static_cast<int>(dims_[1])},
        {0, 3, 1, 2});
    auto* out_data = output_trans.mutable_data<float>();
    for (int i = 0; i < out->dims().production(); ++i) {
      EXPECT_NEAR(out_data[i], out_ref_data[i], abs_error);
    }
  }
};

void TestInterpOuthw(float abs_error = 2e-5) {
  for (auto x_dims : std::vector<std::vector<int64_t>>{{3, 4, 8, 9}}) {
    /* for (auto interp_method : std::vector<std::string>{"nearest",
     * "bilinear"}) { */
    for (auto interp_method : std::vector<std::string>{"nearest"}) {
      for (int out_h : {6, 8, 12}) {
        for (int out_w : {6, 9}) {
          printf("testcase %s: out_w %d, out_h %d\n",
                 interp_method.c_str(),
                 out_w,
                 out_h);
          InterpComputeTester tester(
              "def", DDim(x_dims), interp_method, -1.f, out_h, out_w);
          tester.Execute(abs_error);
        }
      }
    }
  }
}

void TestInterpScale(float abs_error = 2e-5) {
  for (auto x_dims : std::vector<std::vector<int64_t>>{{3, 4, 8, 9}}) {
    /* for (auto interp_method : std::vector<std::string>{"nearest",
     * "bilinear"}) { */
    for (auto interp_method : std::vector<std::string>{"nearest"}) {
      for (float scale : {0.3f, 1.f, 1.7f}) {
        printf("testcase %s: scale: %f\n", interp_method.c_str(), scale);
        InterpComputeTester tester("def", DDim(x_dims), interp_method, scale);
        tester.Execute(abs_error);
      }
    }
  }
}

void TestInterpOutsize(float abs_error = 2e-5) {
  for (auto x_dims : std::vector<std::vector<int64_t>>{{3, 4, 8, 9}}) {
    /* for (auto interp_method : std::vector<std::string>{"nearest",
     * "bilinear"}) { */
    for (auto interp_method : std::vector<std::string>{"nearest"}) {
      printf("testcase %s: outsize: %d %d\n", interp_method.c_str(), 4, 4);
      InterpComputeTester tester(
          "def", DDim(x_dims), interp_method, -1, 4, 4, true, 1, true);
      tester.Execute(abs_error);
    }
  }
}

void TestInterpAlignCorners(float abs_error = 2e-5) {
  for (auto x_dims : std::vector<std::vector<int64_t>>{{3, 4, 8, 9}}) {
    for (bool align_corners : {true, false}) {
      printf(
          "testcase nearest: scale: 0.4, out_w -1 out_h -1, align_corners %d\n",
          align_corners);
      InterpComputeTester tester(
          "def", DDim(x_dims), "nearest", 0.4, -1, -1, align_corners);
      tester.Execute(abs_error);
    }
  }
}

void TestInterpAlignMode(float abs_error = 2e-5) {
  for (auto x_dims : std::vector<std::vector<int64_t>>{{3, 4, 8, 9}}) {
    for (bool align_corners : {true, false}) {
      for (int align_mode : {0, 1}) {
        printf(
            "testcase bilinear: scale: 0.7, out_w -1 out_h -1, align_corners "
            "%d, mode %d\n",
            align_corners,
            align_mode);
        InterpComputeTester tester("def",
                                   DDim(x_dims),
                                   "bilinear",
                                   0.7,
                                   -1,
                                   -1,
                                   align_corners,
                                   align_mode);
        tester.Execute(abs_error);
      }
    }
  }
}

TEST(MLUBridges, interpolate) {
  float abs_error = 2e-5;
  TestInterpOuthw(abs_error);
  TestInterpScale(abs_error);
  // bug, not usable
  // TestInterpOutsize(abs_error);
  TestInterpAlignCorners(abs_error);
  // only for bilinear interp
  // TestInterpAlignMode(abs_error);
}

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

USE_SUBGRAPH_BRIDGE(nearest_interp, kMLU);
