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

#include <gtest/gtest.h>
#include <string>
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/core/tensor.h"
#include "lite/core/test/arena/framework.h"
#include "lite/tests/utils/fill_data.h"

namespace paddle {
namespace lite {

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
  float scale_w = (with_align) ? (static_cast<float>(win - 1) / (wout - 1))
                               : (static_cast<float>(win) / (wout));
  float scale_h = (with_align) ? (static_cast<float>(hin - 1) / (hout - 1))
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
  CHECK_EQ(x_dims.size(), 4);

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
class NearestInterpComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string x_ = "X";
  std::string sizetensor0_ = "SizeTensor0";
  std::string sizetensor1_ = "SizeTensor1";
  std::string input_scale_ = "Scale";
  std::string outsize_ = "OutSize";
  std::string out_ = "Out";
  DDim dims_{{1, 2, 3, 4}};

  std::string interp_method_ = "nearest";
  float scale_ = -1.f;
  int out_h_ = -1;
  int out_w_ = -1;
  bool align_corners_ = true;
  int align_mode_ = 1;
  bool use_sizetensor_ = false;
  bool use_input_scale_ = false;
  bool use_outsize_ = false;
  std::string dtype_ = "fp32";

 public:
  NearestInterpComputeTester(const Place& place,
                             const std::string& alias,
                             DDim dims,
                             std::string interp_method = "nearest",
                             float scale = -1.f,
                             int out_h = -1,
                             int out_w = -1,
                             bool align_corners = true,
                             int align_mode = 1,
                             bool use_sizetensor = false,
                             bool use_input_scale = false,
                             bool use_outsize = false,
                             std::string dtype = "fp32")
      : TestCase(place, alias),
        dims_(dims),
        interp_method_(interp_method),
        scale_(scale),
        out_h_(out_h),
        out_w_(out_w),
        align_corners_(align_corners),
        align_mode_(align_mode),
        use_sizetensor_(use_sizetensor),
        use_input_scale_(use_input_scale),
        use_outsize_(use_outsize),
        dtype_(dtype) {}

  void RunBaseline(Scope* scope) override {
    int out_h = out_h_;
    int out_w = out_w_;
    if (scale_ > 0) {
      out_h = dims_[2] * scale_;
      out_w = dims_[3] * scale_;
    }

    auto input = scope->FindTensor(x_);
    auto output = scope->NewTensor(out_);
    std::vector<int64_t> out_shape{dims_[0], dims_[1], out_h, out_w};
    output->Resize(out_shape);
    if (dtype_ == "fp32") {
      if (interp_method_ == "nearest") {
        ResizeNearestAlign<float>(input, output, align_corners_);
      } else if (interp_method_ == "bilinear") {
        BilinearInterpRef<float>(input, output, align_corners_, align_mode_);
      }
    } else if (dtype_ == "fp16") {
#ifdef ENABLE_ARM_FP16
      if (interp_method_ == "nearest") {
        ResizeNearestAlign<lite_api::float16_t>(input, output, align_corners_);
      } else if (interp_method_ == "bilinear") {
        BilinearInterpRef<lite_api::float16_t>(
            input, output, align_corners_, align_mode_);
      }
#endif
    } else {
      LOG(FATAL) << "this dtype: " << dtype_ << " doesn't support";
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    if (interp_method_ == "nearest") {
      op_desc->SetType("nearest_interp");
    } else if (interp_method_ == "bilinear") {
      op_desc->SetType("bilinear_interp");
    } else {
      LOG(FATAL) << "unsupport";
    }
    op_desc->SetInput("X", {x_});
    if (use_sizetensor_) {
      op_desc->SetInput("SizeTensor", {sizetensor0_, sizetensor1_});
    }
    if (use_input_scale_) {
      op_desc->SetInput("Scale", {input_scale_});
    }
    if (use_outsize_) {
      op_desc->SetInput("OutSize", {outsize_});
    }
    op_desc->SetOutput("Out", {out_});
    op_desc->SetAttr("scale", scale_);
    op_desc->SetAttr("out_h", out_h_);
    op_desc->SetAttr("out_w", out_w_);
    op_desc->SetAttr("align_corners", align_corners_);
    op_desc->SetAttr("align_mode", align_mode_);
    op_desc->SetAttr("interp_method", interp_method_);
  }

  void PrepareData() override {
    std::vector<float> din(dims_.production());
    fill_data_rand(din.data(), -1.f, 1.f, dims_.production());
    if (dtype_ == "fp16") {
#ifdef ENABLE_ARM_FP16
      std::vector<lite_api::float16_t> din_fp16(dims_.production());
      for (int i = 0; i < dims_.production(); i++) {
        din_fp16[i] = din[i];
      }
      SetCommonTensor(x_, dims_, din_fp16.data());
#endif
    } else if (dtype_ == "fp32") {
      SetCommonTensor(x_, dims_, din.data());
    } else {
      LOG(FATAL) << "this dtype: " << dtype_ << " doesn't support";
    }

    if (use_sizetensor_) {
      DDim sizetensor_dims(std::vector<int64_t>{1});
      std::vector<int> dsizetensor0{out_h_};
      std::vector<int> dsizetensor1{out_w_};
      SetCommonTensor(
          sizetensor0_, sizetensor_dims, dsizetensor0.data(), {}, true);
      SetCommonTensor(
          sizetensor1_, sizetensor_dims, dsizetensor1.data(), {}, true);
    }

    if (use_input_scale_) {
      DDim input_scale_dims(std::vector<int64_t>{1});
      std::vector<float> dinput_scale{scale_};
      SetCommonTensor(
          input_scale_, input_scale_dims, dinput_scale.data(), {}, true);
    }

    if (use_outsize_) {
      DDim outsize_dims(std::vector<int64_t>{2});
      std::vector<int> doutsize{out_h_, out_w_};
      SetCommonTensor(outsize_, outsize_dims, doutsize.data(), {}, true);
    }
  }
};

void TestInterpOuthw(Place place, float abs_error = 2e-5) {
  for (auto x_dims : std::vector<std::vector<int64_t>>{{3, 4, 8, 9}}) {
    for (auto interp_method : std::vector<std::string>{"nearest", "bilinear"}) {
      for (int out_h : {6, 8, 12}) {
        for (int out_w : {6, 9, 12}) {
#if defined(NNADAPTER_WITH_NVIDIA_TENSORRT)
          std::unique_ptr<arena::TestCase> tester(
              new NearestInterpComputeTester(place,
                                             "def",
                                             DDim(x_dims),
                                             interp_method,
                                             -1.f,
                                             out_h,
                                             out_w,
                                             false,
                                             0));
#else
          std::unique_ptr<arena::TestCase> tester(
              new NearestInterpComputeTester(place,
                                             "def",
                                             DDim(x_dims),
                                             interp_method,
                                             -1.f,
                                             out_h,
                                             out_w));
#endif
          arena::Arena arena(std::move(tester), place, abs_error);
          arena.TestPrecision();
        }
      }
    }
  }
}

void TestInterpScale(Place place, float abs_error = 2e-5) {
  for (auto x_dims : std::vector<std::vector<int64_t>>{{3, 4, 8, 9}}) {
    for (auto interp_method : std::vector<std::string>{"nearest", "bilinear"}) {
      for (float scale : {0.3f, 1.f, 1.7f}) {
#if defined(NNADAPTER_WITH_NVIDIA_TENSORRT)
        std::unique_ptr<arena::TestCase> tester(
            new NearestInterpComputeTester(place,
                                           "def",
                                           DDim(x_dims),
                                           interp_method,
                                           scale,
                                           -1,
                                           -1,
                                           false,
                                           0));
#else
        std::unique_ptr<arena::TestCase> tester(new NearestInterpComputeTester(
            place, "def", DDim(x_dims), interp_method, scale));
#endif
        arena::Arena arena(std::move(tester), place, abs_error);
        arena.TestPrecision();
      }
    }
  }
}

void TestInterpSizetensor(Place place, float abs_error = 2e-5) {
  for (auto x_dims : std::vector<std::vector<int64_t>>{{3, 4, 8, 9}}) {
    for (auto interp_method : std::vector<std::string>{"nearest", "bilinear"}) {
      std::unique_ptr<arena::TestCase> tester(
          new NearestInterpComputeTester(place,
                                         "def",
                                         DDim(x_dims),
                                         interp_method,
                                         -1.f,
                                         10,
                                         12,
                                         true,
                                         1,
                                         true,
                                         false,
                                         false));
      arena::Arena arena(std::move(tester), place, abs_error);
      arena.TestPrecision();
    }
  }
}

void TestInterpInputScale(Place place, float abs_error = 2e-5) {
  for (auto x_dims : std::vector<std::vector<int64_t>>{{3, 4, 8, 9}}) {
    for (auto interp_method : std::vector<std::string>{"nearest", "bilinear"}) {
      std::unique_ptr<arena::TestCase> tester(
          new NearestInterpComputeTester(place,
                                         "def",
                                         DDim(x_dims),
                                         interp_method,
                                         0.7,
                                         -1,
                                         -1,
                                         true,
                                         1,
                                         false,
                                         true,
                                         false));
      arena::Arena arena(std::move(tester), place, abs_error);
      arena.TestPrecision();
    }
  }
}

void TestInterpOutsize(Place place, float abs_error = 2e-5) {
  for (auto x_dims : std::vector<std::vector<int64_t>>{{3, 4, 8, 9}}) {
    for (auto interp_method : std::vector<std::string>{"nearest", "bilinear"}) {
      std::unique_ptr<arena::TestCase> tester(
          new NearestInterpComputeTester(place,
                                         "def",
                                         DDim(x_dims),
                                         interp_method,
                                         -1,
                                         4,
                                         4,
                                         true,
                                         1,
                                         false,
                                         false,
                                         true));
      arena::Arena arena(std::move(tester), place, abs_error);
      arena.TestPrecision();
    }
  }
}

void TestInterpAlignCorners(Place place, float abs_error = 2e-5) {
  for (auto x_dims : std::vector<std::vector<int64_t>>{{3, 4, 8, 9}}) {
    for (bool align_corners : {true, false}) {
      std::unique_ptr<arena::TestCase> tester(new NearestInterpComputeTester(
          place, "def", DDim(x_dims), "nearest", -1, 4, 4, align_corners));
      arena::Arena arena(std::move(tester), place, abs_error);
      arena.TestPrecision();
    }
  }
}

void TestInterpAlignMode(Place place, float abs_error = 2e-5) {
  for (auto x_dims : std::vector<std::vector<int64_t>>{{3, 4, 8, 9}}) {
    for (bool align_corners : {true, false}) {
      for (int align_mode : {0, 1}) {
        // may exist bug in arm kernel
        if (place == TARGET(kARM) && align_mode == 1 && !align_corners) {
          continue;
        }
#if defined(LITE_WITH_NNADAPTER) && defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
        if (align_mode == 0 && align_corners) continue;
#endif
        std::unique_ptr<arena::TestCase> tester(
            new NearestInterpComputeTester(place,
                                           "def",
                                           DDim(x_dims),
                                           "bilinear",
                                           -1.,
                                           5,
                                           6,
                                           align_corners,
                                           align_mode));
        arena::Arena arena(std::move(tester), place, abs_error);
        arena.TestPrecision();
      }
    }
  }
}

#ifdef ENABLE_ARM_FP16
void TestInterpOuthw_fp16(Place place, float abs_error = 2e-5) {
  for (auto x_dims : std::vector<std::vector<int64_t>>{{3, 4, 8, 9}}) {
    for (auto interp_method : std::vector<std::string>{"nearest", "bilinear"}) {
      for (int out_h : {6, 8, 12, 36, 72}) {
        for (int out_w : {6, 9, 12, 36, 48, 72}) {
          std::unique_ptr<arena::TestCase> tester(
              new NearestInterpComputeTester(place,
                                             "def",
                                             DDim(x_dims),
                                             interp_method,
                                             -1.f,
                                             out_h,
                                             out_w,
                                             true,
                                             1,
                                             false,
                                             false,
                                             false,
                                             "fp16"));
          arena::Arena arena(std::move(tester), place, abs_error);
          arena.TestPrecision();
        }
      }
    }
  }
}
#endif

TEST(Interp, precision) {
  Place place;
  float abs_error = 2e-5;
#if defined(LITE_WITH_NNADAPTER)
  place = TARGET(kNNAdapter);
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  abs_error = 5e-2;
  TestInterpOuthw(place, abs_error);
  TestInterpScale(place, abs_error);
  TestInterpInputScale(place, abs_error);
  TestInterpOutsize(place, abs_error);
  TestInterpAlignCorners(place, abs_error);
  TestInterpAlignMode(place, abs_error);
  return;
#elif defined(NNADAPTER_WITH_VERISILICON_TIMVX)
  abs_error = 5e-2;
  TestInterpOuthw(place, abs_error);
  return;
#elif defined(NNADAPTER_WITH_CAMBRICON_MLU)
  abs_error = 1e-5;
  TestInterpOuthw(place, abs_error);
  TestInterpScale(place, abs_error);
  TestInterpInputScale(place, abs_error);
  TestInterpOutsize(place, abs_error);
  TestInterpAlignCorners(place, abs_error);
  // TestInterpAlignMode(place, abs_error);
  return;
#elif defined(NNADAPTER_WITH_HUAWEI_KIRIN_NPU)
  abs_error = 5e-2;
  TestInterpOuthw(place, abs_error);
  TestInterpScale(place, abs_error);
  TestInterpInputScale(place, abs_error);
  TestInterpOutsize(place, abs_error);
  TestInterpAlignCorners(place, abs_error);
  TestInterpAlignMode(place, abs_error);
  // TestInterpSizetensor(place, abs_error);
  return;
#elif defined(NNADAPTER_WITH_INTEL_OPENVINO)
  abs_error = 1e-5;
  TestInterpOuthw(place, abs_error);
  TestInterpScale(place, abs_error);
  TestInterpInputScale(place, abs_error);
  TestInterpOutsize(place, abs_error);
  TestInterpAlignCorners(place, abs_error);
  TestInterpAlignMode(place, abs_error);
  return;
#elif defined(NNADAPTER_WITH_NVIDIA_TENSORRT)
  abs_error = 2e-5;
  TestInterpOuthw(place, abs_error);
  TestInterpScale(place, abs_error);
  return;
#elif defined(NNADAPTER_WITH_QUALCOMM_QNN)
  abs_error = 1e-2;
  TestInterpOuthw(place, abs_error);
  TestInterpScale(place, abs_error);
  TestInterpAlignCorners(place, abs_error);
  return;
#else
  return;
#endif
#elif defined(LITE_WITH_NPU)
  place = TARGET(kNPU);
  abs_error = 1e-2;  // use fp16 in npu
#elif defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#ifdef ENABLE_ARM_FP16
  Place place_fp16{TARGET(kARM), PRECISION(kFP16)};
  abs_error = 1e-2;
  TestInterpOuthw_fp16(place_fp16, abs_error);
#endif
#elif defined(LITE_WITH_X86)
  place = TARGET(kX86);
#else
  return;
#endif

  TestInterpOuthw(place, abs_error);
  TestInterpScale(place, abs_error);
  TestInterpSizetensor(place, abs_error);
  TestInterpInputScale(place, abs_error);
  TestInterpOutsize(place, abs_error);
  TestInterpAlignCorners(place, abs_error);
  TestInterpAlignMode(place, abs_error);
}

}  // namespace lite
}  // namespace paddle
