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

#include <arm_neon.h>
#include <gtest/gtest.h>
#include <string>
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/core/arena/framework.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {

void bilinear_interp(const float* src,
                     int w_in,
                     int h_in,
                     float* dst,
                     int w_out,
                     int h_out,
                     float scale_x,
                     float scale_y,
                     bool align_corners) {
  int* buf = new int[w_out + h_out + w_out * 2 + h_out * 2];

  int* xofs = buf;
  int* yofs = buf + w_out;

  float* alpha = reinterpret_cast<float*>(buf + w_out + h_out);
  float* beta = reinterpret_cast<float*>(buf + w_out + h_out + w_out * 2);

  float fx = 0.0f;
  float fy = 0.0f;
  int sx = 0;
  int sy = 0;
  if (align_corners) {
    scale_x = static_cast<float>(w_in - 1) / (w_out - 1);
    scale_y = static_cast<float>(h_in - 1) / (h_out - 1);
    // calculate x axis coordinate
    for (int dx = 0; dx < w_out; dx++) {
      fx = dx * scale_x;
      sx = static_cast<int>(fx);
      fx -= sx;
      xofs[dx] = sx;
      alpha[dx * 2] = 1.f - fx;
      alpha[dx * 2 + 1] = fx;
    }
    // calculate y axis coordinate
    for (int dy = 0; dy < h_out; dy++) {
      fy = dy * scale_y;
      sy = static_cast<int>(fy);
      fy -= sy;
      yofs[dy] = sy;
      beta[dy * 2] = 1.f - fy;
      beta[dy * 2 + 1] = fy;
    }
  } else {
    scale_x = static_cast<float>(w_in / w_out);
    scale_y = static_cast<float>(h_in / h_out);
    // calculate x axis coordinate
    for (int dx = 0; dx < w_out; dx++) {
      fx = scale_x * (dx + 0.5f) - 0.5f;
      fx = fx < 0 ? 0.f : fx;
      sx = static_cast<int>(fx);
      fx -= sx;
      xofs[dx] = sx;
      alpha[dx * 2] = 1.f - fx;
      alpha[dx * 2 + 1] = fx;
    }
    // calculate y axis coordinate
    for (int dy = 0; dy < h_out; dy++) {
      fy = scale_y * (dy + 0.5f) - 0.5f;
      fy = fy < 0 ? 0.f : fy;
      sy = static_cast<int>(fy);
      fy -= sy;
      yofs[dy] = sy;
      beta[dy * 2] = 1.f - fy;
      beta[dy * 2 + 1] = fy;
    }
  }
  float* rowsbuf0 = new float[w_out];
  float* rowsbuf1 = new float[w_out];
  float* rows0 = rowsbuf0;
  float* rows1 = rowsbuf1;
  // output w , h boundary
  int w_bound = w_out;
  int h_bound = h_out;
  if (align_corners) {
    w_bound = ceil((w_in - 1) / scale_x);
    h_bound = ceil((h_in - 1) / scale_y);
  } else {
    w_bound = ceil((w_in - 0.5f) / scale_x - 0.5f);
    h_bound = ceil((h_in - 0.5f) / scale_y - 0.5f);
  }
  // h_bound loop
  for (int dy = 0; dy < h_bound; dy++) {
    int sy = yofs[dy];

    const float* s0 = src + sy * w_in;
    const float* s1 = src + (sy + 1) * w_in;

    const float* alphap = alpha;
    float* rows0p = rows0;
    float* rows1p = rows1;

    int dx = 0;
    // w_bound loop
    for (; dx + 1 < w_bound; dx += 2) {
      int sx = xofs[dx];
      int sxn = xofs[dx + 1];
      const float* s0p = s0 + sx;
      const float* s1p = s1 + sx;
      const float* s0np = s0 + sxn;
      const float* s1np = s1 + sxn;

      float32x4_t _a = vld1q_f32(alphap);
      float32x2_t _s0 = vld1_f32(s0p);
      float32x2_t _s1 = vld1_f32(s1p);
      float32x2_t _s0n = vld1_f32(s0np);
      float32x2_t _s1n = vld1_f32(s1np);
      float32x4_t _s0s0n = vcombine_f32(_s0, _s0n);
      float32x4_t _ms0 = vmulq_f32(_s0s0n, _a);
      float32x4_t _s1s1n = vcombine_f32(_s1, _s1n);
      float32x4_t _ms1 = vmulq_f32(_s1s1n, _a);
      float32x2_t _rows0 = vpadd_f32(vget_low_f32(_ms0), vget_high_f32(_ms0));
      vst1_f32(rows0p + dx, _rows0);
      float32x2_t _rows1 = vpadd_f32(vget_low_f32(_ms1), vget_high_f32(_ms1));
      vst1_f32(rows1p + dx, _rows1);
      alphap += 4;
    }
    // w_bound remain loop
    for (; dx < w_bound; dx++) {
      int sx = xofs[dx];
      const float* s0p = s0 + sx;
      const float* s1p = s1 + sx;

      float a0 = alphap[0];
      float a1 = alphap[1];
      rows0p[dx] = s0p[0] * a0 + s0p[1] * a1;
      rows1p[dx] = s1p[0] * a0 + s1p[1] * a1;

      alphap += 2;
    }

    const float buffer1[2] = {*(src + sy * w_in + w_in - 1),
                              *(src + sy * w_in + w_in - 1)};
    const float buffer2[2] = {*(src + (sy + 1) * w_in + w_in - 1),
                              *(src + (sy + 1) * w_in + w_in - 1)};
    // w_bound - w_out loop
    for (; dx + 1 < w_out; dx += 2) {
      const float* s0p = buffer1;
      const float* s1p = buffer2;
      const float* s0np = buffer1;
      const float* s1np = buffer2;

      float32x4_t _a = vld1q_f32(alphap);
      float32x2_t _s0 = vld1_f32(s0p);
      float32x2_t _s1 = vld1_f32(s1p);
      float32x2_t _s0n = vld1_f32(s0np);
      float32x2_t _s1n = vld1_f32(s1np);

      float32x4_t _s0s0n = vcombine_f32(_s0, _s0n);
      float32x4_t _ms0 = vmulq_f32(_s0s0n, _a);
      float32x4_t _s1s1n = vcombine_f32(_s1, _s1n);
      float32x4_t _ms1 = vmulq_f32(_s1s1n, _a);

      float32x2_t _rows0 = vpadd_f32(vget_low_f32(_ms0), vget_high_f32(_ms0));
      vst1_f32(rows0p + dx, _rows0);
      float32x2_t _rows1 = vpadd_f32(vget_low_f32(_ms1), vget_high_f32(_ms1));
      vst1_f32(rows1p + dx, _rows1);

      alphap += 4;
    }
    // w_bound - w_out remain loop
    for (; dx < w_out; dx++) {
      const float* s0p = buffer1;
      const float* s1p = buffer2;

      float a0 = alphap[0];
      float a1 = alphap[1];
      rows0p[dx] = s0p[0] * a0 + s0p[1] * a1;
      rows1p[dx] = s1p[0] * a0 + s1p[1] * a1;

      alphap += 2;
    }

    float b0 = beta[0];
    float b1 = beta[1];

    float* dp = dst + dy * w_out;

    int nn = w_out >> 3;
    int remain = w_out - (nn << 3);

#ifdef __aarch64__
    float32x4_t _b0 = vdupq_n_f32(b0);
    float32x4_t _b1 = vdupq_n_f32(b1);
    // calculate and store results
    for (; nn > 0; nn--) {
      float32x4_t _rows0 = vld1q_f32(rows0p);
      float32x4_t _d = vmulq_f32(_rows0, _b0);
      float32x4_t _rows1 = vld1q_f32(rows1p);
      _d = vmlaq_f32(_d, _rows1, _b1);

      float32x4_t _rows0n = vld1q_f32(rows0p + 4);
      float32x4_t _rows1n = vld1q_f32(rows1p + 4);

      float32x4_t _dn = vmulq_f32(_rows0n, _b0);
      vst1q_f32(dp, _d);
      _dn = vmlaq_f32(_dn, _rows1n, _b1);
      vst1q_f32(dp + 4, _dn);

      dp += 8;
      rows0p += 8;
      rows1p += 8;
    }

#else
    if (nn > 0) {
      asm volatile(
          "vdup.32 q0, %[b0]                   @dup b0 to q1\n"
          "vdup.32 q1, %[b1]                   @dup b1 to q0\n"
          "1:                                                      \n"
          "vld1.32 {d4-d5}, [%[rows0p]]!       @loads rows0p to q2\n"
          "vld1.32 {d6-d7}, [%[rows1p]]!       @loads rows0p to q3\n"
          "vmul.f32 q2, q2, q0                 @mul\n"
          "vmla.f32 q2, q3, q1                 @mul add\n"
          "vst1.32 {d4-d5}, [%[out]]!          @store out to q2 \n"
          "pld [%[rows0p]]                     @preload rows0p\n"

          "vld1.32 {d4-d5}, [%[rows0p]]!       @loads rows0p to q2\n"
          "vld1.32 {d6-d7}, [%[rows1p]]!       @load rows1p to q3\n"
          "vmul.f32 q2, q2, q0                 @mul\n"
          "vmla.f32 q2, q3, q1                 @mul add\n"
          "vst1.32 {d4-d5}, [%[out]]!          @store out to q2 \n"
          "pld [%[rows1p]]                     @preload rows1p\n"
          "subs %[loopc], #1                   @loop count minus #1\n"
          "bne 1b                              @jump to 1\n"
          : [rows0p] "+r"(rows0p),
            [rows1p] "+r"(rows1p),
            [out] "+r"(dp),
            [loopc] "+r"(nn)
          : [b0] "r"(b0), [b1] "r"(b1)
          : "cc", "memory", "q0", "q1", "q2", "q3");
    }
#endif
    // calculate and store remain resluts
    for (; remain; --remain) {
      *dp++ = *rows0p++ * b0 + *rows1p++ * b1;
    }
    beta += 2;
  }

  // h_bound - h_out loop
  for (int dy = h_bound; dy < h_out; dy++) {
    int sy = h_in - 1;
    const float* s0 = src + sy * w_in;
    const float* s1 = s0;
    const float* alphap = alpha;
    float* rows0p = rows0;
    float* rows1p = rows1;

    int dx = 0;
    // w_bound loop
    for (; dx + 1 < w_bound; dx += 2) {
      int sx = xofs[dx];
      int sxn = xofs[dx + 1];
      const float* s0p = s0 + sx;
      const float* s1p = s1 + sx;
      const float* s0np = s0 + sxn;
      const float* s1np = s1 + sxn;

      float32x4_t _a = vld1q_f32(alphap);
      float32x2_t _s0 = vld1_f32(s0p);
      float32x2_t _s1 = vld1_f32(s1p);
      float32x2_t _s0n = vld1_f32(s0np);
      float32x2_t _s1n = vld1_f32(s1np);

      float32x4_t _s0s0n = vcombine_f32(_s0, _s0n);
      float32x4_t _ms0 = vmulq_f32(_s0s0n, _a);
      float32x4_t _s1s1n = vcombine_f32(_s1, _s1n);
      float32x4_t _ms1 = vmulq_f32(_s1s1n, _a);

      float32x2_t _rows0 = vpadd_f32(vget_low_f32(_ms0), vget_high_f32(_ms0));
      vst1_f32(rows0p + dx, _rows0);
      float32x2_t _rows1 = vpadd_f32(vget_low_f32(_ms1), vget_high_f32(_ms1));
      vst1_f32(rows1p + dx, _rows1);

      alphap += 4;
    }

    // w_bound remain loop
    for (; dx < w_bound; dx++) {
      int sx = xofs[dx];
      const float* s0p = s0 + sx;
      float a0 = alphap[0];
      float a1 = alphap[1];
      rows0p[dx] = s0p[0] * a0 + s0p[1] * a1;
      rows1p[dx] = rows0p[dx];

      alphap += 2;
    }

    const float buffer1[2] = {*(src + sy * w_in + w_in - 1),
                              *(src + sy * w_in + w_in - 1)};

    // w_bound - w_out loop
    for (; dx + 1 < w_out; dx += 2) {
      const float* s0p = buffer1;
      const float* s1p = buffer1;
      const float* s0np = buffer1;
      const float* s1np = buffer1;

      float32x4_t _a = vld1q_f32(alphap);
      float32x2_t _s0 = vld1_f32(s0p);
      float32x2_t _s1 = vld1_f32(s1p);
      float32x2_t _s0n = vld1_f32(s0np);
      float32x2_t _s1n = vld1_f32(s1np);

      float32x4_t _s0s0n = vcombine_f32(_s0, _s0n);
      float32x4_t _ms0 = vmulq_f32(_s0s0n, _a);
      float32x4_t _s1s1n = vcombine_f32(_s1, _s1n);
      float32x4_t _ms1 = vmulq_f32(_s1s1n, _a);

      float32x2_t _rows0 = vpadd_f32(vget_low_f32(_ms0), vget_high_f32(_ms0));
      vst1_f32(rows0p + dx, _rows0);
      float32x2_t _rows1 = vpadd_f32(vget_low_f32(_ms1), vget_high_f32(_ms1));
      vst1_f32(rows1p + dx, _rows1);

      alphap += 4;
    }

    // w_bound - wout remain loop
    for (; dx < w_out; dx++) {
      const float* s0p = buffer1;
      float a0 = alphap[0];
      float a1 = alphap[1];
      rows0p[dx] = s0p[0] * a0 + s0p[1] * a1;
      rows1p[dx] = rows0p[dx];
      alphap += 2;
    }

    float b0 = beta[0];
    float b1 = beta[1];

    float* dp = dst + dy * w_out;

    int nn = w_out >> 3;
    int remain = w_out - (nn << 3);

#ifdef __aarch64__
    float32x4_t _b0 = vdupq_n_f32(b0);
    float32x4_t _b1 = vdupq_n_f32(b1);
    // calculate and store results

    for (; nn > 0; nn--) {
      float32x4_t _rows0 = vld1q_f32(rows0p);
      float32x4_t _d = vmulq_f32(_rows0, _b0);
      float32x4_t _rows1 = vld1q_f32(rows1p);
      _d = vmlaq_f32(_d, _rows1, _b1);

      float32x4_t _rows0n = vld1q_f32(rows0p + 4);
      float32x4_t _rows1n = vld1q_f32(rows1p + 4);

      float32x4_t _dn = vmulq_f32(_rows0n, _b0);
      vst1q_f32(dp, _d);
      _dn = vmlaq_f32(_dn, _rows1n, _b1);
      vst1q_f32(dp + 4, _dn);

      dp += 8;
      rows0p += 8;
      rows1p += 8;
    }

#else
    if (nn > 0) {
      asm volatile(
          "vdup.32 q0, %[b0]                   @dup b0 to q1\n"
          "vdup.32 q1, %[b1]                   @dup b1 to q0\n"
          "1:                                                      \n"
          "vld1.32 {d4-d5}, [%[rows0p]]!       @loads rows0p to q2\n"
          "vld1.32 {d6-d7}, [%[rows1p]]!       @loads rows0p to q3\n"
          "vmul.f32 q2, q2, q0                 @mul\n"
          "vmla.f32 q2, q3, q1                 @mul add\n"
          "vst1.32 {d4-d5}, [%[out]]!          @store out to q2 \n"
          "pld [%[rows0p]]                     @preload rows0p\n"

          "vld1.32 {d4-d5}, [%[rows0p]]!       @loads rows0p to q2\n"
          "vld1.32 {d6-d7}, [%[rows1p]]!       @load rows1p to q3\n"
          "vmul.f32 q2, q2, q0                 @mul\n"
          "vmla.f32 q2, q3, q1                 @mul add\n"
          "vst1.32 {d4-d5}, [%[out]]!          @store out to q2 \n"
          "pld [%[rows1p]]                     @preload rows1p\n"
          "subs %[loopc], #1                   @loop count minus #1\n"
          "bne 1b                              @jump to 1\n"
          : [rows0p] "+r"(rows0p),
            [rows1p] "+r"(rows1p),
            [out] "+r"(dp),
            [loopc] "+r"(nn)
          : [b0] "r"(b0), [b1] "r"(b1)
          : "cc", "memory", "q0", "q1", "q2", "q3");
    }
#endif

    // calculate and store remain results
    for (; remain; --remain) {
      *dp++ = *rows0p++ * b0 + *rows1p++ * b1;
    }

    beta += 2;
  }
  delete[] buf;
  delete[] rowsbuf0;
  delete[] rowsbuf1;
}

class BilinearInterpComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string input0_ = "X";
  std::string input1_ = "OutSize";
  std::string output_ = "Out";

  float height_scale_ = 0.f;
  float width_scale_ = 0.f;
  int out_height_ = -1;
  int out_width_ = -1;
  bool align_corners_ = true;
  std::string interp_method_ = "Bilinear";
  DDim dims_{{1, 1}};
  DDim _dims0_{{1, 1, 16, 16}};
  DDim _dims1_{{2}};

 public:
  BilinearInterpComputeTester(const Place& place,
                              const std::string& alias,
                              float height_scale,
                              float width_scale,
                              int out_height,
                              int out_width,
                              bool align_corners,
                              std::string interp_method)
      : TestCase(place, alias),
        height_scale_(height_scale),
        width_scale_(width_scale),
        out_height_(out_height),
        out_width_(out_width),
        align_corners_(align_corners),
        interp_method_(interp_method) {}

  void RunBaseline(Scope* scope) override {
    width_scale_ = height_scale_;
    std::vector<const lite::Tensor*> inputs;
    inputs.emplace_back(scope->FindTensor(input0_));
    inputs.emplace_back(scope->FindTensor(input1_));
    auto outsize_data = inputs[1]->data<int>();
    if (out_width_ != -1 && out_height_ != -1) {
      height_scale_ = static_cast<float>(out_height_ / inputs[0]->dims()[2]);
      width_scale_ = static_cast<float>(out_width_ / inputs[0]->dims()[3]);
    }
    auto* outputs = scope->NewTensor(output_);
    CHECK(outputs);
    if (inputs.size() > 1) {
      int h_out = outsize_data[0];  // HW
      int w_out = outsize_data[1];  // HW
      int num_cout = inputs[0]->dims()[0];
      int c_cout = inputs[0]->dims()[1];
      outputs->Resize({num_cout, c_cout, h_out, w_out});
    } else {
      int out_h;
      int out_w;
      if (-1 == out_height_ && -1 == out_width_) {
        out_h = inputs[0]->dims()[2] * height_scale_;
        out_w = inputs[0]->dims()[3] * width_scale_;
      } else {
        out_h = out_height_;
        out_w = out_width_;
      }
      outputs->Resize(
          {inputs[0]->dims()[0], inputs[0]->dims()[1], out_h, out_w});
    }

    float* dout = outputs->mutable_data<float>();
    const float* din = inputs[0]->data<float>();
    int out_num = outputs->dims()[0];
    int out_c = outputs->dims()[1];
    int count = out_num * out_c;
    int in_h = inputs[0]->dims()[2];
    int in_w = inputs[0]->dims()[3];
    int out_h = outputs->dims()[2];
    int out_w = outputs->dims()[3];
    int spatial_in = in_h * in_w;
    int spatial_out = out_h * out_w;
    bilinear_interp(din,
                    in_w,
                    in_h,
                    dout,
                    out_w,
                    out_h,
                    1.f / width_scale_,
                    1.f / height_scale_,
                    align_corners_);
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("bilinear_interp");
    op_desc->SetInput("X", {input0_});
    op_desc->SetInput("OutSize", {input1_});
    op_desc->SetOutput("Out", {output_});
    op_desc->SetAttr("scale", height_scale_);
    op_desc->SetAttr("out_h", out_height_);
    op_desc->SetAttr("out_w", out_width_);
    op_desc->SetAttr("align_corners", align_corners_);
    op_desc->SetAttr("interp_method", interp_method_);
  }

  void PrepareData() override {
    std::vector<float> data0(_dims0_.production());
    for (int i = 0; i < _dims0_.production(); i++) {
      data0[i] = i * 1.1;
    }
    SetCommonTensor(input0_, _dims0_, data0.data());

    std::vector<int> data1(_dims1_.production());
    for (int i = 0; i < _dims1_.production(); i++) {
      data1[i] = 16;
    }
    SetCommonTensor(input1_, _dims1_, data1.data());
  }
};

void test_bilinear_interp(Place place) {
  std::string interp_method = "Bilinear";
  for (float scale : {1., 0.5, 0.3}) {
    for (int out_height : {8, 16}) {
      for (int out_width : {8, 16}) {
        for (bool align_corners : {true, false}) {
          std::unique_ptr<arena::TestCase> tester(
              new BilinearInterpComputeTester(place,
                                              "def",
                                              scale,
                                              scale,
                                              out_height,
                                              out_width,
                                              align_corners,
                                              interp_method));
          arena::Arena arena(std::move(tester), place, 2e-5);
          arena.TestPrecision();
        }
      }
    }
  }
}

TEST(BilinearInterp, precision) {
// #ifdef LITE_WITH_X86
//   Place place(TARGET(kX86));
// #endif
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
  test_bilinear_interp(place);
#endif
}

}  // namespace lite
}  // namespace paddle
