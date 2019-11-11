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
#include "lite/core/arena/framework.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {

inline std::vector<int> get_new_shape(
    std::vector<const lite::Tensor*> list_new_shape_tensor) {
  // get tensor from
  std::vector<int> vec_new_shape;
  for (size_t i = 0; i < list_new_shape_tensor.size(); ++i) {
    auto tensor = list_new_shape_tensor[i];
    vec_new_shape.push_back(static_cast<int32_t>(*(tensor->data<int32_t>())));
  }
  return vec_new_shape;
}

template <typename T>
inline std::vector<T> get_new_data_from_tensor(const Tensor* new_data_tensor) {
  std::vector<T> vec_new_data;
  auto* new_data = new_data_tensor->data<T>();
  lite::Tensor cpu_starts_tensor;
  vec_new_data =
      std::vector<T>(new_data, new_data + new_data_tensor->dims().production());
  return vec_new_data;
}

template <typename dtype>
void resize_bilinear_align(std::vector<const lite::Tensor*> inputs,
                           lite::Tensor* output) {
  int hin = inputs[0]->dims()[2];
  int win = inputs[0]->dims()[3];
  int channels = inputs[0]->dims()[1];
  int num = inputs[0]->dims()[0];
  int hout = output->dims()[2];
  int wout = output->dims()[3];

  dtype scale_w = static_cast<dtype>(win - 1) / (wout - 1);
  dtype scale_h = static_cast<dtype>(hin - 1) / (hout - 1);
  const dtype* src = inputs[0]->data<dtype>();
  dtype* dst = output->mutable_data<dtype>();
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
          dtype fw = w * scale_w;
          dtype fh = h * scale_h;
          int w_start = static_cast<int>(fw);
          int w_id = w_start < win - 1 ? 1 : 0;
          int w_end = static_cast<int>(fw + w_id);
          int h_start = static_cast<int>(fh);
          int h_id = h_start < hin - 1 ? 1 : 0;
          int h_end = static_cast<int>(fh + h_id);
          fw -= w_start;
          fh -= h_start;
          const dtype w00 = (1.0 - fh) * (1.0 - fw);
          const dtype w01 = fw * (1.0 - fh);
          const dtype w10 = fh * (1.0 - fw);
          const dtype w11 = fw * fh;
          dtype tl =
              src[src_index + w_start * src_stride_w + h_start * src_stride_h];
          dtype tr =
              src[src_index + w_end * src_stride_w + h_start * src_stride_h];
          dtype bl =
              src[src_index + w_start * src_stride_w + h_end * src_stride_h];
          dtype br =
              src[src_index + w_end * src_stride_w + h_end * src_stride_h];
          int dst_index = n * dst_stride_batch + c * dst_stride_c +
                          h * dst_stride_h + w * dst_stride_w;
          dst[dst_index] =
              static_cast<dtype>(w00 * tl + w01 * tr + w10 * bl + w11 * br);
        }
      }
    }
  }
}

template <typename dtype>
void resize_bilinear_no_align(std::vector<const lite::Tensor*> inputs,
                              lite::Tensor* output) {
  int hin = inputs[0]->dims()[2];
  int win = inputs[0]->dims()[3];
  int channels = inputs[0]->dims()[1];
  int num = inputs[0]->dims()[0];
  int hout = output->dims()[2];
  int wout = output->dims()[3];
  dtype scale_w = static_cast<dtype>(win) / (wout);
  dtype scale_h = static_cast<dtype>(hin) / (hout);
  const dtype* src = inputs[0]->data<dtype>();
  dtype* dst = output->mutable_data<dtype>();
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
          dtype fw = scale_w * (w + 0.5f) - 0.5f;
          fw = (fw < 0) ? 0 : fw;
          dtype fh = scale_h * (h + 0.5f) - 0.5f;
          fh = (fh < 0) ? 0 : fh;
          int w_start = static_cast<int>(fw);
          int w_id = w_start < win - 1 ? 1 : 0;
          int w_end = static_cast<int>(fw + w_id);
          int h_start = static_cast<int>(fh);
          int h_id = h_start < hin - 1 ? 1 : 0;
          int h_end = static_cast<int>(fh + h_id);
          fw -= w_start;
          fh -= h_start;
          const dtype w00 = (1.0 - fh) * (1.0 - fw);
          const dtype w01 = fw * (1.0 - fh);
          const dtype w10 = fh * (1.0 - fw);
          const dtype w11 = fw * fh;
          dtype tl =
              src[src_index + w_start * src_stride_w + h_start * src_stride_h];
          dtype tr =
              src[src_index + w_end * src_stride_w + h_start * src_stride_h];
          dtype bl =
              src[src_index + w_start * src_stride_w + h_end * src_stride_h];
          dtype br =
              src[src_index + w_end * src_stride_w + h_end * src_stride_h];
          int dst_index = n * dst_stride_batch + c * dst_stride_c +
                          h * dst_stride_h + w * dst_stride_w;
          dst[dst_index] =
              static_cast<dtype>(w00 * tl + w01 * tr + w10 * bl + w11 * br);
        }
      }
    }
  }
}

class BilinearInterpComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string input0_ = "X";
  std::string sizetensor0_ = "SizeTensor0";
  std::string sizetensor1_ = "SizeTensor1";
  std::string input_scale_ = "Scale";
  std::string input1_ = "OutSize";
  std::string output_ = "Out";

  float height_scale_ = 0.f;
  float width_scale_ = 0.f;
  int out_height_ = -1;
  int out_width_ = -1;
  int outsize_height_ = -1;
  int outsize_width_ = -1;
  bool align_corners_ = true;
  std::string interp_method_ = "Bilinear";
  DDim _dims0_{{1, 1, 16, 16}};
  DDim _dims1_{{2}};
  DDim sizetensor_dims_{{1}};
  DDim scale_dims_{{1}};

 public:
  BilinearInterpComputeTester(const Place& place,
                              const std::string& alias,
                              float scale,
                              int out_height,
                              int out_width,
                              int outsize_height,
                              int outsize_width,
                              bool align_corners,
                              std::string interp_method)
      : TestCase(place, alias),
        height_scale_(scale),
        width_scale_(scale),
        out_height_(out_height),
        out_width_(out_width),
        outsize_height_(outsize_height),
        outsize_width_(outsize_width),
        align_corners_(align_corners),
        interp_method_(interp_method) {}

  void RunBaseline(Scope* scope) override {
    width_scale_ = height_scale_;
    std::vector<const lite::Tensor*> inputs;
    inputs.emplace_back(scope->FindTensor(input0_));
    if (outsize_height_ > 0 && outsize_width_ > 0) {
      inputs.emplace_back(scope->FindTensor(input1_));
    }
    std::vector<const lite::Tensor*> SizeTensor;
    if (outsize_height_ > 0 && outsize_width_ > 0) {
      SizeTensor.emplace_back(scope->FindTensor(sizetensor0_));
      SizeTensor.emplace_back(scope->FindTensor(sizetensor1_));
    }
    const lite::Tensor* input_scale = scope->FindTensor(input_scale_);
    float scale = height_scale_;
    int in_h = inputs[0]->dims()[2];
    int in_w = inputs[0]->dims()[3];
    if (SizeTensor.size() > 0) {
      auto new_size = get_new_shape(SizeTensor);
      out_height_ = new_size[0];
      out_width_ = new_size[1];
    } else {
      auto scale_tensor = input_scale;
      if (scale_tensor != nullptr) {
        auto scale_data = get_new_data_from_tensor<float>(scale_tensor);
        scale = scale_data[0];
      }
      if (scale > 0) {
        out_height_ = static_cast<int>(in_h * scale);
        out_width_ = static_cast<int>(in_w * scale);
      }
      if (inputs.size() > 1) {
        auto out_size = inputs[1];
        auto out_size_data = get_new_data_from_tensor<int>(out_size);
        out_height_ = out_size_data[0];
        out_width_ = out_size_data[1];
      }
    }
    float height_scale = scale;
    float width_scale = scale;

    if (out_width_ != -1 && out_height_ != -1) {
      height_scale_ = static_cast<float>(out_height_ / inputs[0]->dims()[2]);
      width_scale_ = static_cast<float>(out_width_ / inputs[0]->dims()[3]);
    }
    auto* outputs = scope->NewTensor(output_);
    CHECK(outputs);
    int num_cout = inputs[0]->dims()[0];
    int c_cout = inputs[0]->dims()[1];
    outputs->Resize({num_cout, c_cout, out_height_, out_width_});
    if (align_corners_) {
      resize_bilinear_align<float>(inputs, outputs);
    } else {
      resize_bilinear_no_align<float>(inputs, outputs);
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("bilinear_interp");
    op_desc->SetInput("X", {input0_});
    if (outsize_height_ > 0 && outsize_width_ > 0) {
      op_desc->SetInput("OutSize", {input1_});
      op_desc->SetInput("SizeTensor", {sizetensor0_, sizetensor1_});
    }
    if (height_scale_ > 0) {
      op_desc->SetInput("Scale", {input_scale_});
    }
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

    if (outsize_height_ > 0 && outsize_width_ > 0) {
      std::vector<int> data1(2);
      data1[0] = outsize_height_;
      data1[1] = outsize_width_;
      SetCommonTensor(input1_, _dims1_, data1.data());

      std::vector<int> sizetensor_data(1);
      sizetensor_data[0] = outsize_height_;
      SetCommonTensor(sizetensor0_, sizetensor_dims_, sizetensor_data.data());

      sizetensor_data[0] = outsize_width_;
      SetCommonTensor(sizetensor1_, sizetensor_dims_, sizetensor_data.data());
    }

    if (height_scale_ > 0) {
      std::vector<float> scale_data(1);
      scale_data[0] = height_scale_;
      SetCommonTensor(input_scale_, scale_dims_, scale_data.data());
    }
  }
};

void test_bilinear_interp(Place place) {
  std::string interp_method = "Bilinear";
  for (float scale : {2., 1., 0.3}) {
    for (bool align_corners : {true, false}) {
      std::unique_ptr<arena::TestCase> tester(new BilinearInterpComputeTester(
          place, "def", scale, -1, -1, -1, -1, align_corners, interp_method));
      arena::Arena arena(std::move(tester), place, 5e-5);
      arena.TestPrecision();
    }
  }
  for (int out_height : {8, 16, 24}) {
    for (int out_width : {8, 16, 24}) {
      for (bool align_corners : {true, false}) {
        std::unique_ptr<arena::TestCase> tester(
            new BilinearInterpComputeTester(place,
                                            "def",
                                            0,
                                            out_height,
                                            out_width,
                                            -1,
                                            -1,
                                            align_corners,
                                            interp_method));
        arena::Arena arena(std::move(tester), place, 5e-5);
        arena.TestPrecision();
      }
    }
  }
  for (int outsize_height : {8, 16, 24}) {
    for (int outsize_width : {8, 16, 24}) {
      for (bool align_corners : {true, false}) {
        std::unique_ptr<arena::TestCase> tester(
            new BilinearInterpComputeTester(place,
                                            "def",
                                            0,
                                            -1,
                                            -1,
                                            outsize_height,
                                            outsize_width,
                                            align_corners,
                                            interp_method));
        arena::Arena arena(std::move(tester), place, 5e-5);
        arena.TestPrecision();
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
