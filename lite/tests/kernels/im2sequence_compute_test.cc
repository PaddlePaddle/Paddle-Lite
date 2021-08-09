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
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/core/test/arena/framework.h"

namespace paddle {
namespace lite {

void im2sequence(const float* input,
                 const int input_c,
                 const int input_h,
                 const int input_w,
                 const int kernel_h,
                 const int kernel_w,
                 const int pad_top,
                 const int pad_bottom,
                 const int pad_left,
                 const int pad_right,
                 const int stride_h,
                 const int stride_w,
                 const int out_h,
                 const int out_w,
                 float* out) {
  int window_size = kernel_h * kernel_w;
  int out_rows = out_h * out_w;
  int out_cols = input_c * window_size;
  int H_pad = input_h + pad_top + pad_bottom;
  int W_pad = input_w + pad_left + pad_right;
  for (int h_id = 0; h_id < out_h; h_id++) {
    for (int w_id = 0; w_id < out_w; w_id++) {
      // consider dilation.
      int start_h = h_id * stride_h - pad_top;
      int start_w = w_id * stride_w - pad_left;
      for (int c_id = 0; c_id < input_c; c_id++) {
        for (int k_h_id = 0; k_h_id < kernel_h; k_h_id++) {
          int in_h_id = start_h + k_h_id;
          bool exceed_flag = (in_h_id < 0) || (in_h_id >= H_pad);
          int out_start_id =
              (h_id * out_w + w_id) * out_cols + c_id * window_size;
          for (int k_w_id = 0; k_w_id < kernel_w; k_w_id++) {
            int in_w_id = start_w + k_w_id;
            exceed_flag = exceed_flag || (in_w_id < 0) || (in_w_id >= W_pad);
            int input_id = (c_id * input_h + in_h_id) * input_w + in_w_id;
            int out_id = out_start_id + k_h_id * kernel_w + k_w_id;
            out[out_id] = exceed_flag ? 0.f : input[input_id];
          }
        }
      }
    }
  }
}

class Im2SequenceComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string input0_ = "x";
  std::string input1_ = "y";
  std::string output_ = "out";
  std::vector<int> paddings_{{0, 0, 0, 0}};
  std::vector<int> kernels_{{1, 1}};
  std::vector<int> strides_{{1, 1}};
  std::vector<int> out_strides_{{1, 1}};
  DDim dims_{{3, 5, 4, 4}};

 public:
  Im2SequenceComputeTester(const Place& place,
                           const std::string& alias,
                           std::vector<int> kernels,
                           std::vector<int> paddings,
                           std::vector<int> strides,
                           std::vector<int> out_strides,
                           DDim dims)
      : TestCase(place, alias),
        paddings_(paddings),
        kernels_(kernels),
        strides_(strides),
        out_strides_(out_strides),
        dims_(dims) {}

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(output_);

    auto* x = scope->FindTensor(input0_);
    const auto* x_data = x->data<float>();
    int im_num = dims_[0];
    int im_size = dims_.production() / im_num;
    int out_cols = dims_[1] * kernels_[0] * kernels_[1];
    int total_rows = 0;
    std::vector<uint64_t> im_offset;
    im_offset.push_back(total_rows);
    /*compute out shape*/
    auto* y = scope->FindTensor(input1_);
    if (y) {
      const auto* y_data = y->data<int>();
      std::vector<int> im_real_h;
      std::vector<int> im_real_w;
      std::vector<int> out_h_vec;
      std::vector<int> out_w_vec;

      for (int im_id = 0; im_id < im_num; im_id++) {
        int real_h = y_data[im_id * 2 + 0];
        int real_w = y_data[im_id * 2 + 1];
        int tmp_real_h = (real_h + out_strides_[0] - 1) / out_strides_[0];
        int tmp_real_w = (real_w + out_strides_[1] - 1) / out_strides_[1];
        im_real_h.push_back(tmp_real_h);
        im_real_w.push_back(tmp_real_w);
        int out_h = (tmp_real_h + paddings_[0] + paddings_[1] - kernels_[0]) /
                        strides_[0] +
                    1;
        int out_w = (tmp_real_w + paddings_[2] + paddings_[3] - kernels_[1]) /
                        strides_[1] +
                    1;
        out_h_vec.push_back(out_h);
        out_w_vec.push_back(out_w);
        total_rows += out_h * out_w;
        im_offset.push_back(total_rows);
      }
      DDim out_dims{{total_rows, out_cols}};
      out->Resize(out_dims);
      auto* o_data = out->mutable_data<float>();
      int out_offset = 0;
      for (int im_id = 0; im_id < im_num; im_id++) {
        im2sequence(x_data + im_id * im_size,
                    dims_[1],
                    dims_[2],
                    dims_[3],
                    kernels_[0],
                    kernels_[1],
                    paddings_[0],
                    paddings_[1],
                    paddings_[2],
                    paddings_[3],
                    strides_[0],
                    strides_[1],
                    out_h_vec[im_id],
                    out_w_vec[im_id],
                    o_data + im_offset[im_id] * out_cols);
      }
    } else {
      int out_h =
          (dims_[2] + paddings_[0] + paddings_[1] - kernels_[0]) / strides_[0] +
          1;
      int out_w =
          (dims_[3] + paddings_[2] + paddings_[3] - kernels_[1]) / strides_[1] +
          1;
      DDim out_dims{{im_num * out_h * out_w, out_cols}};
      out->Resize(out_dims);
      auto* o_data = out->mutable_data<float>();
      for (int im_id = 0; im_id < im_num; im_id++) {
        int out_size_per_im = out_h * out_w * out_cols;
        im2sequence(x_data + im_id * im_size,
                    dims_[1],
                    dims_[2],
                    dims_[3],
                    kernels_[0],
                    kernels_[1],
                    paddings_[0],
                    paddings_[1],
                    paddings_[2],
                    paddings_[3],
                    strides_[0],
                    strides_[1],
                    out_h,
                    out_w,
                    o_data + im_id * out_size_per_im);
        im_offset.push_back(uint64_t(im_id * out_h * out_w));
      }
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("im2sequence");
    op_desc->SetInput("X", {input0_, input1_});
    op_desc->SetOutput("Out", {output_});
    op_desc->SetAttr("kernels", kernels_);
    op_desc->SetAttr("paddings", paddings_);
    op_desc->SetAttr("strides", strides_);
    op_desc->SetAttr("out_strides", out_strides_);
  }

  void PrepareData() override {
    std::vector<float> data(dims_.production());

    for (int i = 0; i < dims_.production(); i++) {
      data[i] = i * 1.1;
    }

    SetCommonTensor(input0_, dims_, data.data());
    int in_h = dims_[2];
    int in_w = dims_[3];
    int in_num = dims_[0];
    std::vector<int> real_im_size(in_num * 2);
    for (int i = 0; i < in_num; i++) {
      int real_h = std::rand() % static_cast<int>(in_h * 0.3) + (in_h * 0.7);
      int real_w = std::rand() % static_cast<int>(in_w * 0.3) + (in_w * 0.7);
      real_im_size[2 * i] = real_h;
      real_im_size[2 * i] = real_w;
    }
    DDim input1_dims{{in_num, 2}};
    SetCommonTensor(input1_, input1_dims, real_im_size.data());
  }
};

void test_im2sequence(Place place) {
  DDimLite dims{{3, 5, 4, 4}};
  for (int kernel : {1}) {
    std::vector<int> kernels{{kernel, kernel}};
    for (int stride : {1}) {
      std::vector<int> strides{{stride, stride}};
      for (int padding : {0}) {
        std::vector<int> paddings{{padding, padding, padding, padding}};
        for (int out_stride : {1}) {
          std::vector<int> out_strides{{out_stride, out_stride}};
          std::unique_ptr<arena::TestCase> tester(new Im2SequenceComputeTester(
              place, "def", kernels, paddings, strides, out_strides, dims));
          arena::Arena arena(std::move(tester), place, 2e-5);
          arena.TestPrecision();
        }
      }
    }
  }
}

TEST(Im2Sequence, precision) {
// #ifdef LITE_WITH_X86
//   Place place(TARGET(kX86));
// #endif
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
  test_im2sequence(place);
#endif
}

}  // namespace lite
}  // namespace paddle
