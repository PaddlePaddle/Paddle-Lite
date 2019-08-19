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
#include "lite/core/arena/framework.h"

namespace paddle {
namespace lite {

inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

template <typename Dtype>
void col2im(const Dtype* data_col,
            const int channels,
            const int height,
            const int width,
            const int kernel_h,
            const int kernel_w,
            const int pad_h,
            const int pad_w,
            const int stride_h,
            const int stride_w,
            const int dilation_h,
            const int dilation_w,
            Dtype* data_im) {
  memset(data_im, 0, height * width * channels * sizeof(float));
  const int output_h =
      (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w =
      (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            data_col += output_w;
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                data_im[input_row * width + input_col] += *data_col;
              }
              data_col++;
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

template <typename Dtype>
void fill_bias_relu(Dtype* tensor,
                    const Dtype* bias,
                    int channel,
                    int channel_size,
                    bool flag_bias,
                    bool flag_relu);

template <>
void fill_bias_relu<float>(float* tensor,
                           const float* bias,
                           int channel,
                           int channel_size,
                           bool flag_bias,
                           bool flag_relu) {
  float* data = tensor;
  if (flag_relu) {
    for (int j = 0; j < channel; ++j) {
      float bias_data = flag_bias ? bias[j] : 0.f;
      for (int i = 0; i < channel_size; i++) {
        data[i] += bias_data;
        data[i] = data[i] > 0 ? data[i] : 0.f;
      }
      data += channel_size;
    }
  } else {
    for (int j = 0; j < channel; ++j) {
      float bias_data = flag_bias ? bias[j] : 0.f;
      for (int i = 0; i < channel_size; i++) {
        data[i] += bias_data;
      }
      data += channel_size;
    }
  }
}

template <typename type, typename type2>
static void basic_gemm(int m,
                       int n,
                       int k,
                       const type* a,
                       const type* b,
                       const type2* bias,
                       type2* c,
                       type2 alpha,
                       type2 beta,
                       bool trans_a = false,
                       bool trans_b = false,
                       bool flag_bias = false,
                       bool flag_relu = false) {
#pragma omp parallel for
  for (int i = 0; i < m; ++i) {
    type2 bias_data = (type2)0;
    if (flag_bias) {
      bias_data = bias[i];
    }
    for (int j = 0; j < n; ++j) {
      type2 sum = static_cast<type2>(0);
      for (int l = 0; l < k; ++l) {
        type av;
        type bv;
        if (trans_a) {
          av = a[l * m + i];
        } else {
          av = a[i * k + l];
        }
        if (trans_b) {
          bv = b[j * k + l];
        } else {
          bv = b[l * n + j];
        }
        sum += av * bv;
      }
      type2 tmp = alpha * sum + beta * c[i * n + j] + bias_data;
      if (flag_relu) {
        c[i * n + j] = tmp > (type2)0 ? tmp : (type2)0;
      } else {
        c[i * n + j] = tmp;
      }
    }
  }
}

//! for float, dtype1 and type2 is float
//! for int8, dytpe1 is char, dtype2 is int
template <typename Dtype1, typename Dtype2>
bool deconv_basic(const Dtype1* din,
                  Dtype2* dout,
                  int num,
                  int chout,
                  int hout,
                  int wout,
                  int chin,
                  int hin,
                  int win,
                  const Dtype1* weights,
                  const Dtype2* bias,
                  int group,
                  int kernel_w,
                  int kernel_h,
                  int stride_w,
                  int stride_h,
                  int dila_w,
                  int dila_h,
                  int pad_w,
                  int pad_h,
                  bool flag_bias,
                  bool flag_relu) {
  int m = chout * kernel_w * kernel_h / group;
  int n = hin * win;
  int k = chin / group;

  if (chin != chout || group != chin) {
    CHECK_OR_FALSE(chin % group == 0);
    CHECK_OR_FALSE(chout % group == 0);
  }
  lite::Tensor workspace_tensor;
  std::vector<int64_t> wt_shape = {1, 1, 1, group * m * n};
  workspace_tensor.Resize(wt_shape);
  auto* workspace_ptr = workspace_tensor.mutable_data<Dtype2>();

  int group_size_in = win * hin * chin / group;
  int group_size_out = wout * hout * chout / group;
  int group_size_coldata = m * n;
  int group_size_weights = chin * chout * kernel_w * kernel_h / (group * group);
  bool flag_1x1s1p1 = (kernel_w == 1) && (kernel_h == 1) && (stride_h == 1) &&
                      (stride_w == 1) && (pad_w == 1) && (pad_h == 1) &&
                      (dila_w == 1) && (dila_h == 1);

  for (int i = 0; i < num; ++i) {
    const Dtype1* din_batch = din + i * chin * hin * win;
    Dtype2* dout_batch = dout + i * chout * hout * wout;

    Dtype2* col_data = workspace_ptr;
    if (flag_1x1s1p1) {
      col_data = dout_batch;
    }
    memset(col_data, 0, sizeof(Dtype2) * group_size_coldata);
    for (int g = 0; g < group; ++g) {
      const Dtype1* din_group = din_batch + g * group_size_in;
      const Dtype1* weights_group = weights + g * group_size_weights;
      Dtype2* coldata_group = col_data + g * group_size_coldata;
      basic_gemm<Dtype1, Dtype2>(m,
                                 n,
                                 k,
                                 weights_group,
                                 din_group,
                                 nullptr,
                                 coldata_group,
                                 (Dtype2)1,
                                 (Dtype2)0,
                                 true,
                                 false,
                                 false,
                                 (!flag_bias && flag_relu));
    }
    if (!flag_1x1s1p1) {
      col2im(col_data,
             chout,
             hout,
             wout,
             kernel_h,
             kernel_w,
             pad_h,
             pad_w,
             stride_h,
             stride_w,
             dila_h,
             dila_w,
             dout_batch);
    }
    if (flag_bias) {
      fill_bias_relu(
          dout_batch, bias, chout, wout * hout, flag_bias, flag_relu);
    }
  }
  return true;
}

class Conv2DTransposeComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string x_ = "x";
  std::string output_ = "out";
  std::string filter_ = "filter";
  std::string bias_ = "bias";

  std::vector<int> strides_{1, 1};
  std::vector<int> paddings_{0, 0};
  int groups_{1};
  std::vector<int> dilations_{1, 1};
  bool flag_relu_{false};

  int n_ = 1;
  int ic_ = 1;
  int oc_ = 1;
  int ih_ = 9;
  int iw_ = 9;
  bool flag_bias_ = false;
  int ks_ = 1;

 public:
  Conv2DTransposeComputeTester(const Place& place,
                               const std::string& alias,
                               int n,
                               int ic,
                               int oc,
                               int ih,
                               int iw,
                               bool flag_bias,
                               bool flag_relu,
                               int dilation,
                               int stride,
                               int padding,
                               int ks,
                               int groups)
      : TestCase(place, alias) {
    n_ = n;
    ic_ = ic;
    oc_ = oc;
    ih_ = ih;
    iw_ = iw;
    ks_ = ks;
    flag_bias_ = flag_bias;

    strides_ = std::vector<int>({stride, stride});
    paddings_ = std::vector<int>({padding, padding});
    groups_ = groups;
    dilations_ = std::vector<int>({dilation, dilation});
    flag_relu_ = flag_relu;
  }

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(output_);
    CHECK(out);
    int oh = (ih_ - 1) * strides_[0] - 2 * paddings_[0] +
             dilations_[0] * (ks_ - 1) + 1;
    int ow = (iw_ - 1) * strides_[1] - 2 * paddings_[1] +
             dilations_[1] * (ks_ - 1) + 1;
    CHECK(oh > 0 || ow > 0);

    std::vector<int64_t> output_shape = {n_, oc_, oh, ow};
    DDim output_dims(output_shape);
    out->Resize(output_dims);
    auto* output_data = out->mutable_data<float>();

    auto* x = scope->FindTensor(x_);
    const auto* x_data = x->data<float>();
    auto* filter = scope->FindTensor(filter_);
    const auto* filter_data = filter->data<float>();
    const float* bias_data = nullptr;
    if (flag_bias_) {
      auto* bias = scope->FindTensor(bias_);
      bias_data = bias->data<float>();
    }

    deconv_basic<float, float>(x_data,
                               output_data,
                               n_,
                               oc_,
                               oh,
                               ow,
                               ic_,
                               ih_,
                               iw_,
                               filter_data,
                               bias_data,
                               groups_,
                               ks_,
                               ks_,
                               strides_[1],
                               strides_[0],
                               dilations_[1],
                               dilations_[0],
                               paddings_[1],
                               paddings_[0],
                               flag_bias_,
                               flag_relu_);
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("conv2d_transpose");
    op_desc->SetInput("Input", {x_});
    op_desc->SetInput("Filter", {filter_});
    op_desc->SetOutput("Output", {output_});
    op_desc->SetAttr("strides", strides_);
    op_desc->SetAttr("paddings", paddings_);
    op_desc->SetAttr("groups", groups_);
    op_desc->SetAttr("dilations", dilations_);
    if (flag_bias_) {
      op_desc->SetInput("Bias", {bias_});
    }
    op_desc->SetAttr("fuse_relu", flag_relu_);
  }

  void PrepareData() override {
    std::vector<int64_t> input_shape = {n_, ic_, ih_, iw_};
    std::vector<int64_t> filter_shape = {ic_, oc_ / groups_, ks_, ks_};
    std::vector<int64_t> bias_shape = {1, oc_, 1, 1};

    // x tensor
    DDim x_dims(input_shape);
    std::vector<float> x_data(x_dims.production());
    for (int i = 0; i < x_dims.production(); i++) {
      float sign = i % 3 == 0 ? -1.0f : 1.0f;
      x_data[i] = sign * static_cast<float>(i % 128) * 0.013f + 0.001;
    }
    SetCommonTensor(x_, x_dims, x_data.data());

    // filter tensor
    DDim filter_dims(filter_shape);
    std::vector<float> filter_data(filter_dims.production());
    for (int i = 0; i < filter_dims.production(); i++) {
      float sign = i % 3 == 0 ? -1.0f : 1.0f;
      filter_data[i] = sign * static_cast<float>(i % 128) * 0.01f + 0.001;
    }
    SetCommonTensor(filter_, filter_dims, filter_data.data());

    // bias tensor
    if (flag_bias_) {
      DDim bias_dims(bias_shape);
      std::vector<float> bias_data(bias_dims.production());
      for (int i = 0; i < bias_dims.production(); i++) {
        float sign = i % 3 == 0 ? -1.0f : 1.0f;
        bias_data[i] = sign * static_cast<float>(i % 128) * 0.01f + 0.001;
      }
      SetCommonTensor(bias_, bias_dims, bias_data.data());
    }
  }
};

TEST(conv2d_transpose, precision) {
  LOG(INFO) << "test conv2d_transpose op";
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
  for (auto n : {1, 2}) {
    for (auto ic : {1, 4 /*, 128*/}) {
      for (auto oc : {1, 4 /*, 128*/}) {
        LOG(INFO) << "n:" << n << ",ic:" << ic << ",oc:" << oc;
        for (auto ih : {8, 16 /*, 56 , 112, 224, 512*/}) {
          for (auto iw : {8, 16 /*, 56, 112, 224, 512*/}) {
            for (auto flag_bias : {false, true}) {
              for (auto flag_relu : {false, true}) {
                for (auto dilation : {1, 2}) {
                  for (auto stride : {1, 2}) {
                    for (auto padding : {0, 2}) {
                      for (auto ks : {2, 5}) {
                        for (auto group : {1, 2}) {
                          // obtain shape
                          // LOG(INFO) << "n:" << n << ",ic:" << ic << ",oc:" <<
                          // oc
                          //           << ",ih:" << ih << ",iw:" << iw
                          //           << ",flag_bias:" << flag_bias
                          //           << ",flag_relu:" << flag_relu
                          //           << ",dila:" << dilation
                          //           << ",stride:" << stride
                          //           << ",padding:" << padding << ",ks:" << ks
                          //           << ",group:" << group;
                          if (ic % group != 0 || oc % group != 0) {
                            group = 1;
                          }
                          std::unique_ptr<arena::TestCase> tester(
                              new Conv2DTransposeComputeTester(place,
                                                               "def",
                                                               n,
                                                               ic,
                                                               oc,
                                                               ih,
                                                               iw,
                                                               flag_bias,
                                                               flag_relu,
                                                               dilation,
                                                               stride,
                                                               padding,
                                                               ks,
                                                               group));
                          arena::Arena arena(std::move(tester), place, 2e-5);
                          arena.TestPrecision();
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
#endif
}

}  // namespace lite
}  // namespace paddle
