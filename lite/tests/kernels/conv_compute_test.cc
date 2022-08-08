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
#include "lite/tests/utils/fill_data.h"

namespace paddle {
namespace lite {

class ConvComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string op_type_ = "conv2d";
  std::string input_ = "input";
  std::string filter_ = "filter";
  std::string output_ = "output";
  DDim dims_;

  int out_channels_ = 1;
  int ksize_ = 3;
  std::vector<int> strides_{1, 1};
  std::vector<int> paddings_{0, 0};
  int groups_ = 1;
  std::vector<int> dilations_{1, 1};
  std::string padding_algorithm_;
  bool with_bias_ = false;
  std::string bias_ = "bias";
  bool with_act_ = false;
  std::string act_type_;
  float leaky_relu_alpha_ = 0.1;
  bool with_depthwise_ = false;
  bool with_fuse_relu_ = false;
  float with_fuse_relu6_ = 6.f;
  float hard_swish_threshold_ = 6.0;
  float hard_swish_scale_ = 6.0;
  float hard_swish_offset_ = 3.0;

 public:
  ConvComputeTester(const Place& place,
                    const std::string& alias,
                    DDim dims,
                    int out_channels = 1,
                    int ksize = 3,
                    std::vector<int> strides = {1, 1},
                    std::vector<int> paddings = {0, 0},
                    int groups = 1,
                    std::vector<int> dilations = {1, 1},
                    std::string padding_algorithm = "",
                    bool with_bias = false,
                    bool with_act = false,
                    std::string act_type = "",
                    float leaky_relu_alpha = 0.1)
      : TestCase(place, alias),
        dims_(dims),
        out_channels_(out_channels),
        ksize_(ksize),
        strides_(strides),
        paddings_(paddings),
        groups_(groups),
        dilations_(dilations),
        padding_algorithm_(padding_algorithm),
        with_bias_(with_bias),
        with_act_(with_act),
        act_type_(act_type),
        leaky_relu_alpha_(leaky_relu_alpha) {}

  void RunBaseline(Scope* scope) override {
    auto* input = scope->FindTensor(input_);
    auto* filter = scope->FindTensor(filter_);
    auto input_dims = input->dims();
    auto filter_dims = filter->dims();

    auto* output = scope->NewTensor(output_);
    CHECK(output);

    if (paddings_.size() == 2L) {
      paddings_.insert(paddings_.begin(), paddings_[0]);
      paddings_.insert(paddings_.begin() + 2, paddings_[2]);
    }
    if (padding_algorithm_ == "SAME") {
      for (size_t i = 0; i < strides_.size(); ++i) {
        int out_size = (input_dims[i + 2] + strides_[i] - 1) / strides_[i];
        int pad_sum =
            std::max((out_size - 1) * strides_[i] + ksize_ - input_dims[i + 2],
                     (int64_t)0);
        int pad_0 = pad_sum / 2;
        int pad_1 = pad_sum - pad_0;
        // pad
        *(paddings_.begin() + i * 2) = pad_0;
        *(paddings_.begin() + i * 2 + 1) = pad_1;
        // dilation
        *(dilations_.begin() + i) = 1;
      }
    } else if (padding_algorithm_ == "VALID") {
      for (auto& it : paddings_) {
        it = 0;
      }
    }
    std::vector<int64_t> output_shape({input_dims[0], filter_dims[0]});
    for (size_t i = 0; i < strides_.size(); ++i) {
      const int dkernel = dilations_[i] * (filter_dims[i + 2] - 1) + 1;
      int output_size = (input_dims[i + 2] +
                         (paddings_[i * 2] + paddings_[i * 2 + 1]) - dkernel) /
                            strides_[i] +
                        1;
      output_shape.push_back(output_size);
    }
    output->Resize(DDim(output_shape));
    auto output_dims = output->dims();

    auto input_data = input->data<float>();
    auto filter_data = filter->data<float>();
    auto output_data = output->mutable_data<float>();
    int kernel_w = filter_dims[3];
    int kernel_h = filter_dims[2];
    int stride_w = strides_[1];
    int stride_h = strides_[0];
    int dila_w = dilations_[1];
    int dila_h = dilations_[0];
    int pad_w = paddings_[2];
    int pad_h = paddings_[0];
    int batch_size = input_dims[0];
    int in_ch_size = input_dims[1];
    int in_h = input_dims[2];
    int in_w = input_dims[3];
    int out_ch_size = output_dims[1];
    int out_h = output_dims[2];
    int out_w = output_dims[3];
    int out_c_group = out_ch_size / groups_;
    int in_c_group = in_ch_size / groups_;

    const float* bias_data = nullptr;
    bool is_channel_bias = true;
    if (with_bias_) {
      auto bias = scope->FindTensor(bias_);
      bias_data = bias->data<float>();
    }
    for (int n = 0; n < batch_size; ++n) {
      for (int g = 0; g < groups_; ++g) {
        for (int oc = 0; oc < out_c_group; ++oc) {
          for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
              int out_idx = n * groups_ * out_c_group * out_h * out_w +
                            g * out_c_group * out_h * out_w +
                            oc * out_h * out_w + oh * out_w + ow;
              float out_value =
                  bias_data != nullptr
                      ? (is_channel_bias ? bias_data[g * out_c_group + oc]
                                         : bias_data[out_idx])
                      : 0;
              for (int ic = 0; ic < in_c_group; ++ic) {
                for (int kh = 0; kh < kernel_h; ++kh) {
                  for (int kw = 0; kw < kernel_w; ++kw) {
                    int iw = ow * stride_w - pad_w + kw * (dila_w);
                    int ih = oh * stride_h - pad_h + kh * (dila_h);
                    if (iw < 0 || iw >= in_w) continue;
                    if (ih < 0 || ih >= in_h) continue;
                    int in_idx = n * in_ch_size * in_h * in_w +
                                 g * in_c_group * in_h * in_w +
                                 ic * in_h * in_w + ih * in_w + iw;
                    int filter_idx =
                        g * out_c_group * in_c_group * kernel_h * kernel_w +
                        oc * in_c_group * kernel_h * kernel_w +
                        ic * kernel_h * kernel_w + kh * kernel_w + kw;
                    out_value += input_data[in_idx] * filter_data[filter_idx];
                  }
                }
              }
              if (with_act_) {
                if (act_type_ == "relu") {
                  out_value = out_value > 0 ? out_value : 0;
                } else if (act_type_ == "relu6") {
                  out_value = std::min(std::max(0.f, out_value), 6.f);
                } else if (act_type_ == "leaky_relu") {
                  out_value =
                      std::max(out_value, out_value * leaky_relu_alpha_);
                } else if (act_type_ == "hard_swish") {
                  float max_value =
                      std::max(0.f, out_value + hard_swish_offset_);
                  float min_value = std::min(max_value, hard_swish_threshold_);
                  out_value = min_value * out_value / hard_swish_scale_;
                } else {
                  LOG(FATAL) << " activation type " << act_type_
                             << "not supported in conv test";
                }
              }
              output_data[out_idx] = out_value;
            }
          }
        }
      }
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType(with_depthwise_ ? "depthwise_conv2d" : "conv2d");
    op_desc->SetInput("Input", {input_});
    op_desc->SetInput("Filter", {filter_});
    if (with_bias_) {
      op_desc->SetInput("Bias", {bias_});
    }
    op_desc->SetOutput("Output", {output_});
    op_desc->SetAttr("strides", strides_);
    op_desc->SetAttr("paddings", paddings_);
    op_desc->SetAttr("groups", groups_);
    op_desc->SetAttr("dilations", dilations_);
    op_desc->SetAttr("fuse_relu", with_fuse_relu_);
    if (!padding_algorithm_.empty()) {
      op_desc->SetAttr("padding_algorithm", padding_algorithm_);
    }
    if (with_act_) {
      op_desc->SetAttr("with_act", with_act_);
      op_desc->SetAttr("act_type", act_type_);
      if (act_type_ == "relu6") {
        op_desc->SetAttr("fuse_brelu_threshold", with_fuse_relu6_);
      }
      if (act_type_ == "leaky_relu") {
        op_desc->SetAttr("leaky_relu_alpha", leaky_relu_alpha_);
      }
      if (act_type_ == "hard_swish") {
        op_desc->SetAttr("hard_swish_threshold", hard_swish_threshold_);
        op_desc->SetAttr("hard_swish_scale", hard_swish_scale_);
        op_desc->SetAttr("hard_swish_offset", hard_swish_offset_);
      }
    }
  }

  void PrepareData() override {
    std::vector<float> din(dims_.production());
    fill_data_rand(din.data(), -1.f, 1.f, dims_.production());
    SetCommonTensor(input_, dims_, din.data());

    DDim filter_dims(std::vector<int64_t>{
        out_channels_, dims_[1] / groups_, ksize_, ksize_});
    std::vector<float> dfilter(filter_dims.production());
    fill_data_rand(dfilter.data(), -1.f, 1.f, filter_dims.production());
    SetCommonTensor(filter_, filter_dims, dfilter.data(), {}, true);

    if (with_bias_) {
      DDim bias_dims(std::vector<int64_t>{out_channels_});
      std::vector<float> dbias(bias_dims.production());
      fill_data_rand(dbias.data(), -1.f, 1.f, bias_dims.production());
      SetCommonTensor(bias_, bias_dims, dbias.data(), {}, true);
    }
  }
};

void TestConvKsize(Place place, float abs_error = 2e-5) {
  for (auto dims :
       std::vector<std::vector<int64_t>>{{1, 2, 7, 8}, {5, 6, 17, 18}}) {
    for (auto out_channels : {1, 3}) {
      for (auto ksize : {1, 3, 5, 7}) {
        std::unique_ptr<arena::TestCase> tester(new ConvComputeTester(
            place, "def", DDim(dims), out_channels, ksize));
        arena::Arena arena(std::move(tester), place, abs_error);
        arena.TestPrecision();
      }
    }
  }
}

void TestConvGroups(Place place, float abs_error = 2e-5) {
  for (auto dims :
       std::vector<std::vector<int64_t>>{{1, 6, 3, 4}, {5, 12, 7, 8}}) {
    for (auto out_channels : {2, 3, 6}) {
      for (auto groups : {2, 3, 6}) {
#if defined(LITE_WITH_NPU) || defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU) || \
    defined(NNADAPTER_WITH_HUAWEI_KIRIN_NPU) ||                            \
    defined(NNADAPTER_WITH_NVIDIA_TENSORRT) ||                             \
    defined(NNADAPTER_WITH_INTEL_OPENVINO) ||                              \
    defined(NNADAPTER_WITH_QUALCOMM_QNN)
        if (out_channels % groups != 0) continue;
#endif
        std::unique_ptr<arena::TestCase> tester(new ConvComputeTester(
            place, "def", DDim(dims), out_channels, 3, {1, 1}, {0, 0}, groups));
        arena::Arena arena(std::move(tester), place, abs_error);
        arena.TestPrecision();
      }
    }
  }
}

void TestConvDilations(Place place, float abs_error = 2e-5) {
  for (auto dims :
       std::vector<std::vector<int64_t>>{{1, 2, 5, 6}, {5, 6, 9, 10}}) {
    for (auto out_channels : {1, 3}) {
      for (auto dilations : std::vector<std::vector<int>>{{2, 2}, {1, 2}}) {
        std::unique_ptr<arena::TestCase> tester(
            new ConvComputeTester(place,
                                  "def",
                                  DDim(dims),
                                  out_channels,
                                  3,
                                  {1, 1},
                                  {0, 0},
                                  1,
                                  dilations));
        arena::Arena arena(std::move(tester), place, abs_error);
        arena.TestPrecision();
      }
    }
  }
}

void TestConvStrides(Place place, float abs_error = 2e-5) {
  for (auto dims :
       std::vector<std::vector<int64_t>>{{1, 2, 3, 4}, {5, 6, 7, 8}}) {
    for (auto out_channels : {1, 3}) {
      for (auto strides :
           std::vector<std::vector<int>>{{2, 2}, {3, 3}, {1, 2}, {3, 1}}) {
        std::unique_ptr<arena::TestCase> tester(new ConvComputeTester(
            place, "def", DDim(dims), out_channels, 3, strides));
        arena::Arena arena(std::move(tester), place, abs_error);
        arena.TestPrecision();
      }
    }
  }
}

void TestConvPaddings(Place place, float abs_error = 2e-5) {
  for (auto dims :
       std::vector<std::vector<int64_t>>{{1, 2, 3, 4}, {5, 6, 7, 8}}) {
    for (auto out_channels : {1, 3}) {
      for (auto paddings : std::vector<std::vector<int>>{
               {1, 1}, {2, 2}, {1, 0, 0, 1}, {1, 2, 0, 1}}) {
        std::unique_ptr<arena::TestCase> tester(new ConvComputeTester(
            place, "def", DDim(dims), out_channels, 3, {1, 1}, paddings));
        arena::Arena arena(std::move(tester), place, abs_error);
        arena.TestPrecision();
      }
    }
  }
}

void TestConvPaddingAlgorithm(Place place, float abs_error = 2e-5) {
  for (auto dims :
       std::vector<std::vector<int64_t>>{{1, 2, 3, 4}, {5, 6, 7, 8}}) {
    for (auto out_channels : {1, 3}) {
      for (auto padding_algorithm : std::vector<std::string>{"VALID", "SAME"}) {
        std::unique_ptr<arena::TestCase> tester(
            new ConvComputeTester(place,
                                  "def",
                                  DDim(dims),
                                  out_channels,
                                  3,
                                  {1, 1},
                                  {0, 0},
                                  1,
                                  {1, 1},
                                  padding_algorithm));
        arena::Arena arena(std::move(tester), place, abs_error);
        arena.TestPrecision();
      }
    }
  }
}

void TestConvBias(Place place, float abs_error = 2e-5) {
  for (auto dims :
       std::vector<std::vector<int64_t>>{{1, 2, 3, 4}, {5, 6, 7, 8}}) {
    for (auto out_channels : {1, 3}) {
      std::unique_ptr<arena::TestCase> tester(
          new ConvComputeTester(place,
                                "def",
                                DDim(dims),
                                out_channels,
                                3,
                                {1, 1},
                                {0, 0},
                                1,
                                {1, 1},
                                "",
                                true));
      arena::Arena arena(std::move(tester), place, abs_error);
      arena.TestPrecision();
    }
  }
}

void TestConvAct(Place place, float abs_error = 2e-5) {
  for (auto dims :
       std::vector<std::vector<int64_t>>{{1, 2, 3, 4}, {5, 6, 7, 8}}) {
    for (auto out_channels : {1, 3}) {
      std::unique_ptr<arena::TestCase> tester0(
          new ConvComputeTester(place,
                                "def",
                                DDim(dims),
                                out_channels,
                                3,
                                {1, 1},
                                {0, 0},
                                1,
                                {1, 1},
                                "",
                                false,
                                true,
                                "relu"));
      arena::Arena arena0(std::move(tester0), place, abs_error);
      arena0.TestPrecision();
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU) || \
    defined(NNADAPTER_WITH_NVIDIA_TENSORRT) ||   \
    defined(NNADAPTER_WITH_QUALCOMM_QNN)
      continue;
#endif
      std::unique_ptr<arena::TestCase> tester1(
          new ConvComputeTester(place,
                                "def",
                                DDim(dims),
                                out_channels,
                                3,
                                {1, 1},
                                {0, 0},
                                1,
                                {1, 1},
                                "",
                                false,
                                true,
                                "leaky_relu",
                                0.1));
      arena::Arena arena1(std::move(tester1), place, abs_error);
      arena1.TestPrecision();
    }
  }
}

void TestConvDepthwise(Place place, float abs_error = 2e-5) {
  // Using a limited set can prevent unit test timeout and reduce CI
  // time-consuming
  for (int64_t n : {1, 3, 4}) {
    for (auto win : {3, 5, 7, 12, 16}) {
      for (auto kw : {3, 5}) {
        win = std::max(win, kw);
        for (auto ch : {2, 4, 7, 16}) {
          std::vector<int64_t> dims{n, ch, win, win};
          for (auto stride : {1, 2}) {
            for (auto pad : {0, 1}) {
              for (auto bias : {false, true}) {
                for (auto act : {"hard_swish", "relu", "relu6", "leaky_relu"}) {
#if defined(NNADAPTER_WITH_NVIDIA_TENSORRT)
                  if (strcmp(act, "hard_swish") || strcmp(act, "leaky_relu"))
                    continue;
#endif
#if defined(NNADAPTER_WITH_HUAWEI_KIRIN_NPU)
                  if (act == "hard_swish") continue;
#endif
#if defined(NNADAPTER_WITH_QUALCOMM_QNN)
                  if (strcmp(act, "hard_swish") || strcmp(act, "leaky_relu") ||
                      strcmp(act, "relu6"))
                    continue;
#endif
                  std::unique_ptr<arena::TestCase> tester(
                      new ConvComputeTester(place,
                                            "def",
                                            DDim(dims),
                                            ch,
                                            kw,
                                            {stride, stride},
                                            {pad, pad},
                                            ch,
                                            {1, 1},
                                            "",
                                            bias,
                                            true,
                                            act));
                  arena::Arena arena(std::move(tester), place, abs_error);
                  arena.TestPrecision();
                }
              }
            }
          }
        }
      }
    }
  }
  for (auto pad : {0, 1, 2}) {
    for (auto stride : {1, 2}) {
      std::unique_ptr<arena::TestCase> tester(
          new ConvComputeTester(place,
                                "def",
                                DDim({1, 16, 16, 25}),
                                16,
                                3,
                                {stride, stride},
                                {pad, pad},
                                16));
      arena::Arena arena(std::move(tester), place, abs_error);
      arena.TestPrecision();
    }
  }
  std::unique_ptr<arena::TestCase> tester(new ConvComputeTester(
      place, "def", DDim({1, 40, 16, 50}), 40, 3, {2, 1}, {1, 1}, 40));
  arena::Arena arena(std::move(tester), place, abs_error);
  arena.TestPrecision();
}

TEST(Conv2d, precision) {
  float abs_error = 2e-5;
  Place place;
#if defined(LITE_WITH_NNADAPTER)
  place = TARGET(kNNAdapter);
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  abs_error = 5e-2;
#elif defined(NNADAPTER_WITH_HUAWEI_KIRIN_NPU)
  abs_error = 1e-1;
#elif defined(NNADAPTER_WITH_QUALCOMM_QNN)
  abs_error = 1e-2;
#elif defined(NNADAPTER_WITH_CAMBRICON_MLU)
  abs_error = 5e-2;
  TestConvKsize(place, abs_error);
  TestConvDilations(place, abs_error);
  TestConvStrides(place, abs_error);
  TestConvPaddings(place, abs_error);
  TestConvBias(place, abs_error);
  return;
#elif defined(NNADAPTER_WITH_VERISILICON_TIMVX)
  abs_error = 5e-2;
  TestConvKsize(place, abs_error);
  return;
#elif defined(NNADAPTER_WITH_ANDROID_NNAPI)
  abs_error = 5e-2;
#elif defined(NNADAPTER_WITH_NVIDIA_TENSORRT)
  abs_error = 2e-5;
#elif defined(NNADAPTER_WITH_INTEL_OPENVINO)
  abs_error = 2e-5;
#else
  return;
#endif
#elif defined(LITE_WITH_NPU)
  place = TARGET(kNPU);
  abs_error = 5e-2;  // Using fp16 in NPU
#elif defined(LITE_WITH_X86)
  place = TARGET(kX86);
  TestConvKsize(place, abs_error);
  TestConvDepthwise(place, abs_error);
  return;
#else
  return;
#endif
  TestConvKsize(place, abs_error);
  TestConvGroups(place, abs_error);
  TestConvDilations(place, abs_error);
  TestConvStrides(place, abs_error);
  TestConvPaddings(place, abs_error);
  TestConvPaddingAlgorithm(place, abs_error);
  TestConvBias(place, abs_error);
  TestConvAct(place, abs_error);
#if !defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU) && \
    !defined(NNADAPTER_WITH_HUAWEI_KIRIN_NPU)
  TestConvDepthwise(place, abs_error);
#endif
}

}  // namespace lite
}  // namespace paddle
