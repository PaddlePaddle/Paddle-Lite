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
#include "lite/tests/utils/naive_math_impl.h"

namespace paddle {
namespace lite {

class ConvTransposeComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string op_type_ = "conv2d_transpose";
  std::string input_ = "input";
  std::string filter_ = "filter";
  std::string output_ = "output";
  DDim dims_;

  int filter_channels_ = 1;
  std::vector<int> ksize_{3, 3};
  std::vector<int> strides_{1, 1};
  std::vector<int> paddings_{0, 0};
  int groups_ = 1;
  std::vector<int> dilations_{1, 1};
  std::string padding_algorithm_ = "";
  std::vector<int> output_size_{};
  std::vector<int> output_padding_{};
  std::string bias_ = "";
  bool fuse_relu_ = false;

 public:
  ConvTransposeComputeTester(const Place& place,
                             const std::string& alias,
                             DDim dims,
                             int filter_channels = 1,
                             std::vector<int> ksize = {3, 3},
                             std::vector<int> strides = {1, 1},
                             std::vector<int> paddings = {0, 0},
                             int groups = 1,
                             std::vector<int> dilations = {1, 1},
                             std::string padding_algorithm = "",
                             std::vector<int> output_size = {},
                             std::vector<int> output_padding = {},
                             std::string bias = "",
                             bool fuse_relu = false)
      : TestCase(place, alias),
        dims_(dims),
        filter_channels_(filter_channels),
        ksize_(ksize),
        strides_(strides),
        paddings_(paddings),
        groups_(groups),
        dilations_(dilations),
        padding_algorithm_(padding_algorithm),
        output_size_(output_size),
        output_padding_(output_padding),
        bias_(bias),
        fuse_relu_(fuse_relu) {}

  void RunBaseline(Scope* scope) override {
    if (paddings_.size() == 2L) {
      paddings_.insert(paddings_.begin(), paddings_[0]);
      paddings_.insert(paddings_.begin() + 2, paddings_[2]);
    }
    CHECK_EQ(paddings_.size(), 4);

    if (padding_algorithm_ == "SAME") {
      for (size_t i = 0; i < strides_.size(); ++i) {
        int out_size = (dims_[i + 2] + strides_[i] - 1) / strides_[i];
        int pad_sum =
            std::max((out_size - 1) * strides_[i] + ksize_[i] - dims_[i + 2],
                     (int64_t)0);
        int pad_0 = pad_sum / 2;
        int pad_1 = pad_sum - pad_0;
        // pad
        paddings_[i * 2] = pad_0;
        paddings_[i * 2 + 1] = pad_1;
        // dilation
        dilations_[i] = 1;
      }
    } else if (padding_algorithm_ == "VALID") {
      for (auto& it : paddings_) {
        it = 0;
      }
    }

    std::vector<int64_t> output_shape{dims_[0], filter_channels_ * groups_};
    for (size_t i = 0; i < strides_.size(); ++i) {
      const int dkernel = dilations_[i] * (ksize_[i] - 1) + 1;
      int output_size = (dims_[i + 2] - 1) * strides_[i] - paddings_[i * 2] -
                        paddings_[i * 2 + 1] + dkernel;
      output_shape.push_back(output_size);
    }

    if (!output_padding_.empty()) {
      for (size_t i = 0; i < output_padding_.size(); ++i) {
        output_shape[i + 2] += output_padding_[i];
      }
    }

    if (!output_size_.empty()) {
      for (size_t i = 0; i < output_size_.size(); ++i) {
        output_shape[i + 2] = output_size_[i];
      }
    }
    auto output = scope->NewTensor(output_);
    output->Resize(output_shape);

    const Tensor* input = scope->FindTensor(input_);
    const Tensor* filter = scope->FindTensor(filter_);
    const Tensor* bias = scope->FindTensor(bias_);
    auto input_dims = input->dims();
    auto filter_dims = filter->dims();
    auto output_dims = output->dims();
    auto input_data = input->data<float>();
    auto filter_data = filter->data<float>();
    auto output_data = output->mutable_data<float>();

    bool flag_bias = bias != nullptr;
    const float* bias_data = flag_bias ? bias->data<float>() : nullptr;
    deconv_basic<float, float>(input_data,
                               output_data,
                               input_dims[0],
                               output_dims[1],
                               output_dims[2],
                               output_dims[3],
                               input_dims[1],
                               input_dims[2],
                               input_dims[3],
                               filter_data,
                               bias_data,
                               groups_,
                               filter_dims[3],
                               filter_dims[2],
                               strides_[1],
                               strides_[0],
                               dilations_[1],
                               dilations_[0],
                               paddings_[2],
                               paddings_[3],
                               paddings_[0],
                               paddings_[1],
                               flag_bias,
                               fuse_relu_);
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType(op_type_);
    op_desc->SetInput("Input", {input_});
    op_desc->SetInput("Filter", {filter_});
    if (!bias_.empty()) {
      op_desc->SetInput("Bias", {bias_});
    }
    op_desc->SetOutput("Output", {output_});
    op_desc->SetAttr("strides", strides_);
    op_desc->SetAttr("paddings", paddings_);
    op_desc->SetAttr("groups", groups_);
    op_desc->SetAttr("dilations", dilations_);
    if (!padding_algorithm_.empty()) {
      op_desc->SetAttr("padding_algorithm", padding_algorithm_);
    }
    if (!output_padding_.empty()) {
      op_desc->SetAttr("output_padding", output_padding_);
    }
    if (!output_size_.empty()) {
      op_desc->SetAttr("output_size", output_size_);
    }
    if (fuse_relu_) {
      op_desc->SetAttr("with_act", true);
      op_desc->SetAttr("act_type", std::string("relu"));
    }
  }

  void PrepareData() override {
    std::vector<float> din(dims_.production());
    fill_data_rand(din.data(), -1.f, 1.f, dims_.production());
    SetCommonTensor(input_, dims_, din.data());

    DDim filter_dims(
        std::vector<int64_t>{dims_[1], filter_channels_, ksize_[0], ksize_[1]});
    std::vector<float> dfilter(filter_dims.production());
    fill_data_rand(dfilter.data(), -1.f, 1.f, filter_dims.production());
    SetCommonTensor(filter_, filter_dims, dfilter.data(), {}, true);

    if (!bias_.empty()) {
      DDim bias_dims(std::vector<int64_t>{filter_channels_ * groups_});
      std::vector<float> dbias(bias_dims.production());
      fill_data_rand(din.data(), -1.f, 1.f, bias_dims.production());
      SetCommonTensor(bias_, bias_dims, dbias.data(), {}, true);
    }
  }
};

void TestConvTransposeKsize(Place place, float abs_error = 2e-5) {
  for (auto dims : std::vector<std::vector<int64_t>>{{5, 6, 11, 12}}) {
    for (auto filter_channels : {1, 3}) {
      for (auto ksize :
           std::vector<std::vector<int>>{{1, 1}, {2, 2}, {3, 3}, {2, 3}}) {
        std::unique_ptr<arena::TestCase> tester(new ConvTransposeComputeTester(
            place, "def", DDim(dims), filter_channels, ksize));
        arena::Arena arena(std::move(tester), place, abs_error);
        arena.TestPrecision();
      }
    }
  }
}

void TestConvTransposeStrides(Place place, float abs_error = 2e-5) {
  for (auto dims : std::vector<std::vector<int64_t>>{{5, 6, 11, 12}}) {
    for (auto strides : std::vector<std::vector<int>>{{2, 2}, {3, 3}, {1, 2}}) {
      std::unique_ptr<arena::TestCase> tester(new ConvTransposeComputeTester(
          place, "def", DDim(dims), 3, {3, 3}, strides));
      arena::Arena arena(std::move(tester), place, abs_error);
      arena.TestPrecision();
    }
  }
}

void TestConvTransposePaddings(Place place, float abs_error = 2e-5) {
  for (auto dims : std::vector<std::vector<int64_t>>{{5, 6, 11, 12}}) {
    for (auto paddings : std::vector<std::vector<int>>{
             {1, 1}, {2, 2}, {0, 1}, {1, 0, 0, 1}, {1, 2, 0, 1}}) {
      std::unique_ptr<arena::TestCase> tester(new ConvTransposeComputeTester(
          place, "def", DDim(dims), 3, {3, 3}, {1, 1}, paddings));
      arena::Arena arena(std::move(tester), place, abs_error);
      arena.TestPrecision();
    }
  }
}

void TestConvTransposeGroups(Place place, float abs_error = 2e-5) {
  for (auto dims : std::vector<std::vector<int64_t>>{{5, 6, 11, 12}}) {
    for (auto groups : {2, 3, 6}) {
      std::unique_ptr<arena::TestCase> tester(new ConvTransposeComputeTester(
          place, "def", DDim(dims), 12, {3, 3}, {1, 1}, {0, 0}, groups));
      arena::Arena arena(std::move(tester), place, abs_error);
      arena.TestPrecision();
    }
  }
}

void TestConvTransposeDilations(Place place, float abs_error = 2e-5) {
  for (auto dims : std::vector<std::vector<int64_t>>{{5, 6, 11, 12}}) {
    for (auto dilations : std::vector<std::vector<int>>{{2, 2}, {1, 2}}) {
      std::unique_ptr<arena::TestCase> tester(new ConvTransposeComputeTester(
          place, "def", DDim(dims), 3, {3, 3}, {1, 1}, {0, 0}, 1, dilations));
      arena::Arena arena(std::move(tester), place, abs_error);
      arena.TestPrecision();
    }
  }
}

void TestConvTransposePaddingAlgorithm(Place place, float abs_error = 2e-5) {
  for (auto dims : std::vector<std::vector<int64_t>>{{5, 6, 11, 12}}) {
    for (auto padding_algorithm : std::vector<std::string>{"SAME", "VALID"}) {
      std::unique_ptr<arena::TestCase> tester(
          new ConvTransposeComputeTester(place,
                                         "def",
                                         DDim(dims),
                                         3,
                                         {3, 3},
                                         {2, 2},
                                         {0, 0},
                                         1,
                                         {1, 1},
                                         padding_algorithm));
      arena::Arena arena(std::move(tester), place, abs_error);
      arena.TestPrecision();
    }
  }
}

void TestConvTransposeOutputSize(Place place, float abs_error = 2e-5) {
  for (auto dims : std::vector<std::vector<int64_t>>{{5, 6, 12, 12}}) {
    for (auto output_size : std::vector<std::vector<int>>{{25, 26}, {26, 26}}) {
      std::unique_ptr<arena::TestCase> tester(
          new ConvTransposeComputeTester(place,
                                         "def",
                                         DDim(dims),
                                         3,
                                         {3, 3},
                                         {2, 2},
                                         {0, 0},
                                         1,
                                         {1, 1},
                                         "",
                                         output_size));
      arena::Arena arena(std::move(tester), place, abs_error);
      arena.TestPrecision();
    }
  }
}

void TestConvTransposeOutputPadding(Place place, float abs_error = 2e-5) {
  for (auto dims : std::vector<std::vector<int64_t>>{{5, 6, 12, 12}}) {
    for (auto output_padding : std::vector<std::vector<int>>{{0, 0}, {1, 1}}) {
      std::unique_ptr<arena::TestCase> tester(
          new ConvTransposeComputeTester(place,
                                         "def",
                                         DDim(dims),
                                         3,
                                         {3, 3},
                                         {2, 2},
                                         {0, 0},
                                         1,
                                         {1, 1},
                                         "",
                                         {},
                                         output_padding));
      arena::Arena arena(std::move(tester), place, abs_error);
      arena.TestPrecision();
    }
  }
}

void TestConvTransposeBiasRelu(Place place, float abs_error = 2e-5) {
  for (auto dims : std::vector<std::vector<int64_t>>{{5, 6, 11, 12}}) {
    for (auto bias : std::vector<std::string>{"", "bias"}) {
      for (bool fuse_relu : {false, true}) {
        if (bias.empty() && fuse_relu) continue;
        std::unique_ptr<arena::TestCase> tester(
            new ConvTransposeComputeTester(place,
                                           "def",
                                           DDim(dims),
                                           3,
                                           {3, 3},
                                           {1, 1},
                                           {0, 0},
                                           1,
                                           {1, 1},
                                           "",
                                           {},
                                           {},
                                           bias,
                                           fuse_relu));
        arena::Arena arena(std::move(tester), place, abs_error);
        arena.TestPrecision();
      }
    }
  }
}

// X86 not support fuse_relu yet
void TestConvDepthWiseS1(Place place, float abs_error = 2e-5) {
  for (auto dims : std::vector<std::vector<int64_t>>{{5, 15, 19, 19},
                                                     {5, 15, 21, 19},
                                                     {5, 15, 27, 27},
                                                     {5, 15, 19, 29}}) {
    std::unique_ptr<arena::TestCase> tester(
        new ConvTransposeComputeTester(place,
                                       "def",
                                       DDim(dims),
                                       1,
                                       {3, 3},
                                       {1, 1},
                                       {0, 0},
                                       dims[1],
                                       {1, 1},
                                       "",
                                       {},
                                       {}));
    arena::Arena arena(std::move(tester), place, abs_error);
    arena.TestPrecision();
  }
}

void TestConvDepthWiseS2(Place place, float abs_error = 2e-5) {
  for (auto dims : std::vector<std::vector<int64_t>>{{5, 15, 10, 10},
                                                     {5, 15, 14, 14},
                                                     {5, 15, 11, 14},
                                                     {5, 15, 15, 14}}) {
    std::unique_ptr<arena::TestCase> tester(
        new ConvTransposeComputeTester(place,
                                       "def",
                                       DDim(dims),
                                       1,
                                       {3, 3},
                                       {2, 2},
                                       {0, 0},
                                       dims[1],
                                       {1, 1},
                                       "",
                                       {},
                                       {}));
    arena::Arena arena(std::move(tester), place, abs_error);
    arena.TestPrecision();
  }
}

TEST(Conv_transpose, precision) {
  float abs_error = 2e-5;
  Place place;
#if defined(LITE_WITH_NNADAPTER)
  place = TARGET(kNNAdapter);
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  abs_error = 1e-2;
  TestConvTransposeKsize(place, abs_error);
  TestConvTransposeStrides(place, abs_error);
  TestConvTransposePaddings(place, abs_error);
  // TestConvTransposeGroups(place, abs_error);
  TestConvTransposeDilations(place, abs_error);
  TestConvTransposePaddingAlgorithm(place, abs_error);
  TestConvTransposeOutputSize(place, abs_error);
  TestConvTransposeOutputPadding(place, abs_error);
  TestConvTransposeBiasRelu(place, abs_error);
  return;
#elif defined(NNADAPTER_WITH_CAMBRICON_MLU)
  abs_error = 1e-2;
  TestConvTransposeKsize(place, abs_error);
  TestConvTransposeStrides(place, abs_error);
  TestConvTransposePaddings(place, abs_error);
  // TestConvTransposeGroups(place, abs_error);
  TestConvTransposeDilations(place, abs_error);
  // TestConvTransposePaddingAlgorithm(place, abs_error);
  // TestConvTransposeOutputSize(place, abs_error);
  // TestConvTransposeOutputPadding(place, abs_error);
  TestConvTransposeBiasRelu(place, abs_error);
  return;
#elif defined(NNADAPTER_WITH_NVIDIA_TENSORRT)
  abs_error = 1e-2;
  TestConvTransposeKsize(place, abs_error);
  TestConvTransposeStrides(place, abs_error);
  TestConvTransposePaddings(place, abs_error);
  TestConvTransposeDilations(place, abs_error);
  TestConvTransposePaddingAlgorithm(place, abs_error);
  return;
#elif defined(NNADAPTER_WITH_HUAWEI_KIRIN_NPU)
  abs_error = 1e-2;
  // TODO(liusiyuan): support later
  return;
  TestConvTransposeKsize(place, abs_error);
  TestConvTransposeStrides(place, abs_error);
  TestConvTransposePaddings(place, abs_error);
  // TestConvTransposeGroups(place, abs_error);
  TestConvTransposeDilations(place, abs_error);
  // TestConvTransposePaddingAlgorithm(place, abs_error);
  TestConvTransposeOutputSize(place, abs_error);
  TestConvTransposeOutputPadding(place, abs_error);
  TestConvTransposeBiasRelu(place, abs_error);
#elif defined(NNADAPTER_WITH_INTEL_OPENVINO)
  abs_error = 1e-5;
#elif defined(NNADAPTER_WITH_QUALCOMM_QNN)
  abs_error = 1e-2;
  TestConvTransposeKsize(place, abs_error);
  TestConvTransposeStrides(place, abs_error);
  TestConvTransposePaddings(place, abs_error);
  TestConvTransposeGroups(place, abs_error);
  // TestConvTransposeDilations(place, abs_error);
  TestConvTransposePaddingAlgorithm(place, abs_error);
  TestConvTransposeOutputSize(place, abs_error);
  TestConvTransposeOutputPadding(place, abs_error);
  TestConvTransposeBiasRelu(place, abs_error);
  return;
#else
  return;
#endif
#elif defined(LITE_WITH_NPU)
  place = TARGET(kNPU);
  abs_error = 5e-2;  // Using fp16 in NPU
#elif defined(LITE_WITH_XPU)
  place = TARGET(kXPU);
  abs_error = 5e-4;
  TestConvTransposeKsize(place, abs_error);
  TestConvTransposeStrides(place, abs_error);
  TestConvTransposePaddings(place, abs_error);
  TestConvTransposeGroups(place, abs_error);
  TestConvTransposeOutputPadding(place, abs_error);
  return;
#elif defined(LITE_WITH_ARM)
  place = TARGET(kARM);
  TestConvTransposeOutputPadding(place, abs_error);
  return;
#elif defined(LITE_WITH_X86)
  place = TARGET(kX86);
  TestConvTransposeKsize(place, abs_error);
  TestConvTransposeStrides(place, abs_error);
  TestConvTransposePaddings(place, abs_error);
  TestConvTransposeGroups(place, abs_error);
  TestConvTransposeDilations(place, abs_error);
  TestConvTransposePaddingAlgorithm(place, abs_error);
  TestConvTransposeOutputSize(place, abs_error);
  TestConvTransposeOutputPadding(place, abs_error);
  // TestConvTransposeBiasRelu(place, abs_error);  // not support fuse yet
  TestConvDepthWiseS1(place, abs_error);
  TestConvDepthWiseS2(place, abs_error);
  return;
#else
  return;
#endif

  TestConvTransposeKsize(place, abs_error);
  TestConvTransposeStrides(place, abs_error);
  TestConvTransposePaddings(place, abs_error);
  TestConvTransposeGroups(place, abs_error);
  TestConvTransposeDilations(place, abs_error);
  TestConvTransposePaddingAlgorithm(place, abs_error);
  TestConvTransposeOutputSize(place, abs_error);
  TestConvTransposeOutputPadding(place, abs_error);
  TestConvTransposeBiasRelu(place, abs_error);
}

}  // namespace lite
}  // namespace paddle
