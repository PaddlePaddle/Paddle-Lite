// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/tests/math/deformable_conv_compute_test.h"
#include <gtest/gtest.h>
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/core/test/arena/framework.h"
#include "lite/tests/utils/fill_data.h"
#include "lite/tests/utils/naive_math_impl.h"

namespace paddle {
namespace lite {

class DeformableConvComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string op_type_ = "deformable_conv";
  std::string input_ = "input";
  std::string offset_ = "offset";
  std::string mask_ = "mask";
  std::string filter_ = "filter";
  std::string output_ = "output";
  std::string bias_ = "bias";

  DDim input_dims_;

  int out_channels_ = 1;
  int im2col_step_ = 1;
  int deformable_groups_ = 1;
  std::vector<int> ksize_{3, 3};
  std::vector<int> strides_{1, 1};
  std::vector<int> paddings_{0, 0};
  int groups_ = 1;
  std::vector<int> dilations_{1, 1};
  std::string padding_algorithm_ = "";
  bool with_bias_ = false;
  bool with_act_ = false;
  std::string act_type_ = "relu";
  bool flag_modulated_ = true;

 public:
  DeformableConvComputeTester(const Place& place,
                              const std::string& alias,
                              DDim input_dims,
                              int out_channels = 1,
                              std::vector<int> ksize = {3, 3},
                              std::vector<int> strides = {1, 1},
                              std::vector<int> paddings = {0, 0},
                              int groups = 1,
                              std::vector<int> dilations = {1, 1},
                              std::string padding_algorithm = "",
                              bool with_bias = false,
                              bool with_act = false,
                              std::string act_type = "relu",
                              bool flag_modulated = true)
      : TestCase(place, alias),
        input_dims_(input_dims),
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
        flag_modulated_(flag_modulated) {}

  void RunBaseline(Scope* scope) override {
    if (paddings_.size() == 2L) {
      paddings_.insert(paddings_.begin(), paddings_[0]);
      paddings_.insert(paddings_.begin() + 2, paddings_[2]);
    }

    if (padding_algorithm_ == "SAME") {
      for (size_t i = 0; i < strides_.size(); ++i) {
        int out_size = (input_dims_[i + 2] + strides_[i] - 1) / strides_[i];
        int pad_sum = std::max(
            (out_size - 1) * strides_[i] + ksize_[i] - input_dims_[i + 2],
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

    const Tensor* input = scope->FindTensor(input_);
    const Tensor* mask = scope->FindTensor(mask_);
    const Tensor* offset = scope->FindTensor(offset_);
    const Tensor* filter = scope->FindTensor(filter_);
    const Tensor* bias = nullptr;
    if (with_bias_) {
      bias = scope->FindTensor(bias_);
    }

    auto input_dims = input->dims();
    auto filter_dims = filter->dims();

    std::vector<int64_t> output_shape{input_dims[0], filter_dims[0]};
    for (size_t i = 0; i < strides_.size(); ++i) {
      const int dkernel = dilations_[i] * (filter_dims[i + 2] - 1) + 1;
      int output_size = (input_dims[i + 2] +
                         (paddings_[i * 2] + paddings_[i * 2 + 1]) - dkernel) /
                            strides_[i] +
                        1;
      output_shape.push_back(output_size);
    }

    auto output = scope->NewTensor(output_);
    output->Resize(DDim(output_shape));
    auto output_dims = output->dims();

    bool fuse_relu = false;
    if (with_act_) {
      if (act_type_ == "relu") {
        fuse_relu = true;
      } else {
        LOG(FATAL) << "unsupported";
      }
    }

    CHECK_EQ(paddings_.size(), 4L)
        << "[HUAWEI_ASCEND_NPU] Paddings size should be "
           "the same or twice as the input size.";
    deformable_conv_compute_basic(input,
                                  offset,
                                  mask,
                                  output,
                                  *filter,
                                  bias,
                                  groups_,
                                  deformable_groups_,
                                  im2col_step_,
                                  strides_,
                                  paddings_,
                                  dilations_,
                                  fuse_relu);
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType(op_type_);
    op_desc->SetInput("Input", {input_});
    op_desc->SetInput("Mask", {mask_});
    op_desc->SetInput("Filter", {filter_});
    op_desc->SetInput("Offset", {offset_});
    if (with_bias_) {
      op_desc->SetInput("Bias", {bias_});
    }
    op_desc->SetOutput("Output", {output_});
    op_desc->SetAttr("strides", strides_);
    op_desc->SetAttr("paddings", paddings_);
    op_desc->SetAttr("groups", groups_);
    op_desc->SetAttr("dilations", dilations_);
    op_desc->SetAttr("deformable_groups", deformable_groups_);
    op_desc->SetAttr("im2col_step", im2col_step_);
    op_desc->SetAttr("flag_modulated", flag_modulated_);

    if (!padding_algorithm_.empty()) {
      op_desc->SetAttr("padding_algorithm", padding_algorithm_);
    }

    if (with_act_) {
      op_desc->SetAttr("with_act", with_act_);
      op_desc->SetAttr("act_type", act_type_);
    }
  }

  void PrepareData() override {
    // input
    std::vector<float> din(input_dims_.production());
    fill_data_rand(din.data(), -1.f, 1.f, input_dims_.production());
    SetCommonTensor(input_, input_dims_, din.data());

    // filter
    DDim filter_dims(std::vector<int64_t>{
        out_channels_, input_dims_[1] / groups_, ksize_[0], ksize_[1]});
    std::vector<float> dfilter(filter_dims.production());
    fill_data_rand(dfilter.data(), -1.f, 1.f, filter_dims.production());
    SetCommonTensor(filter_, filter_dims, dfilter.data(), {}, true);

    // offsets
    int64_t h_out = ((input_dims_[2] + 2 * paddings_[0] -
                      (dilations_[0] * (ksize_[0] - 1) + 1)) /
                     strides_[0]) +
                    1;

    int64_t w_out = ((input_dims_[3] + 2 * paddings_[1] -
                      (dilations_[1] * (ksize_[1] - 1) + 1)) /
                     strides_[1]) +
                    1;

    DDim offset_dims(std::vector<int64_t>{
        input_dims_[0], 2 * ksize_[0] * ksize_[1], h_out, w_out});
    std::vector<float> doffset(offset_dims.production());
    fill_data_rand(doffset.data(), -1.f, 1.f, offset_dims.production());
    SetCommonTensor(offset_, offset_dims, doffset.data(), {}, true);

    // mask
    DDim mask_dims(std::vector<int64_t>{
        input_dims_[0], ksize_[0] * ksize_[1], h_out, w_out});
    std::vector<float> dmask(mask_dims.production());
    fill_data_rand(dmask.data(), -1.f, 1.f, mask_dims.production());
    SetCommonTensor(mask_, mask_dims, dmask.data(), {}, true);

    // bias
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
       std::vector<std::vector<int64_t>>{{1, 3, 12, 12}, {5, 4, 17, 18}}) {
    for (auto out_channels : {6}) {
      for (auto ksize : {1, 3, 7}) {
        std::unique_ptr<arena::TestCase> tester(new DeformableConvComputeTester(
            place, "def", DDim(dims), out_channels, {ksize, ksize}));
        arena::Arena arena(std::move(tester), place, abs_error);
        arena.TestPrecision();
      }
    }
  }
}

void TestConvDilations(Place place, float abs_error = 2e-5) {
  for (auto dims :
       std::vector<std::vector<int64_t>>{{1, 2, 5, 6}, {5, 6, 9, 10}}) {
    for (auto out_channels : {3}) {
      for (auto dilations : std::vector<std::vector<int>>{{2, 2}, {1, 2}}) {
#if defined(NNADAPTER_WITH_INTEL_OPENVINO)
        if (dilations[0] != dilations[1]) {
          continue;
        }
#endif
        std::unique_ptr<arena::TestCase> tester(
            new DeformableConvComputeTester(place,
                                            "def",
                                            DDim(dims),
                                            out_channels,
                                            {3, 3},
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
    for (auto out_channels : {3}) {
      for (auto strides :
           std::vector<std::vector<int>>{{2, 2}, {3, 3}, {1, 2}}) {
        std::unique_ptr<arena::TestCase> tester(new DeformableConvComputeTester(
            place, "def", DDim(dims), out_channels, {3, 3}, strides));
        arena::Arena arena(std::move(tester), place, abs_error);
        arena.TestPrecision();
      }
    }
  }
}

void TestConvPaddings(Place place, float abs_error = 2e-5) {
  for (auto dims :
       std::vector<std::vector<int64_t>>{{1, 2, 3, 4}, {5, 6, 7, 8}}) {
    for (auto out_channels : {3}) {
      for (auto paddings :
           std::vector<std::vector<int>>{{1, 1}, {0, 0}, {2, 2}}) {
        std::unique_ptr<arena::TestCase> tester(new DeformableConvComputeTester(
            place, "def", DDim(dims), out_channels, {3, 3}, {1, 1}, paddings));
        arena::Arena arena(std::move(tester), place, abs_error);
        arena.TestPrecision();
      }
    }
  }
}

void TestConvBias(Place place, float abs_error = 2e-5) {
  for (auto dims : std::vector<std::vector<int64_t>>{{1, 2, 3, 4}}) {
    for (auto out_channels : {1}) {
      std::unique_ptr<arena::TestCase> tester(
          new DeformableConvComputeTester(place,
                                          "def",
                                          DDim(dims),
                                          out_channels,
                                          {3, 3},
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
    for (auto out_channels : {1}) {
      std::unique_ptr<arena::TestCase> tester(
          new DeformableConvComputeTester(place,
                                          "def",
                                          DDim(dims),
                                          out_channels,
                                          {3, 3},
                                          {1, 1},
                                          {0, 0},
                                          1,
                                          {1, 1},
                                          "",
                                          false,
                                          true,
                                          "relu"));
      arena::Arena arena(std::move(tester), place, abs_error);
      arena.TestPrecision();
    }
  }
}

TEST(Deformable_conv, precision) {
  std::cout << "start..." << std::endl;
  float abs_error = 2e-5;
  Place place;
#if defined(LITE_WITH_NNADAPTER)
  place = TARGET(kNNAdapter);
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  abs_error = 3e-2;
#elif defined(NNADAPTER_WITH_CAMBRICON_MLU)
  abs_error = 1e-3;
#elif defined(NNADAPTER_WITH_INTEL_OPENVINO)
  abs_error = 3e-2;
#else
  return;
#endif
#elif defined(LITE_WITH_ARM)
  place = TARGET(kARM);
  abs_error = 1e-5;
#elif defined(LITE_WITH_X86)
  place = TARGET(kHost);
  abs_error = 1e-5;
#else
  return;
#endif

  TestConvKsize(place, abs_error);
  TestConvDilations(place, abs_error);
  TestConvStrides(place, abs_error);
  TestConvPaddings(place, abs_error);
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU) || defined(LITE_WITH_ARM) || \
    defined(NNADAPTER_WITH_INTEL_OPENVINO)
  TestConvBias(place, abs_error);
  TestConvAct(place, abs_error);
#endif
}

}  // namespace lite
}  // namespace paddle
