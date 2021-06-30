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

#include <gtest/gtest.h>
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/core/arena/framework.h"
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
  std::string bias_ = "bias";
  std::string filter_ = "filter";
  std::string output_ = "output";
  std::string deformable_groups_ = "deformable_groups";
  std::string im2col_step_ = "im2col_step_";

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
  bool with_act_ = false;
  std::string act_type_;

 public:
  DeformableConvComputeTester(const Place& place,
                              const std::string& alias,
                              DDim dims,
                              int filter_channel = 1,
                              std::vector<int> ksize = {3, 3},
                              std::vector<int> strides = {1, 1},
                              std::vector<int> paddings = {0, 0},
                              int groups = 1,
                              std::vector<int> dilations = {1, 1},
                              std::string padding_algorithm = "",
                              std::string bias = "",
                              bool with_act = false,
                              std::string act_type = "",
                              flag_modulated = true)
      : TestCase(place, alias),
        dims_(dims),
        filter_channel_(filter_channels),
        ksize_(ksize),
        strides_(strides),
        paddings_(paddings),
        groups_(groups),
        dilations_(dilations),
        padding_algorithm_(padding_algorithm),
        bias_(bias),
        with_act_(with_act) act_type_(act_type) {}

  void RunBaseline(Scope* scope) override {
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

    if (with_act_) {
      if (act_type_ == "relu") {
        bool fuse_fule = true;
      } else {
        LOG(FATAL) << "unsupported";
      }
    }

    CHECK_EQ(paddings.size(), 4L)
        << "[HUAWEI_ASCEND_NPU] Paddings size should be "
           "the same or twice as the input size.";
    deformable_conv_basic<float, float>(input_data,
                                        offset_data,
                                        mask_data,
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
                                        flag_bias,
                                        fuse_fule,
                                        flag_modulated);
  }

  void PrepareOpDesc(cpp : OpDesc* op_desc) {
    op_desc->SetType(op_type_);
    op_desc->SetInput("Input", {input_});
    op_desc->SetInput("Mask", {mask_});
    op_desc->SetInput("Filter", {filter_});
    op_desc->SetInput("Offset", {offset_});
    if (!bias_.empty) {
      op_desc->SetInput("Bias", {bias_})
    }
    op_desc->SetOutput("Output", {output_});
    op_desc->SetAttr("strides", strides_);
    op_desc->SetAttr("paddings", paddings_);
    op_desc->SetAttr("groups", groups_);
    op_desc->SetAttr("dilations", dilations_);
    op_desc->SetAttr("deformable_groups", deformable_groups_);
    op_desc->SetAttr("im2col_step", im2col_step_);
    if (with_act_) {
      op_desc->SetAttr("with_act", with_act_);
      op_desc->SetAttr("act_type", act_type_);
      if (act_type_ == "leaky_relu") {
        op_desc->SetAttr("leaky_relu_alpha", leaky_relu_alpha_);
      }
    }
  }

  void PrepareData() override {
    // input
    std::vector<float> din(dims_.production());
    fill_data_rand(din.data(), -1.f, 1.f, dims_.production());
    SetCommonTensor(input_, dims_, din.data());

    // filter
    DDim filter_dims(std::vector<int64_t>{
        out_channels_, dims_[1] / groups_, ksize_, ksize_});
    std::vector<float> dfilter(filter_dims.production());
    fill_data_rand(dfilter.data(), -1.f, 1.f, filter_dims.production());
    SetCommonTensor(filter_, filter_dims, dfilter.data(), {}, true);

    // offsets
    h_out =
        ((dims_[2] + 2 * paddings_[0] - (dilations_[0] * (ksize_ - 1) + 1)) /
         strides_[0]) +
        1;
    w_out =
        ((dims_[3] + 2 * paddings_[1] - (dilations_[1] * (ksize_ - 1) + 1)) /
         strides_[1]) +
        1;

    DDim offset_dims(std::vector<int64_t>){
      dims[0], 2*ksize_*ksize_, h_out, w_out});
    std::vector<float> doffset(offset_dims.production());
    fill_data_rand(doffset.data(), -1.f, 1.f, offset_dims.production());
    SetCommonTensor(offset_, offset_dims, doffset.data(), {}, true);

    // mask
    DDim mask_dims(std::vector<int64_t>){dims[1], ksize_*ksize_, h_out, w_out});
    std::vector<float> dmask(mask_dims.production());
    fill_data_rand(dmask.data(), -1.f, 1.f, mask_dims.production());
    SetCommonTensor(mask_, mask_dims, dmask.data(), {}, true);

    if (with_bias_) {
      DDim bias_dims(std::vector<int64_t>{out_channels_});
      std::vector<float> dbias(bias_dims.production());
      fill_data_rand(din.data(), -1.f, 1.f, bias_dims.production());
      SetCommonTensor(bias_, bias_dims, dbias.data(), {}, true);
    }
  }
};

void TestConvKsize(Place place, float abs_error = 2e-5) {
  for (auto dims :
       std::vector<std::vector<int64_t>>{{1, 3, 12, 12}, {5, 6, 17, 18}}) {
    for (auto out_channels : {1, 3}) {
      for (auto ksize : {1, 3, 5, 7}) {
        std::unique_ptr<arena::TestCase> tester(new DeformableConvComputeTester(
            place, "def", DDim(dims), out_channels, ksize));
        arena::Arena arena(std::move(tester), place, abs_error);
        arena.TestPrecision();
      }
    }
  }
}

TEST(Deformable_conv, precision) {
  float abs_error = 2e-5;
  Place place;
#if defined(LITE_WITH_HUAWEI_ASCEND_NPU)
  place = TARGET(kHuaweiAscendNPU);
  abs_error = 1e-2;  // precision_mode default is force_fp16
#else
  return;
#endif
}

}  // namespace lite
}  // namespace paddle
