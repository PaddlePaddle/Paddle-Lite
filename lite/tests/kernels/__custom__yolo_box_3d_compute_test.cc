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
#include <vector>
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/core/tensor.h"
#include "lite/core/test/arena/framework.h"

namespace paddle {
namespace lite {
namespace {

template <typename T>
inline T Sigmoid(T x) {
  return (T)1. / ((T)1. + std::exp(-x));
}

template <>
inline float Sigmoid(float x) {
  return 1.f / (1.f + expf(-x));
}

inline int GetEntryIndex(int batch_idx,
                         int an_idx,
                         int hw_idx,
                         int an_num,
                         int an_stride,
                         int stride,
                         int entry) {
  return (batch_idx * an_num + an_idx) * an_stride + entry * stride + hw_idx;
}

template <typename T>
inline void GetYoloBox(T* box,
                       const T* x_data,
                       const int* anchors,
                       int w_idx,
                       int h_idx,
                       int an_idx,
                       int h,
                       int w,
                       int downsample_ratio,
                       int index,
                       int stride,
                       int img_height,
                       int img_width,
                       T scale,
                       T bias) {
  box[0] = (w_idx + Sigmoid(x_data[index]) * scale + bias) / w;
  box[1] = (h_idx + Sigmoid(x_data[index + stride]) * scale + bias) / h;
  box[2] = std::exp(x_data[index + 2 * stride]) * anchors[2 * an_idx] /
           (downsample_ratio * w);
  box[3] = std::exp(x_data[index + 3 * stride]) * anchors[2 * an_idx + 1] /
           (downsample_ratio * h);
}

template <typename T>
inline void CalcDetectionBox(T* boxes_data,
                             T* box,
                             const int box_idx,
                             const int img_height,
                             const int img_width,
                             bool clip_bbox) {
  boxes_data[box_idx] = (box[0] - box[2] * 0.5f) * img_width;
  boxes_data[box_idx + 1] = (box[1] - box[3] * 0.5f) * img_height;
  boxes_data[box_idx + 2] = (box[0] + box[2] * 0.5f) * img_width;
  boxes_data[box_idx + 3] = (box[1] + box[3] * 0.5f) * img_height;
}

template <typename T>
std::vector<T> Softmax(std::vector<T> input) {
  T total = 0;
  auto max_value = *std::max_element(input.begin(), input.end());
  for (auto x : input) {
    total += std::exp(x - max_value);
  }
  std::vector<T> result;
  for (auto x : input) {
    result.push_back(std::exp(x - max_value) / total);
  }
  return result;
}

template <typename T>
inline void CalcLabelScore(T* scores,
                           const T* input,
                           const int score_input_idx,
                           const int score_output_idx,
                           const int class_num,
                           const T conf,
                           const int stride) {
  std::vector<T> softmax_inputs;
  for (int i = 0; i < class_num; i++) {
    softmax_inputs.push_back(input[score_input_idx + i * stride]);
  }
  auto softmax_result = Softmax(softmax_inputs);
  for (int i = 0; i < class_num; i++) {
    scores[score_output_idx + i] = conf * softmax_result[i];
  }
}
}  // namespace

template <typename T>
class YoloBox3dComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string input0_ = "X";
  std::string input1_ = "ImgSize";
  std::string output0_ = "Boxes";
  std::string output1_ = "Scores";
  std::string output2_ = "Location";
  std::string output3_ = "Dim";
  std::string output4_ = "Alpha";
  std::vector<int> anchors_;
  int class_num_ = 4;
  float conf_thresh_ = 0.0045;
  int downsample_ratio_ = 8;
  bool clip_bbox_ = false;
  float scale_x_y_ = 1.05f;
  DDim _dims0_{{1, 51, 80, 160}};
  DDim _dims1_{{1, 2}};

 public:
  YoloBox3dComputeTester(const Place& place,
                         const std::string& alias,
                         std::vector<int> anchors,
                         int class_num,
                         T conf_thresh,
                         int downsample_ratio)
      : TestCase(place, alias),
        anchors_(anchors),
        class_num_(class_num),
        conf_thresh_(conf_thresh),
        downsample_ratio_(downsample_ratio) {
    int anchor_num = anchors_.size() / 2;
    _dims0_[1] = anchor_num * (5 + class_num + 8);
  }

  void RunBaseline(Scope* scope) override {
    const lite::Tensor* X = scope->FindTensor(input0_);
    const lite::Tensor* ImgSize = scope->FindTensor(input1_);
    auto* in = X;
    auto* imgsize = ImgSize;
    auto anchors = anchors_;
    int class_num = class_num_;
    T conf_thresh = conf_thresh_;
    int downsample_ratio = downsample_ratio_;

    const int n = in->dims()[0];
    const int h = in->dims()[2];
    const int w = in->dims()[3];
    const int an_num = anchors.size() / 2;
    int box_num = in->dims()[2] * in->dims()[3] * an_num;
    T scale_x_y = static_cast<T>(scale_x_y_);
    T bias = static_cast<T>(-0.5 * (scale_x_y_ - 1.f));

    lite::Tensor* Boxes = scope->NewTensor(output0_);
    lite::Tensor* Scores = scope->NewTensor(output1_);
    lite::Tensor* Location = scope->NewTensor(output2_);
    lite::Tensor* Dim = scope->NewTensor(output3_);
    lite::Tensor* Alpha = scope->NewTensor(output4_);
    CHECK(Boxes);
    CHECK(Scores);
    CHECK(Location);
    CHECK(Dim);
    CHECK(Alpha);
    Boxes->Resize({in->dims()[0], box_num, 4});
    Scores->Resize({in->dims()[0], box_num, class_num});
    Location->Resize({in->dims()[0], box_num, 3});
    Dim->Resize({in->dims()[0], box_num, 3});
    Alpha->Resize({in->dims()[0], box_num, 2});
    auto* boxes = Boxes;
    auto* scores = Scores;
    auto* location = Location;
    auto* dim = Dim;
    auto* alpha = Alpha;
    const int b_num = boxes->dims()[0];
    const int stride = h * w;
    const int an_stride = (class_num + 5 + 8) * stride;

    auto anchors_data = anchors.data();
    const T* x_data = in->data<T>();
    const int* img_size_data = imgsize->data<int>();
    T* boxes_data = boxes->mutable_data<T>();
    T* scores_data = scores->mutable_data<T>();
    T* location_data = location->mutable_data<T>();
    T* dim_data = dim->mutable_data<T>();
    T* alpha_data = alpha->mutable_data<T>();
    T box[4];
    for (int batch_idx = 0; batch_idx < n; batch_idx++) {
      int img_height = img_size_data[2 * batch_idx];
      int img_width = img_size_data[2 * batch_idx + 1];

      for (int an_idx = 0; an_idx < an_num; an_idx++) {
        for (int h_idx = 0; h_idx < h; h_idx++) {
          for (int w_idx = 0; w_idx < w; w_idx++) {
            // Calc boxes output
            int box_input_idx = GetEntryIndex(batch_idx,
                                              an_idx,
                                              h_idx * w + w_idx,
                                              an_num,
                                              an_stride,
                                              stride,
                                              0);
            GetYoloBox(box,
                       x_data,
                       anchors_data,
                       w_idx,
                       h_idx,
                       an_idx,
                       h,
                       w,
                       downsample_ratio,
                       box_input_idx,
                       stride,
                       img_height,
                       img_width,
                       scale_x_y,
                       bias);
            int box_output_idx =
                (batch_idx * b_num + an_idx * stride + h_idx * w + w_idx) * 4;
            CalcDetectionBox(boxes_data,
                             box,
                             box_output_idx,
                             img_height,
                             img_width,
                             clip_bbox_);

            // Calc score output
            int obj_idx = GetEntryIndex(batch_idx,
                                        an_idx,
                                        h_idx * w + w_idx,
                                        an_num,
                                        an_stride,
                                        stride,
                                        4);
            T conf = Sigmoid(x_data[obj_idx]);
            int score_input_idx = GetEntryIndex(batch_idx,
                                                an_idx,
                                                h_idx * w + w_idx,
                                                an_num,
                                                an_stride,
                                                stride,
                                                13);
            int score_output_idx =
                (batch_idx * b_num + an_idx * stride + h_idx * w + w_idx) *
                class_num;
            CalcLabelScore(scores_data,
                           x_data,
                           score_input_idx,
                           score_output_idx,
                           class_num,
                           conf,
                           stride);

            // Calc location output
            int location_input_idx = GetEntryIndex(batch_idx,
                                                   an_idx,
                                                   h_idx * w + w_idx,
                                                   an_num,
                                                   an_stride,
                                                   stride,
                                                   5);
            int location_output_idx =
                (batch_idx * b_num + an_idx * stride + h_idx * w + w_idx) * 3;
            location_data[location_output_idx] =
                (w_idx + Sigmoid(x_data[location_input_idx]) * scale_x_y +
                 bias) *
                img_width / w;
            location_data[location_output_idx + 1] =
                (h_idx +
                 Sigmoid(x_data[location_input_idx + stride]) * scale_x_y +
                 bias) *
                img_height / h;
            location_data[location_output_idx + 2] =
                x_data[location_input_idx + 2 * stride];

            // Calc dim output
            int dim_input_idx = GetEntryIndex(batch_idx,
                                              an_idx,
                                              h_idx * w + w_idx,
                                              an_num,
                                              an_stride,
                                              stride,
                                              8);
            int dim_output_idx =
                (batch_idx * b_num + an_idx * stride + h_idx * w + w_idx) * 3;
            dim_data[dim_output_idx] = x_data[dim_input_idx];
            dim_data[dim_output_idx + 1] = x_data[dim_input_idx + stride];
            dim_data[dim_output_idx + 2] = x_data[dim_input_idx + 2 * stride];

            // Calc alpha output
            int alpha_input_idx = GetEntryIndex(batch_idx,
                                                an_idx,
                                                h_idx * w + w_idx,
                                                an_num,
                                                an_stride,
                                                stride,
                                                11);
            int alpha_output_idx =
                (batch_idx * b_num + an_idx * stride + h_idx * w + w_idx) * 2;
            alpha_data[alpha_output_idx] = x_data[alpha_input_idx];
            alpha_data[alpha_output_idx + 1] = x_data[alpha_input_idx + stride];
          }
        }
      }
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("__custom__yolo_box_3d");
    op_desc->SetInput("X", {input0_});
    op_desc->SetInput("ImgSize", {input1_});
    op_desc->SetOutput("Boxes", {output0_});
    op_desc->SetOutput("Scores", {output1_});
    op_desc->SetOutput("Location", {output2_});
    op_desc->SetOutput("Dim", {output3_});
    op_desc->SetOutput("Alpha", {output4_});
    op_desc->SetAttr("anchors", anchors_);
    op_desc->SetAttr("class_num", class_num_);
    op_desc->SetAttr("conf_thresh", conf_thresh_);
    op_desc->SetAttr("downsample_ratio", downsample_ratio_);
    op_desc->SetAttr("clip_bbox", clip_bbox_);
    op_desc->SetAttr("scale_x_y", scale_x_y_);
  }

  void PrepareData() override {
    std::vector<T> data0(_dims0_.production());
    for (int i = 0; i < _dims0_.production(); i++) {
      data0[i] = i % 10;
    }
    std::vector<int> data1(_dims1_.production());
    data1[0] = 640;
    data1[1] = 1280;
    SetCommonTensor(input0_, _dims0_, data0.data());
    SetCommonTensor(input1_, _dims1_, data1.data());
  }
};

template <typename T>
void TestYoloBox3d(Place place, float abs_error) {
  for (int class_num : {4}) {
    for (float conf_thresh : {0.f}) {
      for (int downsample_ratio : {8}) {
        std::vector<int> anchor{10, 15, 24, 36, 72, 42};
        std::unique_ptr<arena::TestCase> tester(new YoloBox3dComputeTester<T>(
            place, "def", anchor, class_num, conf_thresh, downsample_ratio));
        arena::Arena arena(std::move(tester), place, abs_error);
        arena.TestPrecision();
      }
    }
  }
}

TEST(YoloBox3d, precision) {
  float abs_error = 1e-3;
  Place place;
#if defined(LITE_WITH_NNADAPTER)
  place = TARGET(kNNAdapter);
#if defined(NNADAPTER_WITH_QUALCOMM_QNN)
  abs_error = 1e-3;
#else
  return;
#endif
#elif defined(LITE_WITH_ARM) || defined(LITE_WITH_X86)
  place = TARGET(kHost);
#else
  return;
#endif

  TestYoloBox3d<float>(place, abs_error);
}

}  // namespace lite
}  // namespace paddle
