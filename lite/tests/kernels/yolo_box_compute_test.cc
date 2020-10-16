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
#include "lite/core/arena/framework.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {

namespace {
inline float sigmoid(float x) { return 1.f / (1.f + expf(-x)); }

inline void get_yolo_box(float* box,
                         const float* x,
                         const int* anchors,
                         int i,
                         int j,
                         int an_idx,
                         int grid_size,
                         int input_size,
                         int index,
                         int stride,
                         int img_height,
                         int img_width) {
  box[0] = (i + sigmoid(x[index])) * img_width / grid_size;
  box[1] = (j + sigmoid(x[index + stride])) * img_height / grid_size;
  box[2] = std::exp(x[index + 2 * stride]) * anchors[2 * an_idx] * img_width /
           input_size;
  box[3] = std::exp(x[index + 3 * stride]) * anchors[2 * an_idx + 1] *
           img_height / input_size;
}

inline int get_entry_index(int batch,
                           int an_idx,
                           int hw_idx,
                           int an_num,
                           int an_stride,
                           int stride,
                           int entry) {
  return (batch * an_num + an_idx) * an_stride + entry * stride + hw_idx;
}

inline void calc_detection_box(float* boxes,
                               float* box,
                               const int box_idx,
                               const int img_height,
                               const int img_width) {
  boxes[box_idx] = box[0] - box[2] / 2;
  boxes[box_idx + 1] = box[1] - box[3] / 2;
  boxes[box_idx + 2] = box[0] + box[2] / 2;
  boxes[box_idx + 3] = box[1] + box[3] / 2;

  boxes[box_idx] = boxes[box_idx] > 0 ? boxes[box_idx] : static_cast<float>(0);
  boxes[box_idx + 1] =
      boxes[box_idx + 1] > 0 ? boxes[box_idx + 1] : static_cast<float>(0);
  boxes[box_idx + 2] = boxes[box_idx + 2] < img_width - 1
                           ? boxes[box_idx + 2]
                           : static_cast<float>(img_width - 1);
  boxes[box_idx + 3] = boxes[box_idx + 3] < img_height - 1
                           ? boxes[box_idx + 3]
                           : static_cast<float>(img_height - 1);
}

inline void calc_label_score(float* scores,
                             const float* input,
                             const int label_idx,
                             const int score_idx,
                             const int class_num,
                             const float conf,
                             const int stride) {
  for (int i = 0; i < class_num; i++) {
    scores[score_idx + i] = conf * sigmoid(input[label_idx + i * stride]);
  }
}
}  // namespace

class YoloBoxComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string input0_ = "X";
  std::string input1_ = "ImgSize";
  std::string output0_ = "Boxes";
  std::string output1_ = "Scores";
  std::vector<int> anchors_;
  int class_num_ = 0;
  float conf_thresh_ = 0.f;
  int downsample_ratio_ = 0;
  bool clip_bbox_ = true;
  float scale_x_y_ = 1.0;

  DDim _dims0_{{1, 255, 13, 13}};
  DDim _dims1_{{1, 2}};

 public:
  YoloBoxComputeTester(const Place& place,
                       const std::string& alias,
                       std::vector<int> anchors,
                       int class_num,
                       float conf_thresh,
                       int downsample_ratio)
      : TestCase(place, alias),
        anchors_(anchors),
        class_num_(class_num),
        conf_thresh_(conf_thresh),
        downsample_ratio_(downsample_ratio) {
    int anchor_num = anchors_.size() / 2;
    _dims0_[1] = anchor_num * (5 + class_num);
  }

  void RunBaseline(Scope* scope) override {
    const lite::Tensor* X = scope->FindTensor(input0_);
    const lite::Tensor* ImgSize = scope->FindTensor(input1_);
    auto* in = X;
    auto* imgsize = ImgSize;
    auto anchors = anchors_;
    int class_num = class_num_;
    float conf_thresh = conf_thresh_;
    int downsample_ratio = downsample_ratio_;

    const int n = in->dims()[0];
    const int h = in->dims()[2];
    const int w = in->dims()[3];
    const int an_num = anchors.size() / 2;
    int in_size = downsample_ratio * h;
    int box_num = in->dims()[2] * in->dims()[3] * an_num;

    lite::Tensor* Boxes = scope->NewTensor(output0_);
    lite::Tensor* Scores = scope->NewTensor(output1_);
    CHECK(Boxes);
    CHECK(Scores);
    Boxes->Resize({in->dims()[0], box_num, 4});
    Scores->Resize({in->dims()[0], box_num, class_num});
    auto* boxes = Boxes;
    auto* scores = Scores;
    const int b_num = boxes->dims()[0];

    const int stride = h * w;
    const int an_stride = (class_num + 5) * stride;

    auto anchors_data = anchors.data();
    const float* in_data = in->data<float>();
    const int* imgsize_data = imgsize->data<int>();
    float* boxes_data = boxes->mutable_data<float>();
    float* scores_data = scores->mutable_data<float>();

    float box[4];
    for (int i = 0; i < n; i++) {
      int img_height = imgsize_data[2 * i];
      int img_width = imgsize_data[2 * i + 1];
      for (int j = 0; j < an_num; j++) {
        for (int k = 0; k < h; k++) {
          for (int l = 0; l < w; l++) {
            int obj_idx =
                get_entry_index(i, j, k * w + l, an_num, an_stride, stride, 4);
            float conf = sigmoid(in_data[obj_idx]);
            if (conf < conf_thresh) {
              continue;
            }
            int box_idx =
                get_entry_index(i, j, k * w + l, an_num, an_stride, stride, 0);
            get_yolo_box(box,
                         in_data,
                         anchors_data,
                         l,
                         k,
                         j,
                         h,
                         in_size,
                         box_idx,
                         stride,
                         img_height,
                         img_width);
            box_idx = (i * b_num + j * stride + k * w + l) * 4;
            calc_detection_box(boxes_data, box, box_idx, img_height, img_width);

            int label_idx =
                get_entry_index(i, j, k * w + l, an_num, an_stride, stride, 5);
            int score_idx = (i * b_num + j * stride + k * w + l) * class_num;
            calc_label_score(scores_data,
                             in_data,
                             label_idx,
                             score_idx,
                             class_num,
                             conf,
                             stride);
          }
        }
      }
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("yolo_box");
    op_desc->SetInput("X", {input0_});
    op_desc->SetInput("ImgSize", {input1_});
    op_desc->SetOutput("Boxes", {output0_});
    op_desc->SetOutput("Scores", {output1_});
    op_desc->SetAttr("anchors", anchors_);
    op_desc->SetAttr("class_num", class_num_);
    op_desc->SetAttr("conf_thresh", conf_thresh_);
    op_desc->SetAttr("downsample_ratio", downsample_ratio_);
    op_desc->SetAttr("clip_bbox", clip_bbox_);
    op_desc->SetAttr("scale_x_y", scale_x_y_);
  }

  void PrepareData() override {
    std::vector<float> data0(_dims0_.production());
    for (int i = 0; i < _dims0_.production(); i++) {
      data0[i] = i * 1.1;
    }
    std::vector<int> data1(_dims1_.production());
    for (int i = 0; i < _dims1_.production(); i++) {
      data1[i] = 608;
    }
    SetCommonTensor(input0_, _dims0_, data0.data());
    SetCommonTensor(input1_, _dims1_, data1.data());
  }
};

void TestYoloBox(Place place, float abs_error) {
  for (int class_num : {1, 4}) {
    for (float conf_thresh : {0.01, 0.2}) {
      for (int downsample_ratio : {16, 32}) {
        std::vector<int> anchor{10, 13, 16, 30, 33, 30};
        std::unique_ptr<arena::TestCase> tester(new YoloBoxComputeTester(
            place, "def", anchor, class_num, conf_thresh, downsample_ratio));
        arena::Arena arena(std::move(tester), place, abs_error);
        arena.TestPrecision();
      }
    }
  }
}

TEST(YoloBox, precision) {
  float abs_error = 2e-5;
  Place place;
#if defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#elif defined(LITE_WITH_XPU) && defined(LITE_WITH_XTCL)
  place = TARGET(kXPU);
#else
  return;
#endif

  TestYoloBox(place, abs_error);
}

}  // namespace lite
}  // namespace paddle
