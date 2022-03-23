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
namespace {}  // namespace

template <typename T>
class YoloBoxComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string input0_ = "X";
  std::string output0_ = "Boxes";
  std::vector<int> anchors_;
  int class_num_ = 0;
  float conf_thresh_ = 0.f;
  int downsample_ratio_ = 0;
  bool clip_bbox_ = true;
  float scale_x_y_ = 1.0;

  DDim _dims0_{{1, 255, 13, 13}};

 public:
  YoloBoxComputeTester(const Place& place,
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
    _dims0_[1] = anchor_num * (5 + class_num);
  }

  void RunBaseline(Scope* scope) override {
    const lite::Tensor* X = scope->FindTensor(input0_);
    auto* in = X;
    auto anchors = anchors_;
    int class_num = class_num_;
    T conf_thresh = conf_thresh_;
    int downsample_ratio = downsample_ratio_;
    const int an_num = anchors.size() / 2;
    anchors = anchors_;
    class_num = class_num_;
    conf_thresh = conf_thresh_;
    downsample_ratio = downsample_ratio_;

    const int n = in->dims()[0];
    const int h = in->dims()[2];
    const int w = in->dims()[3];
    const int an_stride = (class_num + 5) * h * w;
    const int stride = h * w;
    int in_size = downsample_ratio * h;
    in_size = in_size;

    lite::Tensor* Boxes = scope->NewTensor(output0_);
    CHECK(Boxes);
    Boxes->Resize(in->dims());
    auto* boxes = Boxes;

    const T* in_data = in->data<T>();
    T* boxes_data = boxes->mutable_data<T>();

    for (int i = 0; i < n; i++) {
      for (int j = 0; j < an_num; j++) {  // (5 + class_num)
        for (int k = 0; k < h; k++) {
          for (int l = 0; l < w; l++) {
            const float alpha = scale_x_y_;
            const float beta = -0.5 * (scale_x_y_ - 1);
            int index = i * an_num * an_stride + j * an_stride + k * w + l;
            boxes_data[index] = in_data[index] * alpha + beta;
            boxes_data[index + 1 * stride] =
                in_data[index + 1 * stride] * alpha + beta;
            boxes_data[index + 2 * stride] =
                pow(in_data[index + 2 * stride] * 2, 2);
            boxes_data[index + 3 * stride] =
                pow(in_data[index + 3 * stride] * 2, 2);
            boxes_data[index + 4 * stride] = in_data[index + 4 * stride];
            for (int class_i = 0; class_i < class_num; class_i++)
              boxes_data[index + (5 + class_i) * stride] =
                  in_data[index + (5 + class_i) * stride];
          }
        }
      }
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("yolo_box_head");
    op_desc->SetInput("X", {input0_});
    op_desc->SetOutput("Boxes", {output0_});
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
      data0[i] = i * 0.1;
    }
    SetCommonTensor(input0_, _dims0_, data0.data());
  }
};

template <typename T>
void TestYoloBox(Place place, float abs_error) {
  for (int class_num : {1, 4}) {
    for (float conf_thresh : {0.01, 0.2}) {
      for (int downsample_ratio : {16, 32}) {
        std::vector<int> anchor{10, 13, 16, 30, 33, 30};
        std::unique_ptr<arena::TestCase> tester(new YoloBoxComputeTester<T>(
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
#if defined(NNADAPTER_WITH_NVIDIA_TENSORRT)
  place = TARGET(kNNAdapter);
#else
  return;
#endif

  TestYoloBox<float>(place, abs_error);
}

}  // namespace lite
}  // namespace paddle
