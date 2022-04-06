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
#include "lite/tests/utils/fill_data.h"

namespace paddle {
namespace lite {
namespace {}  // namespace

// for compare results
void set_objectness_distinct(int start,
                             float* out_ptr,
                             int batch,
                             int h,
                             int w,
                             int class_num,
                             int anchor_num) {
  for (int batch_id = 0; batch_id < batch; batch_id++) {
    //[3, 5 + class_num, h , w]
    for (int h_id = 0; h_id < h; h_id++) {
      for (int w_id = 0; w_id < w; w_id++) {
        for (int anchor_id = 0; anchor_id < anchor_num; anchor_id++) {
          *(out_ptr + batch_id * anchor_num * (5 + class_num) * h * w +
            anchor_id * (5 + class_num) * h * w + 4 * h * w + h_id * w + w_id) =
              (start + 1) / 100.0;
          start++;
        }
      }
    }
  }
}

void correct_yolo_box(float& x,
                      float& y,
                      float& w,
                      float& h,
                      float pic_w,
                      float pic_h,
                      float netw,
                      float neth) {
  int new_w = 0;
  int new_h = 0;
  if ((netw / pic_w) < (neth / pic_h)) {
    new_w = netw;
    new_h = (pic_h * netw) / pic_w;
  } else {
    new_h = neth;
    new_w = (pic_w * neth) / pic_h;
  }

  x = (x - (netw - new_w) / 2.) / new_w;
  y = (y - (neth - new_h) / 2.) / new_h;
  w /= (float)new_w;
  h /= (float)new_h;
}

std::vector<float> yolo_tensor_parse_naive(const std::vector<float>& xywh_obj,
                                           const std::vector<float>& class_prob,
                                           const int x_id,
                                           const int y_id,
                                           const float pic_h,
                                           const float pic_w,
                                           const uint gridSize,
                                           const uint numOutputClasses,
                                           const uint netw,
                                           const uint neth,
                                           const int anchor_h,
                                           const int anchor_w,
                                           float prob_thresh) {
  std::vector<float> one_bbox;

  // objectness
  float objectness = xywh_obj[4];
  if (objectness < prob_thresh) {
    return one_bbox;
  }

  // x
  float x = xywh_obj[0];
  x = (float)((x + (float)x_id) * (float)netw) / (float)gridSize;

  // y
  float y = xywh_obj[1];
  y = (float)((y + (float)y_id) * (float)neth) / (float)gridSize;

  // w
  float w = xywh_obj[2];
  w = w * anchor_w;

  // h
  float h = xywh_obj[3];
  h = h * anchor_h;

  correct_yolo_box(x, y, w, h, pic_w, pic_h, netw, neth);

  one_bbox.push_back(objectness);
  one_bbox.push_back(x);
  one_bbox.push_back(y);
  one_bbox.push_back(w);
  one_bbox.push_back(h);

  // Probabilities of classes
  for (uint i = 0; i < numOutputClasses; ++i) {
    float prob = class_prob[i] * objectness;
    one_bbox.push_back(prob < prob_thresh ? 0. : prob);
  }
  return one_bbox;
}

void memcpy_xywh_obj(std::vector<float>& to, const float* from, int stride) {
  for (int i = 0; i < 5; i++) to.push_back(*(from + i * stride));
}

void memcpy_class_prob(std::vector<float>& to,
                       const float* from,
                       int stride,
                       int num) {
  for (int i = 0; i < num; i++) to.push_back(*(from + i * stride));
}

template <typename T>
class YoloBoxComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string input0_ = "x0";
  std::string input1_ = "x1";
  std::string input2_ = "x2";
  std::string input3_ = "image_shape";
  std::string input4_ = "image_scale";

  std::string output0_ = "boxes_scores";
  std::string output1_ = "boxes_num";
  std::vector<int> anchors0_ = {116, 90, 156, 198, 373, 326};
  std::vector<int> anchors1_ = {30, 61, 62, 45, 59, 119};
  std::vector<int> anchors2_ = {10, 13, 16, 30, 33, 23};

  int class_num_ = 80;
  float conf_thresh_ = 0.004;
  int downsample_ratio0_ = 32;
  int downsample_ratio1_ = 16;
  int downsample_ratio2_ = 8;
  bool clip_bbox_ = true;
  float scale_x_y_ = 1.0;
  float nms_thresh_ = 0.3;
  DDim _dims0_{{2, 255, 19, 19}};
  DDim _dims1_{{2, 255, 38, 38}};
  DDim _dims2_{{2, 255, 76, 76}};
  DDim _dims3_{{2, 2}};
  DDim _dims4_{{2, 2}};

 public:
  YoloBoxComputeTester(const Place& place,
                       const std::string& alias,
                       std::vector<int> anchors,
                       int class_num,
                       T conf_thresh,
                       int downsample_ratio)
      : TestCase(place, alias) {
    int anchor_num = anchors0_.size() / 2;
    _dims0_[1] = anchor_num * (5 + class_num);
    _dims1_[1] = anchor_num * (5 + class_num);
    _dims2_[1] = anchor_num * (5 + class_num);
  }

  void RunBaseline(Scope* scope) override {
    const lite::Tensor* x0 = scope->FindTensor(input0_);
    const lite::Tensor* x1 = scope->FindTensor(input1_);
    const lite::Tensor* x2 = scope->FindTensor(input2_);
    std::vector<const lite::Tensor*> x = {x0, x1, x2};
    const lite::Tensor* image_shape = scope->FindTensor(input3_);
    const lite::Tensor* image_scale = scope->FindTensor(input4_);
    const T* image_shape_ptr = image_shape->data<T>();
    const T* image_scale_ptr = image_scale->data<T>();
    lite::Tensor* boxes_scores = scope->NewTensor(output0_);
    lite::Tensor* boxes_num = scope->NewTensor(output1_);

    std::vector<std::vector<int>> anchors = {anchors0_, anchors1_, anchors2_};

    int class_num = class_num_;
    T conf_thresh = conf_thresh_;
    int downsample_ratio0 = downsample_ratio0_;
    int downsample_ratio1 = downsample_ratio1_;
    int downsample_ratio2 = downsample_ratio2_;
    std::vector<int> downsample_ratio = {
        downsample_ratio0, downsample_ratio1, downsample_ratio2};

    int batch = image_shape->dims()[0];
    int result_numbox = 0;
    std::vector<float> result_bbox;

    FILE* f = fopen("/zhoukangkang/lite_jetson_yolo_head/baseline.txt", "w");

    for (int batch_id = 0; batch_id < batch; batch_id++) {
      const float pic_h =
          *(image_shape_ptr + batch_id * 2) / *(image_scale_ptr + batch_id * 2);
      const float pic_w = *(image_shape_ptr + batch_id * 2 + 1) /
                          *(image_scale_ptr + batch_id * 2 + 1);
      for (int input_id = 0; input_id < x.size(); input_id++) {
        int c = x[input_id]->dims()[1];
        int h = x[input_id]->dims()[2];
        int w = x[input_id]->dims()[3];
        const float* input_ptr = x[input_id]->data<T>() + batch_id * c * h * w;
        std::vector<int> anchor = anchors[input_id];

        //[3, 5 + class_num, h , w]
        for (int h_id = 0; h_id < h; h_id++) {
          for (int w_id = 0; w_id < w; w_id++) {
            for (int anchor_id = 0; anchor_id < anchor.size() / 2;
                 anchor_id++) {
              float objectness =
                  *(input_ptr + anchor_id * (5 + class_num) * h * w +
                    4 * h * w + h_id * w + w_id);
              if (objectness > conf_thresh)
                result_numbox++;
              else
                continue;
              std::vector<float> xywh_obj;
              std::vector<float> class_prob;
              memcpy_xywh_obj(xywh_obj,
                              input_ptr + anchor_id * (5 + class_num) * h * w +
                                  h_id * w + w_id,
                              h * w);
              memcpy_class_prob(class_prob,
                                input_ptr +
                                    anchor_id * (5 + class_num) * h * w +
                                    h_id * w + w_id + 5 * h * w,
                                h * w,
                                class_num);

              std::vector<float> one_bbox =
                  yolo_tensor_parse_naive(xywh_obj,
                                          class_prob,
                                          w_id,
                                          h_id,
                                          pic_h,
                                          pic_w,
                                          h,
                                          class_num,
                                          w * downsample_ratio[input_id],
                                          h * downsample_ratio[input_id],
                                          anchor[anchor_id * 2 + 1],
                                          anchor[anchor_id * 2],
                                          conf_thresh);

              for (int i = 0; i < one_bbox.size(); i++) {
                result_bbox.push_back(one_bbox[i]);
              }
            }
          }
        }
      }
    }

    std::vector<std::vector<float>> tmp_result;
    for (int i = 0; i < result_bbox.size(); i += (5 + class_num)) {
      std::vector<float> ele;
      for (int j = 0; j < (5 + class_num); j++) {
        ele.push_back(result_bbox[i + j]);
      }
      tmp_result.push_back(ele);
    }

    auto cmp = [](const std::vector<float>& a,
                  const std::vector<float>& b) -> bool { return a[0] > b[0]; };

    std::sort(tmp_result.begin(), tmp_result.end(), cmp);
    result_bbox.clear();
    for (int i = 0; i < tmp_result.size(); i++)
      for (int j = 0; j < tmp_result[i].size(); j++) {
        result_bbox.push_back(tmp_result[i][j]);
        fprintf(f, "%f\n", tmp_result[i][j]);
      }
    boxes_scores->Resize({result_numbox, 5 + class_num});
    boxes_num->Resize({2});

    T* boxes_scores_ptr = boxes_scores->mutable_data<T>();
    int* boxes_num_ptr = boxes_num->mutable_data<int>();
    boxes_num_ptr[0] = 1;
    boxes_num_ptr[1] = 1;
    memcpy(boxes_scores_ptr,
           result_bbox.data(),
           result_bbox.size() * sizeof(float));
    fclose(f);
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("yolo_box_parser");
    op_desc->SetInput("x0", {input0_});
    op_desc->SetInput("x1", {input1_});
    op_desc->SetInput("x2", {input2_});
    op_desc->SetInput("image_shape", {input3_});
    op_desc->SetInput("image_scale", {input4_});
    op_desc->SetOutput("boxes_scores", {output0_});
    op_desc->SetOutput("boxes_num", {output1_});
    op_desc->SetAttr("anchors0", anchors0_);
    op_desc->SetAttr("anchors1", anchors1_);
    op_desc->SetAttr("anchors2", anchors2_);
    op_desc->SetAttr("class_num", class_num_);
    op_desc->SetAttr("conf_thresh", conf_thresh_);
    op_desc->SetAttr("downsample_ratio0", downsample_ratio0_);
    op_desc->SetAttr("downsample_ratio1", downsample_ratio1_);
    op_desc->SetAttr("downsample_ratio2", downsample_ratio2_);
    op_desc->SetAttr("clip_bbox", clip_bbox_);
    op_desc->SetAttr("scale_x_y", scale_x_y_);
    op_desc->SetAttr("nms_thresh", nms_thresh_);
  }

  void PrepareData() override {
    std::vector<T> data0(_dims0_.production());
    fill_data_rand<T>(data0.data(), 0, 1, data0.size());
    set_objectness_distinct(0,
                            data0.data(),
                            _dims0_[0],
                            _dims0_[2],
                            _dims0_[3],
                            class_num_,
                            anchors0_.size() / 2);
    SetCommonTensor(input0_, _dims0_, data0.data());

    std::vector<T> data1(_dims1_.production());
    fill_data_rand<T>(data1.data(), 0, 1, data1.size());
    set_objectness_distinct(data1.size(),
                            data1.data(),
                            _dims1_[0],
                            _dims1_[2],
                            _dims1_[3],
                            class_num_,
                            anchors1_.size() / 2);
    SetCommonTensor(input1_, _dims1_, data1.data());

    std::vector<T> data2(_dims2_.production());
    fill_data_rand<T>(data2.data(), 0, 1, data2.size());
    set_objectness_distinct(data2.size() + data1.size(),
                            data2.data(),
                            _dims2_[0],
                            _dims2_[2],
                            _dims2_[3],
                            class_num_,
                            anchors2_.size() / 2);
    SetCommonTensor(input2_, _dims2_, data2.data());

    std::vector<T> data3(_dims3_.production());
    fill_data_rand<T>(data3.data(), 608., 608., data3.size());
    SetCommonTensor(input3_, _dims3_, data3.data());

    std::vector<T> data4(_dims4_.production());
    fill_data_rand<T>(data4.data(), 1., 1., data4.size());
    SetCommonTensor(input4_, _dims4_, data4.data());
  }
};

template <typename T>
void TestYoloBox(Place place, float abs_error) {
  for (int class_num : {80}) {
    for (float conf_thresh : {0.0}) {
      for (int downsample_ratio : {32}) {
        std::vector<int> anchor{10, 13, 16, 30, 33, 30};
        std::unique_ptr<arena::TestCase> tester(new YoloBoxComputeTester<T>(
            place, "def", anchor, class_num, conf_thresh, downsample_ratio));
        arena::Arena arena(std::move(tester), place, abs_error);
        arena.TestPrecision();
      }
    }
  }
}

TEST(YoloBoxParser, precision) {
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
