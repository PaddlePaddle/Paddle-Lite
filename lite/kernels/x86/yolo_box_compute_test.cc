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

#include "lite/kernels/x86/yolo_box_compute.h"
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {
namespace test {
float sigmoid_base(float x) { return 1.f / (1.f + expf(-x)); }

void get_yolo_box_base(float* box,
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
  box[0] = (i + sigmoid_base(x[index])) * img_width / grid_size;
  box[1] = (j + sigmoid_base(x[index + stride])) * img_height / grid_size;
  box[2] = std::exp(x[index + 2 * stride]) * anchors[2 * an_idx] * img_width /
           input_size;
  box[3] = std::exp(x[index + 3 * stride]) * anchors[2 * an_idx + 1] *
           img_height / input_size;
}

int get_entry_index_base(int batch,
                         int an_idx,
                         int hw_idx,
                         int an_num,
                         int an_stride,
                         int stride,
                         int entry) {
  return (batch * an_num + an_idx) * an_stride + entry * stride + hw_idx;
}

void calc_detection_box_base(float* boxes,
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

void calc_label_score_base(float* scores,
                           const float* input,
                           const int label_idx,
                           const int score_idx,
                           const int class_num,
                           const float conf,
                           const int stride) {
  for (int i = 0; i < class_num; i++) {
    scores[score_idx + i] = conf * sigmoid_base(input[label_idx + i * stride]);
  }
}

void RunBaseline(const lite::Tensor* X,
                 const lite::Tensor* ImgSize,
                 lite::Tensor* Boxes,
                 lite::Tensor* Scores,
                 int class_num,
                 float conf_thresh,
                 int downsample_ratio,
                 std::vector<int> anchors) {
  auto* in = X;
  auto* imgsize = ImgSize;

  const int n = in->dims()[0];
  const int h = in->dims()[2];
  const int w = in->dims()[3];
  const int an_num = anchors.size() / 2;
  int in_size = downsample_ratio * h;
  int box_num = in->dims()[2] * in->dims()[3] * an_num;

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
          int obj_idx = test::get_entry_index_base(
              i, j, k * w + l, an_num, an_stride, stride, 4);
          float conf = test::sigmoid_base(in_data[obj_idx]);
          if (conf < conf_thresh) {
            continue;
          }
          int box_idx = test::get_entry_index_base(
              i, j, k * w + l, an_num, an_stride, stride, 0);
          test::get_yolo_box_base(box,
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
          test::calc_detection_box_base(
              boxes_data, box, box_idx, img_height, img_width);

          int label_idx = test::get_entry_index_base(
              i, j, k * w + l, an_num, an_stride, stride, 5);
          int score_idx = (i * b_num + j * stride + k * w + l) * class_num;
          test::calc_label_score_base(scores_data,
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
}  // namespace test

TEST(yolo_box_x86, retrive_op) {
  auto yolo_box =
      KernelRegistry::Global().Create<TARGET(kX86), PRECISION(kFloat)>(
          "yolo_box");
  ASSERT_FALSE(yolo_box.empty());
  ASSERT_TRUE(yolo_box.front());
}

TEST(yolo_box_x86, init) {
  YoloBoxCompute<float> yolo_box;
  ASSERT_EQ(yolo_box.precision(), PRECISION(kFloat));
  ASSERT_EQ(yolo_box.target(), TARGET(kX86));
}

TEST(yolo_box_x86, run_test) {
  lite::Tensor X, ImgSize, Boxes, Scores, Boxes_base, Scores_base;
  YoloBoxCompute<float> yolo_box;
  operators::YoloBoxParam param;
  int s = 3, cls = 4;
  int n = 1, c = s * (5 + cls), h = 16, w = 16;
  param.anchors = {2, 3, 4, 5, 8, 10};
  param.downsample_ratio = 2;
  param.conf_thresh = 0.5;
  param.class_num = cls;
  int m = h * w * param.anchors.size() / 2;
  X.Resize({n, c, h, w});
  ImgSize.Resize({1, 2});
  Boxes.Resize({n, m, 4});
  Boxes_base.Resize({n, m, 4});
  Scores.Resize({n, cls, m});
  Scores_base.Resize({n, cls, m});

  auto x_data = X.mutable_data<float>();
  auto imgsize_data = ImgSize.mutable_data<float>();
  auto boxes_data = Boxes.mutable_data<float>();
  auto scores_data = Scores.mutable_data<float>();
  auto boxes_base_data = Boxes_base.mutable_data<float>();
  auto scores_base_data = Scores_base.mutable_data<float>();

  for (int i = 0; i < X.dims().production(); i++) {
    x_data[i] = static_cast<float>(i);
  }
  for (int i = 0; i < ImgSize.dims().production(); i++) {
    imgsize_data[i] = static_cast<float>(i);
  }
  test::RunBaseline(&X,
                    &ImgSize,
                    &Boxes_base,
                    &Scores_base,
                    param.class_num,
                    param.conf_thresh,
                    param.downsample_ratio,
                    param.anchors);
  param.X = &X;
  param.ImgSize = &ImgSize;
  param.Boxes = &Boxes;
  param.Scores = &Scores;

  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<X86Context>();
  yolo_box.SetContext(std::move(ctx));
  yolo_box.SetParam(std::move(param));
  yolo_box.Run();

  for (int i = 0; i < Boxes.dims().production(); i++) {
    EXPECT_NEAR(boxes_data[i], boxes_base_data[i], 1e-5);
  }
  for (int i = 0; i < Scores.dims().production(); i++) {
    EXPECT_NEAR(scores_data[i], scores_base_data[i], 1e-5);
  }
}

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(yolo_box, kX86, kFloat, kNCHW, def);
