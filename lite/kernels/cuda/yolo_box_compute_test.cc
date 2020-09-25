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

#include "lite/kernels/cuda/yolo_box_compute.h"
#include <gtest/gtest.h>
#include <memory>
#include <utility>

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

inline static float sigmoid(float x) { return 1.f / (1.f + expf(-x)); }

inline static void get_yolo_box(float* box,
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
                                int img_width,
                                float scale,
                                float bias) {
  box[0] = (i + sigmoid(x[index]) * scale + bias) * img_width / grid_size;
  box[1] =
      (j + sigmoid(x[index + stride] * scale + bias)) * img_height / grid_size;
  box[2] = std::exp(x[index + 2 * stride]) * anchors[2 * an_idx] * img_width /
           input_size;
  box[3] = std::exp(x[index + 3 * stride]) * anchors[2 * an_idx + 1] *
           img_height / input_size;
}

inline static int get_entry_index(int batch,
                                  int an_idx,
                                  int hw_idx,
                                  int an_num,
                                  int an_stride,
                                  int stride,
                                  int entry) {
  return (batch * an_num + an_idx) * an_stride + entry * stride + hw_idx;
}

inline static void calc_detection_box(float* boxes,
                                      float* box,
                                      const int box_idx,
                                      const int img_height,
                                      const int img_width,
                                      bool clip_bbox) {
  boxes[box_idx] = box[0] - box[2] / 2;
  boxes[box_idx + 1] = box[1] - box[3] / 2;
  boxes[box_idx + 2] = box[0] + box[2] / 2;
  boxes[box_idx + 3] = box[1] + box[3] / 2;
  if (!clip_bbox) {
    return;
  }
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

inline static void calc_label_score(float* scores,
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

template <typename T>
static void YoloBoxRef(const T* input,
                       const int* imgsize,
                       T* boxes,
                       T* scores,
                       const float conf_thresh,
                       const int* anchors,
                       const int n,
                       const int h,
                       const int w,
                       const int an_num,
                       const int class_num,
                       const int box_num,
                       int input_size,
                       bool clip_bbox,
                       float scale,
                       float bias) {
  const int stride = h * w;
  const int an_stride = (class_num + 5) * stride;
  float box[4];

  for (int i = 0; i < n; i++) {
    int img_height = imgsize[2 * i];
    int img_width = imgsize[2 * i + 1];

    for (int j = 0; j < an_num; j++) {
      for (int k = 0; k < h; k++) {
        for (int l = 0; l < w; l++) {
          int obj_idx =
              get_entry_index(i, j, k * w + l, an_num, an_stride, stride, 4);
          float conf = sigmoid(input[obj_idx]);
          if (conf < conf_thresh) {
            continue;
          }

          int box_idx =
              get_entry_index(i, j, k * w + l, an_num, an_stride, stride, 0);
          get_yolo_box(box,
                       input,
                       anchors,
                       l,
                       k,
                       j,
                       h,
                       input_size,
                       box_idx,
                       stride,
                       img_height,
                       img_width,
                       scale,
                       bias);
          box_idx = (i * box_num + j * stride + k * w + l) * 4;
          calc_detection_box(
              boxes, box, box_idx, img_height, img_width, clip_bbox);

          int label_idx =
              get_entry_index(i, j, k * w + l, an_num, an_stride, stride, 5);
          int score_idx = (i * box_num + j * stride + k * w + l) * class_num;
          calc_label_score(
              scores, input, label_idx, score_idx, class_num, conf, stride);
        }
      }
    }
  }
}

TEST(yolo_box, normal) {
  YoloBoxCompute yolo_box_kernel;
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  auto& context = ctx->As<CUDAContext>();

  operators::YoloBoxParam param;

  lite::Tensor x, sz, x_cpu, sz_cpu;
  lite::Tensor boxes, scores, boxes_cpu, scores_cpu;
  lite::Tensor x_ref, sz_ref, boxes_ref, scores_ref;
  int s = 3, cls = 4;
  int n = 1, c = s * (5 + cls), h = 16, w = 16;
  param.anchors = {2, 3, 4, 5, 8, 10};
  param.downsample_ratio = 2;
  param.conf_thresh = 0.5;
  param.class_num = cls;
  param.clip_bbox = true;
  param.scale_x_y = 1.0;
  float bias = -0.5 * (param.scale_x_y - 1.);
  int m = h * w * param.anchors.size() / 2;

  x.Resize({n, c, h, w});
  sz.Resize({1, 2});
  boxes.Resize({n, m, 4});
  scores.Resize({n, cls, m});

  x_cpu.Resize({n, c, h, w});
  sz_cpu.Resize({1, 2});
  boxes_cpu.Resize({n, m, 4});
  scores_cpu.Resize({n, cls, m});

  x_ref.Resize({n, c, h, w});
  sz_ref.Resize({1, 2});
  boxes_ref.Resize({n, m, 4});
  scores_ref.Resize({n, cls, m});

  auto* boxes_data = boxes.mutable_data<float>(TARGET(kCUDA));
  auto* scores_data = scores.mutable_data<float>(TARGET(kCUDA));

  float* x_cpu_data = x_cpu.mutable_data<float>();
  int* sz_cpu_data = sz_cpu.mutable_data<int>();
  float* boxes_cpu_data = boxes_cpu.mutable_data<float>();
  float* scores_cpu_data = scores_cpu.mutable_data<float>();

  float* x_ref_data = x_ref.mutable_data<float>();
  int* sz_ref_data = sz_ref.mutable_data<int>();
  float* boxes_ref_data = boxes_ref.mutable_data<float>();
  float* scores_ref_data = scores_ref.mutable_data<float>();

  for (int i = 0; i < x_cpu.numel(); ++i) {
    x_cpu_data[i] = i - 5.0;
    x_ref_data[i] = i - 5.0;
  }
  sz_cpu_data[0] = 16;
  sz_cpu_data[1] = 32;
  sz_ref_data[0] = 16;
  sz_ref_data[1] = 32;

  x.Assign<float, lite::DDim, TARGET(kCUDA)>(x_cpu_data, x_cpu.dims());
  sz.Assign<int, lite::DDim, TARGET(kCUDA)>(sz_cpu_data, sz_cpu.dims());

  param.X = &x;
  param.ImgSize = &sz;
  param.Boxes = &boxes;
  param.Scores = &scores;
  yolo_box_kernel.SetParam(param);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  context.SetExecStream(stream);

  yolo_box_kernel.SetContext(std::move(ctx));
  yolo_box_kernel.Launch();
  cudaDeviceSynchronize();

  CopySync<TARGET(kCUDA)>(boxes_cpu_data,
                          boxes_data,
                          sizeof(float) * boxes.numel(),
                          IoDirection::DtoH);
  CopySync<TARGET(kCUDA)>(scores_cpu_data,
                          scores_data,
                          sizeof(float) * scores.numel(),
                          IoDirection::DtoH);

  YoloBoxRef<float>(x_ref_data,
                    sz_ref_data,
                    boxes_ref_data,
                    scores_ref_data,
                    param.conf_thresh,
                    param.anchors.data(),
                    n,
                    h,
                    w,
                    param.anchors.size() / 2,
                    cls,
                    m,
                    param.downsample_ratio * h,
                    param.clip_bbox,
                    param.scale_x_y,
                    bias);

  for (int i = 0; i < boxes.numel(); i++) {
    EXPECT_NEAR(boxes_cpu_data[i], boxes_ref_data[i], 1e-5);
  }
  for (int i = 0; i < scores.numel(); i++) {
    EXPECT_NEAR(scores_cpu_data[i], scores_ref_data[i], 1e-5);
  }
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
