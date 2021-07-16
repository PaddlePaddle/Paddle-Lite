/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <cl_common.h>

// native sigmoid using native_exp
#define nati_sigmoid(x) (1.f / (1.f + native_exp(-x)))

#define get_entry_index(                                     \
    batch, an_idx, hw_idx, an_num, an_stride, stride, entry) \
  ((batch * an_num + an_idx) * an_stride + entry * stride + hw_idx)

#define get_yolo_box(box,          \
                     x_data,       \
                     anchors_data, \
                     l,            \
                     k,            \
                     anchor_idx,   \
                     x_h,          \
                     x_size,       \
                     box_idx,      \
                     x_stride,     \ 
                     img_height,   \
                     img_width,    \
                     scale,        \
                     bias) \
  {                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          \
    box[0] =                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 \
        (l + nati_sigmoid(x_data[box_idx]) * scale + bias) * img_width / x_h;                                                                                                                                                                                                                                                                                                                                                                                                                                                \
    box[1] = (k + nati_sigmoid(x_data[box_idx + x_stride]) * scale + bias) *                                                                                                                                                                                                                                                                                                                                                                                                                                                 \
             img_height / x_h;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               \
    box[2] = native_exp(x_data[box_idx + x_stride * 2]) *                                                                                                                                                                                                                                                                                                                                                                                                                                                                    \
             anchors_data[2 * anchor_idx] * img_width / x_size;                                                                                                                                                                                                                                                                                                                                                                                                                                                              \
    box[3] = native_exp(x_data[box_idx + x_stride * 3]) *                                                                                                                                                                                                                                                                                                                                                                                                                                                                    \
             anchors_data[2 * anchor_idx + 1] * img_height / x_size;                                                                                                                                                                                                                                                                                                                                                                                                                                                         \
  }

#define calc_detection_box(                                                    \
    boxes_data, box, box_idx, img_height, img_width, clip_bbox)                \
  {                                                                            \
    boxes_data[box_idx] = box[0] - box[2] / 2;                                 \
    boxes_data[box_idx + 1] = box[1] - box[3] / 2;                             \
    boxes_data[box_idx + 2] = box[0] + box[2] / 2;                             \
    boxes_data[box_idx + 3] = box[1] + box[3] / 2;                             \
    if (clip_bbox) {                                                           \
      boxes_data[box_idx] = fmax(boxes_data[box_idx], 0);                      \
      boxes_data[box_idx + 1] = fmax(boxes_data[box_idx + 1], 0);              \
      boxes_data[box_idx + 2] = fmin(boxes_data[box_idx + 2], img_width - 1);  \
      boxes_data[box_idx + 3] = fmin(boxes_data[box_idx + 3], img_height - 1); \
    }                                                                          \
  }

#define calc_label_score(                                                 \
    scores_data, x_data, label_idx, score_idx, class_num, conf, x_stride) \
  {                                                                       \
    for (int i = 0; i < class_num; i++) {                                 \
      scores_data[score_idx + i] =                                        \
          conf * nati_sigmoid(x_data[label_idx + i * x_stride]);          \
    }                                                                     \
  }

////////////////////////////////
// main body
////////////////////////////////
__kernel void yolo_box(__global const CL_DTYPE* x_data,
                       const int x_n,
                       const int x_c,
                       const int x_h,
                       const int x_w,
                       const int x_stride,
                       const int x_size,
                       __global const int* imgsize_data,
                       const int imgsize_num,
                       __global CL_DTYPE* boxes_data,
                       const int box_num,
                       __global CL_DTYPE* scores_data,
                       __global const int* anchors_data,
                       const int anchor_num,
                       const int anchor_stride,
                       const int class_num,
                       const int clip_bbox,
                       const float conf_thresh,
                       const float scale,
                       const float bias) {
  const int k = get_global_id(0);           // [0, x_h)
  const int l = get_global_id(1);           // [0, x_w)
  const int anchor_idx = get_global_id(2);  // [0, anchors_num)

  CL_DTYPE box[4];
  for (int imgsize_idx = 0; imgsize_idx < imgsize_num; imgsize_idx++) {
    const int img_height = imgsize_data[2 * imgsize_idx];
    const int img_width = imgsize_data[2 * imgsize_idx + 1];
    const int obj_idx = get_entry_index(imgsize_idx,
                                        anchor_idx,
                                        k * x_w + l,
                                        anchor_num,
                                        anchor_stride,
                                        x_stride,
                                        4);
    float conf = nati_sigmoid(x_data[obj_idx]);
    if (conf < conf_thresh) continue;

    // get yolo box
    int box_idx = get_entry_index(imgsize_idx,
                                  anchor_idx,
                                  k * x_w + l,
                                  anchor_num,
                                  anchor_stride,
                                  x_stride,
                                  0);
    get_yolo_box(box,
                 x_data,
                 anchors_data,
                 l,
                 k,
                 anchor_idx,
                 x_h,
                 x_size,
                 box_idx,
                 x_stride,
                 img_height,
                 img_width,
                 scale,
                 bias);

    // get box id, label id
    box_idx = (imgsize_idx * box_num + anchor_idx * x_stride + k * x_w + l) * 4;
    calc_detection_box(
        boxes_data, box, box_idx, img_height, img_width, clip_bbox);
    const int label_idx = get_entry_index(imgsize_idx,
                                          anchor_idx,
                                          k * x_w + l,
                                          anchor_num,
                                          anchor_stride,
                                          x_stride,
                                          5);
    int score_idx =
        (imgsize_idx * box_num + anchor_idx * x_stride + k * x_w + l) *
        class_num;

    // get label score
    calc_label_score(
        scores_data, x_data, label_idx, score_idx, class_num, conf, x_stride);
  }
}
