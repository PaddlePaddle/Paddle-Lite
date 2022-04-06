// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "driver/nvidia_tensorrt/kernel/cuda/yolo.h"
#include <stdlib.h>
#include <algorithm>

namespace nnadapter {
namespace nvidia_tensorrt {
namespace cuda {

static int nms_comparator(const void *pa, const void *pb) {
  detection a = *(detection *)pa;
  detection b = *(detection *)pb;
  float diff = 0;

  if (b.sort_class >= 0) {
    diff = a.prob[b.sort_class] - b.prob[b.sort_class];
  } else {
    diff = a.objectness - b.objectness;
  }

  if (diff < 0)
    return 1;
  else if (diff > 0)
    return -1;
  return 0;
}

float overlap(float x1, float w1, float x2, float w2) {
  float l1 = x1 - w1 / 2;
  float l2 = x2 - w2 / 2;
  float left = l1 > l2 ? l1 : l2;
  float r1 = x1 + w1 / 2;
  float r2 = x2 + w2 / 2;
  float right = r1 < r2 ? r1 : r2;
  return right - left;
}

float box_intersection(box a, box b) {
  float w = overlap(a.x, a.w, b.x, b.w);
  float h = overlap(a.y, a.h, b.y, b.h);
  if (w < 0 || h < 0) return 0;
  float area = w * h;
  return area;
}

float box_union(box a, box b) {
  float i = box_intersection(a, b);
  float u = a.w * a.h + b.w * b.h - i;
  return u;
}

float box_iou(box a, box b) { return box_intersection(a, b) / box_union(a, b); }

void post_nms(std::vector<detection> &det_bboxes, float thresh, int classes) {
  int total = det_bboxes.size();
  if (total <= 0) {
    return;
  }

  detection *dets = det_bboxes.data();

  int i, j, k;
  k = total - 1;
  for (i = 0; i <= k; ++i) {
    if (dets[i].objectness == 0) {
      detection swap = dets[i];
      dets[i] = dets[k];
      dets[k] = swap;
      --k;
      --i;
    }
  }
  total = k + 1;

  qsort(dets, total, sizeof(detection), nms_comparator);

  for (i = 0; i < total; ++i) {
    if (dets[i].objectness == 0) {
      continue;
    }

    box a = dets[i].bbox;

    for (j = i + 1; j < total; ++j) {
      if (dets[j].objectness == 0) {
        continue;
      }

      box b = dets[j].bbox;

      if (box_iou(a, b) > thresh) {
        dets[j].objectness = 0;
        for (k = 0; k < classes; ++k) {
          dets[j].prob[k] = 0;
        }
      }
    }
  }
}

}  // namespace cuda
}  // namespace nvidia_tensorrt
}  // namespace nnadapter
