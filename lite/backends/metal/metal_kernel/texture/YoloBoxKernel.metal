/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License. */

#include "Common.metal"
#include <metal_stdlib>

using namespace metal;

struct YoloBoxParam {
    int image_height;
    int image_width;
    int x_n;
    int x_c;
    int x_h;
    int x_w;
    int x_stride;
    int x_size;
    int box_num;
    int anchor_num;
    int anchor_stride;
    int class_num;
    int clip_bbox;
    float conf_thresh;
    float scale;
    float bias_value;
};

// yolo_box
// document：https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/ops/yolo_box_cn.html#yolo-box
// anchor_box：https://blog.csdn.net/ybdesire/article/details/82860607
// yolo_v3: https://aistudio.baidu.com/aistudio/projectdetail/1353575

inline ftype sigmoid(const ftype x) {
    return 1.0 / (1.0 + exp(-x));
}

inline ftype4 get_yolo_box(const device ftype* x_data,
    const device int* anchors_data,
    const int l,
    const int k,
    const int anchor_idx,
    const int x_h,
    const int x_size,
    const int box_idx,
    const int x_stride,
    const int img_height,
    const int img_width,
    const ftype scale,
    const ftype bias_value) {
    ftype4 box;
    box[0] = (l + sigmoid(x_data[box_idx]) * scale + bias_value) * img_width / x_h;
    box[1] = (k + sigmoid(x_data[box_idx + x_stride]) * scale + bias_value) * img_height / x_h;
    box[2] =
        exp(x_data[box_idx + x_stride * 2]) * anchors_data[2 * anchor_idx] * img_width / x_size;
    box[3] = exp(x_data[box_idx + x_stride * 3]) * anchors_data[2 * anchor_idx + 1] * img_height /
             x_size;
    return box;
}

inline void calc_detection_box(device ftype* boxes_data,
    const ftype4 box,
    const int box_idx,
    const int img_height,
    const int img_width,
    const int clip_bbox) {
    boxes_data[box_idx] = box[0] - box[2] / 2;
    boxes_data[box_idx + 1] = box[1] - box[3] / 2;
    boxes_data[box_idx + 2] = box[0] + box[2] / 2;
    boxes_data[box_idx + 3] = box[1] + box[3] / 2;
    if (clip_bbox) {
        boxes_data[box_idx] = fmax(boxes_data[box_idx], 0);
        boxes_data[box_idx + 1] = fmax(boxes_data[box_idx + 1], 0);
        boxes_data[box_idx + 2] = fmin(boxes_data[box_idx + 2], img_width - 1);
        boxes_data[box_idx + 3] = fmin(boxes_data[box_idx + 3], img_height - 1);
    }
}

inline void calc_label_score(device ftype* scores_data,
    const device ftype* x_data,
    const int label_idx,
    const int score_idx,
    const int class_num,
    const ftype conf,
    const int x_stride) {
    for (int i = 0; i < class_num; i++) {
        scores_data[score_idx + i] = conf * sigmoid(x_data[label_idx + i * x_stride]);
    }
}


kernel void yolo_box(device ftype* x_data[[buffer(0)]],
    device int* anchors_data[[buffer(1)]],
    device ftype* boxes_data[[buffer(2)]],
    device ftype* scores_data[[buffer(3)]],
    constant YoloBoxParam& param[[buffer(4)]],
    uint3 gid[[thread_position_in_grid]]) {
    const int l = gid.x;
    const int k = gid.y;
    const int anchor_idx = gid.z;

    // TODO:don't support multi images, input.n > 1
    const int imgsize_num = 1;
    const int img_height = param.image_height;
    const int img_width = param.image_width;
    const int x_h = param.x_h;
    const int x_w = param.x_w;
    const int hw_idx = k * x_w + l;
    const int x_stride = param.x_stride;
    const int x_size = param.x_size;
    const int box_num = param.box_num;
    const int anchor_num = param.anchor_num;
    const int anchor_stride = param.anchor_stride;
    const int class_num = param.class_num;
    const int clip_bbox = param.clip_bbox;
    const ftype conf_thresh = ftype(param.conf_thresh);
    const ftype scale = ftype(param.scale);
    const ftype bias_value = ftype(param.bias_value);

    for (int imgsize_idx = 0; imgsize_idx < imgsize_num; imgsize_idx++) {
        const int obj_idx =
            (imgsize_idx * anchor_num + anchor_idx) * anchor_stride + 4 * x_stride + hw_idx;
        ftype conf = sigmoid(x_data[obj_idx]);
        if (conf < conf_thresh) {
            continue;
        }

        // get yolo box
        int box_idx =
            (imgsize_idx * anchor_num + anchor_idx) * anchor_stride + 0 * x_stride + hw_idx;
        ftype4 box = get_yolo_box(x_data,
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
            bias_value);

        // get box id, label id
        box_idx = (imgsize_idx * box_num + anchor_idx * x_stride + k * x_w + l) * 4;
        calc_detection_box(boxes_data, box, box_idx, img_height, img_width, clip_bbox);

        const int label_idx =
            (imgsize_idx * anchor_num + anchor_idx) * anchor_stride + 5 * x_stride + hw_idx;
        int score_idx = (imgsize_idx * box_num + anchor_idx * x_stride + k * x_w + l) * class_num;

        // get label score
        calc_label_score(scores_data, x_data, label_idx, score_idx, class_num, conf, x_stride);
    }
}
