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

#ifndef LITE_KERNELS_MLU_ROI_ALIGN_KERNEL_H_
#define LITE_KERNELS_MLU_ROI_ALIGN_KERNEL_H_

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef uint16_t half;

/**
 * @brief Region of interests align is used to implement bilinear interpolation.
 * It can change the input of uneven size into a fixed size feature map. The
 * operation passes pooled_width and pooled_height divides each recommended
 * area into equal sized blocks. The position remains the same. In each ROI
 * block, take sampling_ratio points (if - 1, all points in the frame are
 * taken). Each point is directly calculated by bilinear interpolation. Then
 * take the average value of the points taken in the block as the coordinate
 * value of the small box.
 *
 * @param[in] input: 4-D sensor of shape [N, H, W, C], n is the batch size, C is
 * the number of input channels, H feature height and W feature width. Datatype
 * is float16
 * @param[in] rois: 2-D tensor of shape [num_rois, 4]. ROIs to be pooled
 * (regions of interest). For example [[x1, Y1, X2, Y2],...], (x1, Y1) is the
 * upper left point coordinate, (X2, Y2) is the lower right point coordinate.
 * Data type is float16
 * @param[in] roi_ind: 1-D tensor of shape [num_boxes] with values in [0,
 * batch). The value of box_ind[i] specifies the image that the i-th roi refers
 * to. Data type is int
 * @param[out] output: 4-D tensor of shape [num_rois, pooled_height,
 * pooled_weight, C].
 * @param[in] height: The height of input
 * @param[in] width: The width of input
 * @param[in] channels: The channel of input
 * @param[in] pooled_height: Output height after pooling
 * @param[in] pooled_width: Output width after pooling
 * @param[in] num_rois: The number of roi
 * @param[in] spatial_scale: The scale factor of multiplicative space, when
 * pooling, transforms the ROI coordinate to the scale used in the
 * operation.image_height * spatial_scale == featuremap_height, width is also
 * like this
 * @param[in] sampling_ratio: The number of sampling points in the interpolation
 * lattice. If it < = 0, they will adapt to ROI_Width and pooled_W, the same is
 * true for height.
 * @retval void
 */
void roi_align_kernel(half *input,
                      half *rois,
                      int *roi_ind,
                      half *output,
                      const int height,
                      const int width,
                      const int channels,
                      const int pooled_height,
                      const int pooled_width,
                      const int rois_num,
                      const half spatial_scale,
                      const int sampling_ratio);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // LITE_KERNELS_MLU_ROI_ALIGN_KERNEL_H_
