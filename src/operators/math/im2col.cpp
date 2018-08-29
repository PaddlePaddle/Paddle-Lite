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

#include "operators/math/im2col.h"
#include <vector>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#include "common/types.h"
namespace paddle_mobile {
namespace operators {
namespace math {

/*
 * im = [input_channels, input_height, input_width]
 * col =
 *   [input_channels, filter_height, filter_width, output_height,
 * output_width]
 */
template <class T>
class Im2ColFunctor<ColFormat::kCFO, CPU, T> {
 public:
  void operator()(const framework::Tensor &im, const std::vector<int> &dilation,
                  const std::vector<int> &stride,
                  const std::vector<int> &padding, framework::Tensor *col) {
    //    PADDLE_ENFORCE(im.dims().size() == 3);
    //    PADDLE_ENFORCE(col->dims().size() == 5);

    int im_channels = im.dims()[0];
    int im_height = im.dims()[1];
    int im_width = im.dims()[2];
    int filter_height = col->dims()[1];
    int filter_width = col->dims()[2];
    int col_height = col->dims()[3];
    int col_width = col->dims()[4];

    //    PADDLE_ENFORCE_EQ((im_height + padding[0] + padding[2]
    //    -
    //                       ((dilation[0] * (filter_height - 1)
    //                       + 1))) /
    //                              stride[0] +
    //                          1,
    //                      col_height,
    //                      "Output_height and
    //                      padding(padding_up, padding_down)
    //                      are " "inconsistent.");
    //    PADDLE_ENFORCE_EQ((im_width + padding[1] + padding[3]
    //    -
    //                       ((dilation[1] * (filter_width - 1)
    //                       + 1))) /
    //                              stride[1] +
    //                          1,
    //                      col_width,
    //                      "Output_height and
    //                      padding(padding_up, padding_down)
    //                      are " "inconsistent.");

    int channels_col = im_channels * filter_height * filter_width;
    const T *im_data = im.data<T>();
    T *col_data = col->data<T>();
#if __ARM_NEON
    const int osize = col_height;
    const int isize = im_height;
    bool pad1 = padding[0] > 0;
    bool pad2 =
        (pad1 &&
         (((isize - 2 * padding[0] + filter_height) % stride[0] == 0) ? 1 : 0));
    int fill = isize % 2;
    if (stride[0] == 1 && filter_height == 3 && pad1 && pad2 &&
        dilation[0] == 1 && im_height > 2) {
      for (int c = 0; c < im_channels; ++c) {
        int oosize = osize * osize;
        int nk4 = osize / 4;
        int mk4 = osize % 4;

        float *col0 = col_data + 0 * oosize + 2 * osize + 2;
        float *col1 = col_data + 1 * oosize + 2 * osize + 1;
        float *col2 = col_data + 2 * oosize + 2 * osize;

        float *col3 = col_data + 3 * oosize + osize + 2;
        float *col4 = col_data + 4 * oosize + osize + 1;
        float *col5 = col_data + 5 * oosize + osize;

        float *col6 = col_data + 6 * oosize + 2;
        float *col7 = col_data + 7 * oosize + 1;
        float *col8 = col_data + 8 * oosize;

        float32x4_t im1;
        const float *im_tmp_data = im_data + osize + 1;

        int rrsize = oosize - osize - 1;
        int nr4 = rrsize / 4;
        int mr4 = rrsize % 4;
        for (int i = 0; i < nr4; ++i) {
          im1 = vld1q_f32(im_tmp_data);
          vst1q_f32(col0, im1);
          vst1q_f32(col1, im1);
          vst1q_f32(col2, im1);
          vst1q_f32(col3, im1);
          vst1q_f32(col4, im1);
          vst1q_f32(col5, im1);
          vst1q_f32(col6, im1);
          vst1q_f32(col7, im1);
          vst1q_f32(col8, im1);

          col0 += 4;
          col1 += 4;
          col2 += 4;
          col3 += 4;
          col4 += 4;
          col5 += 4;
          col6 += 4;
          col7 += 4;
          col8 += 4;

          im_tmp_data += 4;
        }
        for (int i = 0; i < mr4; ++i) {
          *col0 = *im_tmp_data;
          *col1 = *im_tmp_data;
          *col2 = *im_tmp_data;
          *col3 = *im_tmp_data;
          *col4 = *im_tmp_data;
          *col5 = *im_tmp_data;
          *col6 = *im_tmp_data;
          *col7 = *im_tmp_data;
          *col8 = *im_tmp_data;

          col0++;
          col1++;
          col2++;
          col3++;
          col4++;
          col5++;
          col6++;
          col7++;
          col8++;

          im_tmp_data++;
        }

        im_tmp_data = im_data + 1;
        col0 = col_data + 0 * oosize + osize + 2;
        col1 = col_data + 1 * oosize + osize + 1;
        col2 = col_data + 2 * oosize + osize;

        col3 = col_data + 3 * oosize + 2;
        col4 = col_data + 4 * oosize + 1;
        col5 = col_data + 5 * oosize;

        for (int i = 0; i < nk4; i++) {
          im1 = vld1q_f32(im_tmp_data);
          vst1q_f32(col0, im1);
          vst1q_f32(col1, im1);
          vst1q_f32(col2, im1);
          vst1q_f32(col3, im1);
          vst1q_f32(col4, im1);
          vst1q_f32(col5, im1);

          col0 += 4;
          col1 += 4;
          col2 += 4;
          col3 += 4;
          col4 += 4;
          col5 += 4;
          im_tmp_data += 4;
        }

        for (int i = 0; i < mk4; i++) {
          *col0 = *im_tmp_data;
          *col1 = *im_tmp_data;
          *col2 = *im_tmp_data;
          *col3 = *im_tmp_data;
          *col4 = *im_tmp_data;
          *col5 = *im_tmp_data;
          col0++;
          col1++;
          col2++;
          col3++;
          col4++;
          col5++;

          im_tmp_data++;
        }

        // fill 0 1 11;
        for (int i = 0; i < osize; ++i) {
          col_data[0 * oosize + i * osize] = 0.0;
          col_data[3 * oosize + i * osize] = 0.0;
          col_data[6 * oosize + i * osize] = 0.0;

          col_data[2 * oosize + osize - 1 + i * osize] = 0.0;
          col_data[5 * oosize + osize - 1 + i * osize] = 0.0;
          col_data[8 * oosize + osize - 1 + i * osize] = 0.0;
        }

        col_data[0 * oosize + osize + 1] = im_data[0];
        col_data[3 * oosize + 1] = im_data[0];
        col_data[6 * oosize + 1] = im_data[osize];

        col_data[1 * oosize + osize] = im_data[0];
        col_data[4 * oosize] = im_data[0];
        col_data[7 * oosize] = im_data[osize];

        float32x4_t zero4;
        zero4 = vdupq_n_f32(0.0);
        auto col_z0 = col_data;
        auto col_z1 = col_data + oosize;
        auto col_z2 = col_data + 2 * oosize;
        auto col_z6 = col_data + 6 * oosize + osize * (osize - 1);
        auto col_z7 = col_data + 7 * oosize + osize * (osize - 1);
        auto col_z8 = col_data + 8 * oosize + osize * (osize - 1);

        for (int i = 0; i < nk4; ++i) {
          vst1q_f32(col_z0, zero4);
          vst1q_f32(col_z1, zero4);
          vst1q_f32(col_z2, zero4);
          vst1q_f32(col_z6, zero4);
          vst1q_f32(col_z7, zero4);
          vst1q_f32(col_z8, zero4);

          col_z0 += 4;
          col_z1 += 4;
          col_z2 += 4;
          col_z6 += 4;
          col_z7 += 4;
          col_z8 += 4;
        }

        for (int i = 0; i < mk4; ++i) {
          col_z0[i] = 0.0;
          col_z1[i] = 0.0;
          col_z2[i] = 0.0;
          col_z6[i] = 0.0;
          col_z7[i] = 0.0;
          col_z8[i] = 0.0;
        }
        col_data += 9 * oosize;
        im_data += isize * isize;
      }
    } else if (stride[0] == 2 && filter_height == 3 && pad1 &&
               dilation[0] == 1 && im_height > 2) {
      for (int c = 0; c < im_channels; ++c) {
        int oosize = osize * osize;
        int nk4 = osize / 4;
        int mk4 = osize % 4;

        // 3 2 3 1 0 1 3 2 3
        float *col0 = col_data + 0 * oosize + osize + 1;
        float *col1 = col_data + 1 * oosize + osize;
        float *col2 = col_data + 2 * oosize + osize;

        float *col3 = col_data + 3 * oosize + 1;
        float *col4 = col_data + 4 * oosize;
        float *col5 = col_data + 5 * oosize;

        float *col6 = col_data + 6 * oosize + 1;
        float *col7 = col_data + 7 * oosize;
        float *col8 = col_data + 8 * oosize;

        float32x4x2_t im01;
        float32x4x2_t im23;
        const float *im_tmp_data0 = im_data;
        const float *im_tmp_data2 = im_data + isize;

        for (int j = 0; j < osize; ++j) {
          for (int i = 0; i < nk4; ++i) {
            im01 = vld2q_f32(im_tmp_data0);
            im23 = vld2q_f32(im_tmp_data2);
            vst1q_f32(col0, im23.val[1]);
            vst1q_f32(col1, im23.val[0]);
            vst1q_f32(col2, im23.val[1]);
            vst1q_f32(col3, im01.val[1]);
            vst1q_f32(col4, im01.val[0]);
            vst1q_f32(col5, im01.val[1]);
            vst1q_f32(col6, im23.val[1]);
            vst1q_f32(col7, im23.val[0]);
            vst1q_f32(col8, im23.val[1]);

            col0 += 4;
            col1 += 4;
            col2 += 4;
            col3 += 4;
            col4 += 4;
            col5 += 4;
            col6 += 4;
            col7 += 4;
            col8 += 4;

            im_tmp_data0 += 8;
            im_tmp_data2 += 8;
          }
          const float *im_tmp_data1 = im_tmp_data0 + 1;
          const float *im_tmp_data3 = im_tmp_data2 + 1;
          for (int i = 0; i < mk4; ++i) {
            *col0 = *im_tmp_data3;
            *col1 = *im_tmp_data2;
            *col2 = *im_tmp_data3;
            *col3 = *im_tmp_data1;
            *col4 = *im_tmp_data0;
            *col5 = *im_tmp_data1;
            *col6 = *im_tmp_data3;
            *col7 = *im_tmp_data2;
            *col8 = *im_tmp_data3;

            col0++;
            col1++;
            col2++;
            col3++;
            col4++;
            col5++;
            col6++;
            col7++;
            col8++;
            im_tmp_data0 += 2;
            im_tmp_data1 += 2;
            im_tmp_data2 += 2;
            im_tmp_data3 += 2;
          }
          im_tmp_data0 += (isize - fill);
          im_tmp_data2 += (isize - fill);
        }
        for (int i = 0; i < osize; ++i) {
          col_data[0 * oosize + i * osize] = 0.0;
          col_data[3 * oosize + i * osize] = 0.0;
          col_data[6 * oosize + i * osize] = 0.0;
          if (pad2) {
            col_data[2 * oosize + osize - 1 + i * osize] = 0.0;
            col_data[5 * oosize + osize - 1 + i * osize] = 0.0;
            col_data[8 * oosize + osize - 1 + i * osize] = 0.0;
          }
        }
        float32x4_t zero4;
        zero4 = vdupq_n_f32(0.0);
        auto col_z0 = col_data;
        auto col_z1 = col_data + oosize;
        auto col_z2 = col_data + 2 * oosize;
        auto col_z6 = col_data + 6 * oosize + osize * (osize - 1);
        auto col_z7 = col_data + 7 * oosize + osize * (osize - 1);
        auto col_z8 = col_data + 8 * oosize + osize * (osize - 1);

        for (int i = 0; i < nk4; ++i) {
          vst1q_f32(col_z0, zero4);
          vst1q_f32(col_z1, zero4);
          vst1q_f32(col_z2, zero4);
          if (pad2) {
            vst1q_f32(col_z6, zero4);
            vst1q_f32(col_z7, zero4);
            vst1q_f32(col_z8, zero4);
          }
          col_z0 += 4;
          col_z1 += 4;
          col_z2 += 4;
          col_z6 += 4;
          col_z7 += 4;
          col_z8 += 4;
        }

        for (int i = 0; i < mk4; ++i) {
          col_z0[i] = 0.0;
          col_z1[i] = 0.0;
          col_z2[i] = 0.0;
          if (pad2) {
            col_z6[i] = 0.0;
            col_z7[i] = 0.0;
            col_z8[i] = 0.0;
          }
        }

        col_data[1 * oosize + osize] = im_data[isize];
        for (int i = 1; i < osize; ++i) {
          col_data[3 * oosize + i] = im_data[(i - 1) * stride[0] + 1];
        }
        col_data[4 * oosize] = im_data[0];
        col_data[7 * oosize] = im_data[isize];

        col_data += 9 * oosize;
        im_data += isize * isize;
      }
    } else {
      for (int c = 0; c < channels_col; ++c) {
        int w_offset = c % filter_width;
        int h_offset = (c / filter_width) % filter_height;
        int c_im = c / (filter_width * filter_height);
        for (int h = 0; h < col_height; ++h) {
          int im_row_idx = h * stride[0] - padding[0] + h_offset * dilation[0];
          for (int w = 0; w < col_width; ++w) {
            int im_col_idx =
                w * stride[1] - padding[1] + w_offset * dilation[1];
            int col_idx = (c * col_height + h) * col_width + w;
            int im_idx =
                (im_row_idx + c_im * im_height) * im_width + im_col_idx;

            col_data[col_idx] = (im_row_idx < 0 || im_row_idx >= im_height ||
                                 im_col_idx < 0 || im_col_idx >= im_width)
                                    ? static_cast<T>(0)
                                    : im_data[im_idx];
          }
        }
      }
    }
#else
    for (int c = 0; c < channels_col; ++c) {
      int w_offset = c % filter_width;
      int h_offset = (c / filter_width) % filter_height;
      int c_im = c / (filter_width * filter_height);
      for (int h = 0; h < col_height; ++h) {
        int im_row_idx = h * stride[0] - padding[0] + h_offset * dilation[0];
        for (int w = 0; w < col_width; ++w) {
          int im_col_idx = w * stride[1] - padding[1] + w_offset * dilation[1];
          int col_idx = (c * col_height + h) * col_width + w;
          int im_idx = (im_row_idx + c_im * im_height) * im_width + im_col_idx;

          col_data[col_idx] = (im_row_idx < 0 || im_row_idx >= im_height ||
                               im_col_idx < 0 || im_col_idx >= im_width)
                                  ? static_cast<T>(0)
                                  : im_data[im_idx];
        }
      }
    }
#endif
  }
};

/*
 * im = [input_channels, input_height, input_width]
 * col =
 *   [input_channels, filter_height, filter_width, output_height,
 * output_width]
 */
template <class T>
class Col2ImFunctor<ColFormat::kCFO, CPU, T> {
 public:
  void operator()(const framework::Tensor &col,
                  const std::vector<int> &dilation,
                  const std::vector<int> &stride,
                  const std::vector<int> &padding, framework::Tensor *im) {
    //    PADDLE_ENFORCE(im->dims().size() == 3);
    //    PADDLE_ENFORCE(col.dims().size() == 5);
    int im_channels = im->dims()[0];
    int im_height = im->dims()[1];
    int im_width = im->dims()[2];
    int filter_height = col.dims()[1];
    int filter_width = col.dims()[2];
    int col_height = col.dims()[3];
    int col_width = col.dims()[4];

    //    PADDLE_ENFORCE_EQ((im_height + padding[0] + padding[2]
    //    -
    //                       ((dilation[0] * (filter_height - 1)
    //                       + 1))) /
    //                              stride[0] +
    //                          1,
    //                      col_height,
    //                      "Output_height and
    //                      padding(padding_up, padding_down)
    //                      are " "inconsistent.");
    //    PADDLE_ENFORCE_EQ((im_width + padding[1] + padding[3]
    //    -
    //                       ((dilation[1] * (filter_width - 1)
    //                       + 1))) /
    //                              stride[1] +
    //                          1,
    //                      col_width,
    //                      "Output_height and
    //                      padding(padding_up, padding_down)
    //                      are " "inconsistent.");

    int channels_col = im_channels * filter_height * filter_width;

    T *im_data = im->data<T>();
    const T *col_data = col.data<T>();

    for (int c = 0; c < channels_col; ++c) {
      int w_offset = c % filter_width;
      int h_offset = (c / filter_width) % filter_height;
      int c_im = c / (filter_width * filter_height);
      for (int h = 0; h < col_height; ++h) {
        int im_row_idx = h * stride[0] - padding[0] + h_offset * dilation[0];
        for (int w = 0; w < col_width; ++w) {
          int im_col_idx = w * stride[1] - padding[1] + w_offset * dilation[1];
          if ((im_row_idx) >= 0 && (im_row_idx) < im_height &&
              (im_col_idx) >= 0 && (im_col_idx) < im_width) {
            im_data[(im_row_idx + c_im * im_height) * im_width + im_col_idx] +=
                col_data[(c * col_height + h) * col_width + w];
          }
        }
      }
    }
  }
};

template class Im2ColFunctor<ColFormat::kCFO, CPU, float>;
// template class Im2ColFunctor<ColFormat::kCFO, CPU, double>;
template class Col2ImFunctor<ColFormat::kCFO, CPU, float>;
template class Col2ImFunctor<ColFormat::kCFO, CPU, double>;

/*
 * im = [input_channels, input_height, input_width]
 * col =
 *   [output_height, output_width, input_channels, filter_height,
 * filter_width]
 */
template <class T>
class Im2ColFunctor<ColFormat::kOCF, CPU, T> {
 public:
  void operator()(const framework::Tensor &im, const std::vector<int> &dilation,
                  const std::vector<int> &stride,
                  const std::vector<int> &padding, framework::Tensor *col) {
    //    PADDLE_ENFORCE(im.dims().size() == 3);
    //    PADDLE_ENFORCE(col->dims().size() == 5);
    int im_channels = im.dims()[0];
    int im_height = im.dims()[1];
    int im_width = im.dims()[2];
    int filter_height = col->dims()[3];
    int filter_width = col->dims()[4];
    int col_height = col->dims()[0];
    int col_width = col->dims()[1];

    //    PADDLE_ENFORCE_EQ(
    //        (im_height + padding[0] + padding[2] -
    //        filter_height) / stride[0]
    //        + 1, col_height, "Output_height and
    //        padding(padding_up,
    //        padding_down) are " "inconsistent.");
    //    PADDLE_ENFORCE_EQ(
    //        (im_width + padding[1] + padding[3] -
    //        filter_width) / stride[1] +
    //        1, col_width, "col_width and padding(padding_left,
    //        padding_right)
    //        are " "inconsistent.");

    const T *im_data = im.data<T>();
    T *col_data = col->data<T>();

    for (int col_row_idx = 0; col_row_idx < col_height; ++col_row_idx) {
      for (int col_col_idx = 0; col_col_idx < col_width; ++col_col_idx) {
        for (int channel = 0; channel < im_channels; ++channel) {
          for (int filter_row_idx = 0; filter_row_idx < filter_height;
               ++filter_row_idx) {
            int im_row_offset =
                col_row_idx * stride[0] + filter_row_idx - padding[0];
            for (int filter_col_idx = 0; filter_col_idx < filter_width;
                 ++filter_col_idx) {
              int im_col_offset =
                  col_col_idx * stride[1] + filter_col_idx - padding[1];

              int col_offset =
                  ((((col_row_idx)*col_width + col_col_idx) * im_channels +
                    channel) *
                       filter_height +
                   filter_row_idx) *
                      filter_width +
                  filter_col_idx;

              int im_offset = (channel * im_height + im_row_offset) * im_width +
                              im_col_offset;
              col_data[col_offset] =
                  (im_row_offset < 0 || im_row_offset >= im_height ||
                   im_col_offset < 0 || im_col_offset >= im_width)
                      ? static_cast<T>(0)
                      : im_data[im_offset];
            }
          }
        }
      }
    }
  }
};

/*
 * im = [input_channels, input_height, input_width]
 * col =
 *   [output_height, output_width, input_channels, filter_height,
 * filter_width]
 */
template <class T>
class Col2ImFunctor<ColFormat::kOCF, CPU, T> {
 public:
  void operator()(const framework::Tensor &col,
                  const std::vector<int> &dilation,
                  const std::vector<int> &stride,
                  const std::vector<int> &padding, framework::Tensor *im) {
    //    PADDLE_ENFORCE(im->dims().size() == 3);
    //    PADDLE_ENFORCE(col.dims().size() == 5);
    int im_channels = im->dims()[0];
    int im_height = im->dims()[1];
    int im_width = im->dims()[2];
    int filter_height = col.dims()[3];
    int filter_width = col.dims()[4];
    int col_height = col.dims()[0];
    int col_width = col.dims()[1];

    //    PADDLE_ENFORCE_EQ(
    //        (im_height + padding[0] + padding[2] -
    //        filter_height) / stride[0]
    //        + 1, col_height, "Output_height and
    //        padding(padding_up,
    //        padding_down) are " "inconsistent.");
    //    PADDLE_ENFORCE_EQ(
    //        (im_width + padding[1] + padding[3] -
    //        filter_width) / stride[1] +
    //        1, col_width, "col_width and padding(padding_left,
    //        padding_right)
    //        are " "inconsistent.");

    T *im_data = im->data<T>();
    const T *col_data = col.data<T>();

    for (int col_row_idx = 0; col_row_idx < col_height; ++col_row_idx) {
      for (int col_col_idx = 0; col_col_idx < col_width; ++col_col_idx) {
        for (int channel = 0; channel < im_channels; ++channel) {
          for (int filter_row_idx = 0; filter_row_idx < filter_height;
               ++filter_row_idx) {
            int im_row_offset =
                col_row_idx * stride[0] + filter_row_idx - padding[0];
            for (int filter_col_idx = 0; filter_col_idx < filter_width;
                 ++filter_col_idx) {
              int im_col_offset =
                  col_col_idx * stride[1] + filter_col_idx - padding[1];

              int col_offset =
                  (((col_row_idx * col_width + col_col_idx) * im_channels +
                    channel) *
                       filter_height +
                   filter_row_idx) *
                      filter_width +
                  filter_col_idx;

              if (im_row_offset >= 0 && im_row_offset < im_height &&
                  im_col_offset >= 0 && im_col_offset < im_width) {
                int im_offset =
                    (channel * im_height + im_row_offset) * im_width +
                    im_col_offset;
                im_data[im_offset] += col_data[col_offset];
              }
            }
          }
        }
      }
    }
  }
};

template class Im2ColFunctor<ColFormat::kOCF, CPU, float>;
template class Im2ColFunctor<ColFormat::kOCF, CPU, double>;
template class Col2ImFunctor<ColFormat::kOCF, CPU, float>;
template class Col2ImFunctor<ColFormat::kOCF, CPU, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile
