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
#include "common/types.h"
#include "framework/context.h"
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace paddle_mobile {
namespace operators {
namespace math {

void ExtractToImg(const float *im_data, float *col_data, const int im_height,
                  const int im_width, const int col_height, const int col_width,
                  const int padding_h, const int padding_w, const int stride_h,
                  const int stride_w, const int kh, const int kw) {
  int h = padding_h - kh;
  int w = padding_w - kw;
  int col_start_height = h > 0 ? (h + stride_h - 1) / stride_h : 0;
  int col_start_width = w > 0 ? (w + stride_w - 1) / stride_w : 0;
  int start_height = kh + col_start_height * stride_h - padding_h;
  int start_width = kw + col_start_width * stride_w - padding_w;

  int end_height = (col_height - col_start_height) * stride_h + start_height;
  end_height = end_height > im_height ? im_height : end_height;
  int end_width = (col_width - col_start_width) * stride_w + start_width;
  end_width = end_width > im_width ? im_width : end_width;
  int extract = (end_width - start_width + stride_w - 1) / stride_w;

  im_data += start_height * im_width + start_width;
  col_data += col_start_height * col_width + col_start_width;

  for (int i = start_height; i < end_height; i += stride_h) {
    if (stride_w == 1) {
      // memcpy(col_data, im_data, extract * sizeof(float));
      int s = 0;
#if __ARM_NEON
      for (; s < extract - 3; s += 4) {
        float32x4_t img = vld1q_f32(im_data + s);
        vst1q_f32(col_data + s, img);
      }
#endif
      for (; s < extract; ++s) {
        col_data[s] = im_data[s];
      }
    } else if (stride_w == 2) {
      int s = 0;
#if __ARM_NEON
      for (; s < extract - 3; s += 4) {
        float32x4x2_t img = vld2q_f32(im_data + s * 2);
        vst1q_f32(col_data + s, img.val[0]);
      }
#endif
      for (; s < extract; ++s) {
        col_data[s] = im_data[s * 2];
      }
    } else if (stride_w == 3) {
      int s = 0;
#if __ARM_NEON
      for (; s < extract - 3; s += 4) {
        float32x4x3_t img = vld3q_f32(im_data + s * 3);
        vst1q_f32(col_data + s, img.val[0]);
      }
#endif
      for (; s < extract; ++s) {
        col_data[s] = im_data[s * 3];
      }
    } else if (stride_w == 4) {
      int s = 0;
#if __ARM_NEON
      for (; s < extract - 3; s += 4) {
        float32x4x4_t img = vld4q_f32(im_data + s * 4);
        vst1q_f32(col_data + s, img.val[0]);
      }
#endif
      for (; s < extract; ++s) {
        col_data[s] = im_data[s * 4];
      }
    } else {
      PADDLE_MOBILE_THROW_EXCEPTION("stride_w must be one of 1, 2, 3 and 4.");
    }
    im_data += im_width * stride_h;
    col_data += col_width;
  }
}

/*
 * im = [input_channels, input_height, input_width]
 * col =
 *   [input_channels, filter_height, filter_width, output_height,
 * output_width]
 */
template <>
void Im2ColFunctor<ColFormat::kCFO, CPU, float>::operator()(
    const framework::Tensor &im, const std::vector<int> &dilation,
    const std::vector<int> &stride, const std::vector<int> &padding,
    framework::Tensor *col) {
  int im_channels = im.dims()[0];
  int im_height = im.dims()[1];
  int im_width = im.dims()[2];
  int filter_height = col->dims()[1];
  int filter_width = col->dims()[2];
  int col_height = col->dims()[3];
  int col_width = col->dims()[4];

  int channels_col = im_channels * filter_height * filter_width;
  const float *im_data = im.data<float>();
  float *col_data = col->data<float>();
#if __ARM_NEON
  if (stride[0] <= 4 && dilation[0] == 1 && dilation[0] == dilation[1]) {
    int im_spatial_size = im_height * im_width;
    int col_spatial_size = col_height * col_width;
    // pad 0
    memset(col_data, 0, col->numel() * sizeof(float));
    #pragma omp parallel for
    // num_threads(framework::threads())
    for (int ic = 0; ic < im_channels; ++ic) {
      const float *local_im_data = im_data + ic * im_spatial_size;
      float *local_col_data =
          col_data + ic * filter_height * filter_width * col_spatial_size;
      for (int kh = 0; kh < filter_height; ++kh) {
        for (int kw = 0; kw < filter_width; ++kw) {
          ExtractToImg(local_im_data, local_col_data, im_height, im_width,
                       col_height, col_width, padding[0], padding[1], stride[0],
                       stride[1], kh, kw);
          local_col_data += col_spatial_size;
        }
      }
    }
  } else {
#endif
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
                                  ? static_cast<float>(0)
                                  : im_data[im_idx];
        }
      }
    }
#if __ARM_NEON
  }
#endif
}

void ExtractToImg(const int8_t *im_data, int8_t *col_data, const int im_height,
                  const int im_width, const int col_height, const int col_width,
                  const int padding_h, const int padding_w, const int stride_h,
                  const int stride_w, const int kh, const int kw) {
  int h = padding_h - kh;
  int w = padding_w - kw;
  int col_start_height = h > 0 ? (h + stride_h - 1) / stride_h : 0;
  int col_start_width = w > 0 ? (w + stride_w - 1) / stride_w : 0;
  int start_height = kh + col_start_height * stride_h - padding_h;
  int start_width = kw + col_start_width * stride_w - padding_w;

  int end_height = (col_height - col_start_height) * stride_h + start_height;
  end_height = end_height > im_height ? im_height : end_height;
  int end_width = (col_width - col_start_width) * stride_w + start_width;
  end_width = end_width > im_width ? im_width : end_width;
  int extract = (end_width - start_width + stride_w - 1) / stride_w;

  im_data += start_height * im_width + start_width;
  col_data += col_start_height * col_width + col_start_width;
  for (int i = start_height; i < end_height; i += stride_h) {
    if (stride_w == 1) {
      memcpy(col_data, im_data, extract * sizeof(int8_t));
    } else if (stride_w == 2) {
      int s = 0;
#if __ARM_NEON
      for (; s < extract - 15; s += 16) {
        int8x16x2_t img = vld2q_s8(im_data + s * 2);
        vst1q_s8(col_data + s, img.val[0]);
      }
#endif
      for (; s < extract; ++s) {
        col_data[s] = im_data[s * 2];
      }
    } else if (stride_w == 3) {
      int s = 0;
#if __ARM_NEON
      for (; s < extract - 15; s += 16) {
        int8x16x3_t img = vld3q_s8(im_data + s * 3);
        vst1q_s8(col_data + s, img.val[0]);
      }
#endif
      for (; s < extract; ++s) {
        col_data[s] = im_data[s * 3];
      }
    } else if (stride_w == 4) {
      int s = 0;
#if __ARM_NEON
      for (; s < extract - 15; s += 16) {
        int8x16x4_t img = vld4q_s8(im_data + s * 4);
        vst1q_s8(col_data + s, img.val[0]);
      }
#endif
      for (; s < extract; ++s) {
        col_data[s] = im_data[s * 4];
      }
    } else {
      PADDLE_MOBILE_THROW_EXCEPTION("stride_w must be one of 1, 2, 3 and 4.");
    }
    im_data += im_width * stride_h;
    col_data += col_width;
  }
}

/*
 * im = [input_channels, input_height, input_width]
 * col =
 *   [input_channels, filter_height, filter_width, output_height,
 * output_width]
 */
template <>
void Im2ColFunctor<ColFormat::kCFO, CPU, int8_t>::operator()(
    const framework::Tensor &im, const std::vector<int> &dilation,
    const std::vector<int> &stride, const std::vector<int> &padding,
    framework::Tensor *col) {
  int im_channels = im.dims()[0];
  int im_height = im.dims()[1];
  int im_width = im.dims()[2];
  int filter_height = col->dims()[1];
  int filter_width = col->dims()[2];
  int col_height = col->dims()[3];
  int col_width = col->dims()[4];

  int channels_col = im_channels * filter_height * filter_width;
  const int8_t *im_data = im.data<int8_t>();
  int8_t *col_data = col->mutable_data<int8_t>();
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
  if (stride[0] <= 4 && dilation[0] == 1 && dilation[0] == dilation[1]) {
    int im_spatial_size = im_height * im_width;
    int col_spatial_size = col_height * col_width;
    // pad 0
    memset(col_data, 0, col->numel() * sizeof(int8_t));
    #pragma omp parallel for
    // num_threads(framework::threads())
    for (int ic = 0; ic < im_channels; ++ic) {
      const int8_t *local_im_data = im_data + ic * im_spatial_size;
      int8_t *local_col_data =
          col_data + ic * filter_height * filter_width * col_spatial_size;
      for (int kh = 0; kh < filter_height; ++kh) {
        for (int kw = 0; kw < filter_width; ++kw) {
          ExtractToImg(local_im_data, local_col_data, im_height, im_width,
                       col_height, col_width, padding[0], padding[1], stride[0],
                       stride[1], kh, kw);
          local_col_data += col_spatial_size;
        }
      }
    }
  } else {
#endif
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
                                  ? static_cast<int8_t>(0)
                                  : im_data[im_idx];
        }
      }
    }
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
  }
#endif
}

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

    int channels_col = im_channels * filter_height * filter_width;

    T *im_data = im->data<T>();
    const T *col_data = col.data<T>();
    memset(static_cast<void *>(im_data), 0, sizeof(T) * im->numel());

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
template class Im2ColFunctor<ColFormat::kCFO, CPU, int8_t>;
template class Col2ImFunctor<ColFormat::kCFO, CPU, float>;
template class Col2ImFunctor<ColFormat::kCFO, CPU, int8_t>;

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
    int im_channels = im.dims()[0];
    int im_height = im.dims()[1];
    int im_width = im.dims()[2];
    int filter_height = col->dims()[3];
    int filter_width = col->dims()[4];
    int col_height = col->dims()[0];
    int col_width = col->dims()[1];

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
    int im_channels = im->dims()[0];
    int im_height = im->dims()[1];
    int im_width = im->dims()[2];
    int filter_height = col.dims()[3];
    int filter_width = col.dims()[4];
    int col_height = col.dims()[0];
    int col_width = col.dims()[1];

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
template class Col2ImFunctor<ColFormat::kOCF, CPU, float>;

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile
