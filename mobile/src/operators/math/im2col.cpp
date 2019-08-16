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

#include <vector>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#include <algorithm>
#include "common/types.h"
#include "operators/math/im2col.h"

namespace paddle_mobile {
namespace operators {
namespace math {

template <>
void ExtractToImg<float>(const float *im_data, float *col_data,
                         const int im_height, const int im_width,
                         const int col_height, const int col_width,
                         const int padding_h, const int padding_w,
                         const int stride_h, const int stride_w, const int kh,
                         const int kw) {
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
    int s = 0;
    if (stride_w == 1) {
#if __ARM_NEON
      for (; s < extract - 3; s += 4) {
        float32x4_t _img = vld1q_f32(im_data + s);
        vst1q_f32(col_data + s, _img);
      }
#endif
      for (; s < extract; ++s) {
        col_data[s] = im_data[s];
      }
    } else if (stride_w == 2) {
#if __ARM_NEON
      for (; s < extract - 3; s += 4) {
        float32x4x2_t _img = vld2q_f32(im_data + s * 2);
        vst1q_f32(col_data + s, _img.val[0]);
      }
#endif
      for (; s < extract; ++s) {
        col_data[s] = im_data[s * 2];
      }
    } else if (stride_w == 3) {
#if __ARM_NEON
      for (; s < extract - 3; s += 4) {
        float32x4x3_t _img = vld3q_f32(im_data + s * 3);
        vst1q_f32(col_data + s, _img.val[0]);
      }
#endif
      for (; s < extract; ++s) {
        col_data[s] = im_data[s * 3];
      }
    } else if (stride_w == 4) {
#if __ARM_NEON
      for (; s < extract - 3; s += 4) {
        float32x4x4_t _img = vld4q_f32(im_data + s * 4);
        vst1q_f32(col_data + s, _img.val[0]);
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

template <>
void ExtractToImg<int8_t>(const int8_t *im_data, int8_t *col_data,
                          const int im_height, const int im_width,
                          const int col_height, const int col_width,
                          const int padding_h, const int padding_w,
                          const int stride_h, const int stride_w, const int kh,
                          const int kw) {
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
    int s = 0;
    if (stride_w == 1) {
      for (; s < extract - 15; s += 16) {
        int8x16_t _img = vld1q_s8(im_data + s);
        vst1q_s8(col_data + s, _img);
      }
      for (; s < extract; ++s) {
        col_data[s] = im_data[s];
      }
    } else if (stride_w == 2) {
#if __ARM_NEON
      for (; s < extract - 15; s += 16) {
        int8x16x2_t _img = vld2q_s8(im_data + s * 2);
        vst1q_s8(col_data + s, _img.val[0]);
      }
#endif
      for (; s < extract; ++s) {
        col_data[s] = im_data[s * 2];
      }
    } else if (stride_w == 3) {
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
template <class T>
class Im2ColFunctor<ColFormat::kCFO, CPU, T> {
 public:
  void operator()(const framework::Tensor &im, const std::vector<int> &dilation,
                  const std::vector<int> &stride,
                  const std::vector<int> &padding, framework::Tensor *col) {
    int im_channels = im.dims()[0];
    int im_height = im.dims()[1];
    int im_width = im.dims()[2];
    int filter_height = col->dims()[1];
    int filter_width = col->dims()[2];
    int col_height = col->dims()[3];
    int col_width = col->dims()[4];

    int channels_col = im_channels * filter_height * filter_width;
    const T *im_data = im.data<T>();
    T *col_data = col->data<T>();
#if __ARM_NEON
    if (stride[0] <= 4 && dilation[0] == 1 && dilation[0] == dilation[1]) {
      int im_spatial_size = im_height * im_width;
      int col_spatial_size = col_height * col_width;
      // pad 0
      memset(col_data, 0, col->numel() * sizeof(T));

      #pragma omp parallel for
      for (int ic = 0; ic < im_channels; ++ic) {
        const T *local_im_data = im_data + ic * im_spatial_size;
        T *local_col_data =
            col_data + ic * filter_height * filter_width * col_spatial_size;
        for (int kh = 0; kh < filter_height; ++kh) {
          for (int kw = 0; kw < filter_width; ++kw) {
            ExtractToImg<T>(local_im_data, local_col_data, im_height, im_width,
                            col_height, col_width, padding[0], padding[1],
                            stride[0], stride[1], kh, kw);
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
#if __ARM_NEON
    }
#endif
  }
};

template <>
void ExtendToImg<float>(const float *col_data, float *im_data,
                        const int im_height, const int im_width,
                        const int col_height, const int col_width,
                        const int padding_h, const int padding_w,
                        const int stride_h, const int stride_w, const int kh,
                        const int kw) {
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
  // int extract = (end_width - start_width + stride_w - 1) / stride_w;
  int extend = end_width - start_width;

  im_data += start_height * im_width + start_width;
  col_data += col_start_height * col_width + col_start_width;

  for (int i = start_height; i < end_height; i += stride_h) {
    int s = 0;
    if (stride_w == 1) {
#if __ARM_NEON
      for (; s < extend - 3; s += 4) {
        float32x4_t _col = vld1q_f32(col_data + s);
        float32x4_t _img = vld1q_f32(im_data + s);
        _img = vaddq_f32(_img, _col);
        vst1q_f32(im_data + s, _img);
      }
#endif
      for (; s < extend; ++s) {
        im_data[s] += col_data[s];
      }
    } else if (stride_w == 2) {
#if __ARM_NEON
      for (; s < extend - 7; s += 8) {
        float32x4_t _col = vld1q_f32(col_data + s / 2);
        float32x4x2_t _img = vld2q_f32(im_data + s);
        _img.val[0] = vaddq_f32(_img.val[0], _col);
        vst2q_f32(im_data + s, _img);
      }
#endif
      for (; s < extend; s += 2) {
        im_data[s] += col_data[s / 2];
      }
    } else {
      PADDLE_MOBILE_THROW_EXCEPTION("stride_w must be one of 1 and 2.");
    }
    im_data += im_width * stride_h;
    col_data += col_width;
  }
}

template <>
void ExtendToImgV2<float>(const float *col_data, float *im_data,
                          const int im_height, const int im_width,
                          const int col_height, const int col_width,
                          const int padding_h, const int padding_w,
                          const int stride_h, const int stride_w, const int kh,
                          const int kernel_w) {
  int col_spatial_size = col_height * col_width;
  int h = padding_h - kh;
  int col_start_height = h > 0 ? (h + stride_h - 1) / stride_h : 0;
  int start_height = kh + col_start_height * stride_h - padding_h;
  int end_height = (col_height - col_start_height) * stride_h + start_height;
  end_height = end_height > im_height ? im_height : end_height;
  im_data += start_height * im_width;
  col_data += col_start_height * col_width;

  int kw = 0;
  for (; kw < kernel_w - 1; kw += 2) {
    int w0 = padding_w - kw;
    int w1 = padding_w - (kw + 1);
    int col_start_width0 = w0 > 0 ? (w0 + stride_w - 1) / stride_w : 0;
    int col_start_width1 = w1 > 0 ? (w1 + stride_w - 1) / stride_w : 0;
    int start_width0 = kw + col_start_width0 * stride_w - padding_w;
    int start_width1 = (kw + 1) + col_start_width1 * stride_w - padding_w;

    int end_width0 = (col_width - col_start_width0) * stride_w + start_width0;
    end_width0 = end_width0 > im_width ? im_width : end_width0;
    int end_width1 = (col_width - col_start_width1) * stride_w + start_width1;
    end_width1 = end_width1 > im_width ? im_width : end_width1;
    int start_width = 0;
    int end_width = 0;
    if (stride_w == 1) {
      start_width = std::max(start_width0, start_width1);
      end_width = std::min(end_width0, end_width1);
    } else if (stride_w == 2) {
      start_width = std::min(start_width0, start_width1);
      end_width = std::min(end_width0, end_width1);
    } else {
      PADDLE_MOBILE_THROW_EXCEPTION("stride_w must be one of 1 and 2.");
    }

    //    DLOG << "start_width0: " << start_width0 << ", end_width0: " <<
    //    end_width0; DLOG << "start_width1: " << start_width1 << ", end_width1:
    //    " << end_width1;
    int extend = end_width - start_width;
    float *im_data01 = im_data + start_width;
    float *im_data0 = im_data + start_width0;
    float *im_data1 = im_data + start_width1;
    const float *col_data0 = col_data + col_start_width0;
    const float *col_data1 = col_data + col_spatial_size + col_start_width1;

    for (int i = start_height; i < end_height; i += stride_h) {
      int s = 0;
      if (stride_w == 1) {
        int offset0 = start_width - start_width0;
        int offset1 = start_width - start_width1;
        for (int ss = 0; ss < start_width - start_width0; ++ss) {
          im_data0[ss] += col_data0[ss];
        }
        for (int ss = 0; ss < start_width - start_width1; ++ss) {
          im_data1[ss] += col_data1[ss];
        }
#if __ARM_NEON
        for (; s < extend - 3; s += 4) {
          float32x4_t _col0 = vld1q_f32(col_data0 + offset0 + s);
          float32x4_t _col1 = vld1q_f32(col_data1 + offset1 + s);
          float32x4_t _img = vld1q_f32(im_data01 + s);
          _img = vaddq_f32(_img, _col0);
          _img = vaddq_f32(_img, _col1);
          vst1q_f32(im_data01 + s, _img);
        }
#endif
        for (int ss = s; ss < end_width0 - start_width0; ++ss) {
          im_data0[ss] += col_data0[ss];
        }
        for (int ss = s; ss < end_width1 - start_width1; ++ss) {
          im_data1[ss] += col_data1[ss];
        }
      } else if (stride_w == 2) {
        if (start_width0 < start_width1) {
#if __ARM_NEON
          for (; s < extend - 7; s += 8) {
            float32x4_t _col0 = vld1q_f32(col_data0 + s / 2);
            float32x4_t _col1 = vld1q_f32(col_data1 + s / 2);
            float32x4x2_t _img = vld2q_f32(im_data01 + s);
            _img.val[0] = vaddq_f32(_img.val[0], _col0);
            _img.val[1] = vaddq_f32(_img.val[1], _col1);
            vst2q_f32(im_data01 + s, _img);
          }
#endif
        } else {
#if __ARM_NEON
          for (; s < extend - 7; s += 8) {
            float32x4_t _col0 = vld1q_f32(col_data0 + s / 2);
            float32x4_t _col1 = vld1q_f32(col_data1 + s / 2);
            float32x4x2_t _img = vld2q_f32(im_data01 + s);
            _img.val[0] = vaddq_f32(_img.val[0], _col1);
            _img.val[1] = vaddq_f32(_img.val[1], _col0);
            vst2q_f32(im_data01 + s, _img);
          }
#endif
        }
        for (int ss = s; ss < end_width0 - start_width0; ss += 2) {
          im_data0[ss] += col_data0[ss / 2];
        }
        for (int ss = s; ss < end_width1 - start_width1; ss += 2) {
          im_data1[ss] += col_data1[ss / 2];
        }
      }

      im_data0 += im_width * stride_h;
      im_data1 += im_width * stride_h;
      im_data01 += im_width * stride_h;
      col_data0 += col_width;
      col_data1 += col_width;
    }
    col_data += 2 * col_spatial_size;
  }

  for (; kw < kernel_w; ++kw) {
    int w = padding_w - kw;
    int col_start_width = w > 0 ? (w + stride_w - 1) / stride_w : 0;
    int start_width = kw + col_start_width * stride_w - padding_w;

    int end_width = (col_width - col_start_width) * stride_w + start_width;
    end_width = end_width > im_width ? im_width : end_width;
    int extend = end_width - start_width;

    float *im_data0 = im_data + start_width;
    const float *col_data0 = col_data + col_start_width;

    for (int i = start_height; i < end_height; i += stride_h) {
      int s = 0;
      if (stride_w == 1) {
#if __ARM_NEON
        for (; s < extend - 3; s += 4) {
          float32x4_t _col = vld1q_f32(col_data + s);
          float32x4_t _img = vld1q_f32(im_data + s);
          _img = vaddq_f32(_img, _col);
          vst1q_f32(im_data + s, _img);
        }
#endif
        for (; s < extend; ++s) {
          im_data[s] += col_data[s];
        }
      } else if (stride_w == 2) {
#if __ARM_NEON
        for (; s < extend - 7; s += 8) {
          float32x4_t _col = vld1q_f32(col_data + s / 2);
          float32x4x2_t _img = vld2q_f32(im_data + s);
          _img.val[0] = vaddq_f32(_img.val[0], _col);
          vst2q_f32(im_data + s, _img);
        }
#endif
        for (; s < extend; s += 2) {
          im_data[s] += col_data[s / 2];
        }
      } else {
        PADDLE_MOBILE_THROW_EXCEPTION("stride_w must be one of 1 and 2.");
      }
      im_data += im_width * stride_h;
      col_data += col_width;
    }
    col_data += col_spatial_size;
  }
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
    int im_channels = im->dims()[0];
    int im_height = im->dims()[1];
    int im_width = im->dims()[2];
    int filter_height = col.dims()[1];
    int filter_width = col.dims()[2];
    int col_height = col.dims()[3];
    int col_width = col.dims()[4];

    int channels_col = im_channels * filter_height * filter_width;
    const T *col_data = col.data<T>();
    T *im_data = im->data<T>();
    memset(static_cast<void *>(im_data), 0, sizeof(T) * im->numel());

#if __ARM_NEON
    if (stride[0] <= 2 && dilation[0] == 1 && dilation[0] == dilation[1]) {
      int im_spatial_size = im_height * im_width;
      int col_spatial_size = col_height * col_width;

      #pragma omp parallel for
      for (int ic = 0; ic < im_channels; ++ic) {
        T *local_im_data = im_data + ic * im_spatial_size;
        const T *local_col_data =
            col_data + ic * filter_height * filter_width * col_spatial_size;
        for (int kh = 0; kh < filter_height; ++kh) {
#if 0
          for (int kw = 0; kw < filter_width; ++kw) {
            ExtendToImg<T>(local_col_data, local_im_data, im_height, im_width,
                           col_height, col_width, padding[0], padding[1],
                           stride[0], stride[1], kh, kw);
            local_col_data += col_spatial_size;
          }
#else
          ExtendToImgV2<T>(local_col_data, local_im_data, im_height, im_width,
                           col_height, col_width, padding[0], padding[1],
                           stride[0], stride[1], kh, filter_width);
          local_col_data += col_spatial_size * filter_width;
#endif
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
            int im_col_idx =
                w * stride[1] - padding[1] + w_offset * dilation[1];
            if ((im_row_idx) >= 0 && (im_row_idx) < im_height &&
                (im_col_idx) >= 0 && (im_col_idx) < im_width) {
              im_data[(im_row_idx + c_im * im_height) * im_width +
                      im_col_idx] +=
                  col_data[(c * col_height + h) * col_width + w];
            }
          }
        }
      }
#if __ARM_NEON
    }
#endif
  }
};

template class Im2ColFunctor<ColFormat::kCFO, CPU, float>;
template class Im2ColFunctor<ColFormat::kCFO, CPU, int8_t>;
template class Col2ImFunctor<ColFormat::kCFO, CPU, float>;
// template class Col2ImFunctor<ColFormat::kCFO, CPU, int8_t>;

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
