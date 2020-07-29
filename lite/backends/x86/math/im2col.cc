/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "lite/backends/x86/math/im2col.h"
#include <vector>
#include "lite/backends/x86/math/im2col_cfo_cpu.h"
#include "lite/utils/cp_logging.h"

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

/*
 * im = [input_channels, input_height, input_width]
 * col =
 *   [input_channels, filter_height, filter_width, output_height, output_width]
 */
template <class T>
class Im2ColFunctor<lite::x86::math::ColFormat::kCFO,
                    lite::TargetType::kX86,
                    T> {
 public:
  void operator()(const lite::X86Context& context,
                  const lite::Tensor& im,
                  const std::vector<int>& dilation,
                  const std::vector<int>& stride,
                  const std::vector<int>& padding,
                  lite::Tensor* col) {
    CHECK_EQ(im.dims().size(), 3);
    CHECK_EQ(col->dims().size(), 5);

    if (stride[0] == 1 && stride[1] == 1 && dilation[0] == 1 &&
        dilation[1] == 1) {
      if (padding[0] == 0 && padding[1] == 0) {
        im2col_sh1sw1dh1dw1ph0pw0<T>(im, col);
        return;
      } else if (padding[0] == 1 && padding[1] == 1) {
        im2col_sh1sw1dh1dw1ph1pw1<T>(im, col);
        return;
      }
      // TODO(TJ): complete padding >=2
    }
    im2col_common<T>(im, dilation, stride, padding, col);
  }
};

/*
 * im = [input_channels, input_height, input_width]
 * col =
 *   [input_channels, filter_height, filter_width, output_height, output_width]
 */
template <class T>
class Col2ImFunctor<lite::x86::math::ColFormat::kCFO,
                    lite::TargetType::kX86,
                    T> {
 public:
  void operator()(const lite::X86Context& context,
                  const lite::Tensor& col,
                  const std::vector<int>& dilation,
                  const std::vector<int>& stride,
                  const std::vector<int>& padding,
                  lite::Tensor* im) {
    CHECK_EQ(im->dims().size(), 3);
    CHECK_EQ(col.dims().size(), 5);
    int im_channels = im->dims()[0];
    int im_height = im->dims()[1];
    int im_width = im->dims()[2];
    int filter_height = col.dims()[1];
    int filter_width = col.dims()[2];
    int col_height = col.dims()[3];
    int col_width = col.dims()[4];

    CHECK_EQ((im_height + padding[0] + padding[2] -
              ((dilation[0] * (filter_height - 1) + 1))) /
                     stride[0] +
                 1,
             col_height)
        << "Output_height and padding(padding_up, padding_down) are "
           "inconsistent.";
    CHECK_EQ((im_width + padding[1] + padding[3] -
              ((dilation[1] * (filter_width - 1) + 1))) /
                     stride[1] +
                 1,
             col_width)
        << "Output_height and padding(padding_up, padding_down) are "
           "inconsistent.";

    int channels_col = im_channels * filter_height * filter_width;

    T* im_data = im->template mutable_data<T>();
    const T* col_data = col.data<T>();

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

template class Im2ColFunctor<lite::x86::math::ColFormat::kCFO,
                             lite::TargetType::kX86,
                             float>;
template class Im2ColFunctor<lite::x86::math::ColFormat::kCFO,
                             lite::TargetType::kX86,
                             double>;
template class Col2ImFunctor<lite::x86::math::ColFormat::kCFO,
                             lite::TargetType::kX86,
                             float>;
template class Col2ImFunctor<lite::x86::math::ColFormat::kCFO,
                             lite::TargetType::kX86,
                             double>;

/*
 * im = [input_channels, input_height, input_width]
 * col =
 *   [output_height, output_width, input_channels, filter_height, filter_width]
 */
template <class T>
class Im2ColFunctor<lite::x86::math::ColFormat::kOCF,
                    lite::TargetType::kX86,
                    T> {
 public:
  void operator()(const lite::X86Context& context,
                  const lite::Tensor& im,
                  const std::vector<int>& dilation,
                  const std::vector<int>& stride,
                  const std::vector<int>& padding,
                  lite::Tensor* col) {
    CHECK_EQ(im.dims().size(), 3);
    CHECK_EQ(col->dims().size(), 5);
    int im_channels = im.dims()[0];
    int im_height = im.dims()[1];
    int im_width = im.dims()[2];
    int filter_height = col->dims()[3];
    int filter_width = col->dims()[4];
    int col_height = col->dims()[0];
    int col_width = col->dims()[1];

    const T* im_data = im.data<T>();
    T* col_data = col->template mutable_data<T>();

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
 *   [output_height, output_width, input_channels, filter_height, filter_width]
 */
template <class T>
class Col2ImFunctor<lite::x86::math::ColFormat::kOCF,
                    lite::TargetType::kX86,
                    T> {
 public:
  void operator()(const lite::X86Context& context,
                  const lite::Tensor& col,
                  const std::vector<int>& dilation,
                  const std::vector<int>& stride,
                  const std::vector<int>& padding,
                  lite::Tensor* im) {
    CHECK_EQ(im->dims().size(), 3);
    CHECK_EQ(col.dims().size(), 5);
    int im_channels = im->dims()[0];
    int im_height = im->dims()[1];
    int im_width = im->dims()[2];
    int filter_height = col.dims()[3];
    int filter_width = col.dims()[4];
    int col_height = col.dims()[0];
    int col_width = col.dims()[1];

    CHECK_EQ(
        (im_height + padding[0] + padding[2] - filter_height) / stride[0] + 1,
        col_height)
        << "Output_height and padding(padding_up, padding_down) are "
           "inconsistent.";
    CHECK_EQ(
        (im_width + padding[1] + padding[3] - filter_width) / stride[1] + 1,
        col_width)
        << "col_width and padding(padding_left, padding_right) are "
           "inconsistent.";

    T* im_data = im->template mutable_data<T>();
    const T* col_data = col.data<T>();

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

template class Im2ColFunctor<lite::x86::math::ColFormat::kOCF,
                             lite::TargetType::kX86,
                             float>;
template class Im2ColFunctor<lite::x86::math::ColFormat::kOCF,
                             lite::TargetType::kX86,
                             double>;
template class Col2ImFunctor<lite::x86::math::ColFormat::kOCF,
                             lite::TargetType::kX86,
                             float>;
template class Col2ImFunctor<lite::x86::math::ColFormat::kOCF,
                             lite::TargetType::kX86,
                             double>;

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
