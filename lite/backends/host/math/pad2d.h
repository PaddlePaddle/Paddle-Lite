// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once
#include <algorithm>
#include <string>
#include <vector>

namespace paddle {
namespace lite {
namespace host {
namespace math {

template <class T>
void Pad2DReflectNCHW(const T* in_data,
                      const int num,
                      const int channels,
                      const int in_height,
                      const int in_width,
                      const int out_height,
                      const int out_width,
                      const int pad_top,
                      const int pad_left,
                      T* out_data) {
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int out_h = 0; out_h < out_height; ++out_h) {
        for (int out_w = 0; out_w < out_width; ++out_w) {
          int in_h = out_h - pad_top;
          int in_w = out_w - pad_left;
          in_h = std::max(in_h, -in_h);  // reflect by 0
          in_h =
              std::min(in_h, 2 * in_height - in_h - 2);  // reflect by in_height
          in_w = std::max(in_w, -in_w);                  // reflect by 0
          in_w =
              std::min(in_w, 2 * in_width - in_w - 2);  // reflect by in_width
          out_data[out_h * out_width + out_w] = in_data[in_h * in_width + in_w];
        }
      }
      in_data += in_height * in_width;
      out_data += out_height * out_width;
    }
  }
}

template <typename T>
void Pad2DEdgeNCHW(const T* in_data,
                   const int num,
                   const int channels,
                   const int in_height,
                   const int in_width,
                   const int out_height,
                   const int out_width,
                   const int pad_top,
                   const int pad_left,
                   T* out_data) {
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int out_h = 0; out_h < out_height; ++out_h) {
        for (int out_w = 0; out_w < out_width; ++out_w) {
          int in_h = std::min(in_height - 1, std::max(out_h - pad_top, 0));
          int in_w = std::min(in_width - 1, std::max(out_w - pad_left, 0));
          out_data[out_h * out_width + out_w] = in_data[in_h * in_width + in_w];
        }
      }
      in_data += in_height * in_width;
      out_data += out_height * out_width;
    }
  }
}

template <typename T>
void Pad2DConstNCHW(const T* in_data,
                    const int num,
                    const int channels,
                    const int in_height,
                    const int in_width,
                    const int out_height,
                    const int out_width,
                    const int pad_top,
                    const int pad_left,
                    T value,
                    T* out_data) {
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int out_h = 0; out_h < out_height; ++out_h) {
        for (int out_w = 0; out_w < out_width; ++out_w) {
          int in_h = out_h - pad_top;
          int in_w = out_w - pad_left;
          out_data[out_h * out_width + out_w] =
              (in_h < 0 || in_w < 0 || in_h >= in_height || in_w >= in_width)
                  ? value
                  : in_data[in_h * in_width + in_w];
        }
      }
      in_data += in_height * in_width;
      out_data += out_height * out_width;
    }
  }
}

template <typename T>
void Pad2DReflectNHWC(const T* in_data,
                      const int num,
                      const int channels,
                      const int in_height,
                      const int in_width,
                      const int out_height,
                      const int out_width,
                      const int pad_top,
                      const int pad_left,
                      T* out_data) {
  for (int n = 0; n < num; ++n) {
    for (int out_h = 0; out_h < out_height; ++out_h) {
      for (int out_w = 0; out_w < out_width; ++out_w) {
        const int out_index = (out_h * out_width + out_w) * channels;
        int in_h = out_h - pad_top;
        int in_w = out_w - pad_left;
        in_h = std::max(in_h, -in_h);
        in_h = std::min(in_h, 2 * in_height - in_h - 2);
        in_w = std::max(in_w, -in_w);
        in_w = std::min(in_w, 2 * in_width - in_w - 2);
        const int in_index = (in_h * in_width + in_w) * channels;

        for (int c = 0; c < channels; ++c) {
          out_data[out_index + c] = in_data[in_index + c];
        }
      }
    }
    in_data += in_height * in_width * channels;
    out_data += out_height * out_width * channels;
  }
}

template <typename T>
void Pad2DEdgeNHWC(const T* in_data,
                   const int num,
                   const int channels,
                   const int in_height,
                   const int in_width,
                   const int out_height,
                   const int out_width,
                   const int pad_top,
                   const int pad_left,
                   T* out_data) {
  for (int n = 0; n < num; ++n) {
    for (int out_h = 0; out_h < out_height; ++out_h) {
      for (int out_w = 0; out_w < out_width; ++out_w) {
        const int out_index = (out_h * out_width + out_w) * channels;
        int in_h = std::min(in_height - 1, std::max(out_h - pad_top, 0));
        int in_w = std::min(in_width - 1, std::max(out_w - pad_left, 0));
        const int in_index = (in_h * in_width + in_w) * channels;
        for (int c = 0; c < channels; ++c) {
          out_data[out_index + c] = in_data[in_index + c];
        }
      }
    }
    in_data += in_height * in_width * channels;
    out_data += out_height * out_width * channels;
  }
}

template <typename T>
void Pad2DConstNHWC(const T* in_data,
                    const int num,
                    const int channels,
                    const int in_height,
                    const int in_width,
                    const int out_height,
                    const int out_width,
                    const int pad_top,
                    const int pad_left,
                    T value,
                    T* out_data) {
  for (int n = 0; n < num; ++n) {
    for (int out_h = 0; out_h < out_height; ++out_h) {
      for (int out_w = 0; out_w < out_width; ++out_w) {
        int in_h = out_h - pad_top;
        int in_w = out_w - pad_left;
        const int out_index = (out_h * out_width + out_w) * channels;
        if (in_h < 0 || in_w < 0 || in_h >= in_height || in_w >= in_width) {
          for (int c = 0; c < channels; ++c) {
            out_data[out_index + c] = value;
          }
        } else {
          const int in_index = (in_h * in_width + in_w) * channels;
          for (int c = 0; c < channels; ++c) {
            out_data[out_index + c] = in_data[in_index + c];
          }
        }
      }
    }
    in_data += in_height * in_width * channels;
    out_data += out_height * out_width * channels;
  }
}

}  // namespace math
}  // namespace host
}  // namespace lite
}  // namespace paddle
