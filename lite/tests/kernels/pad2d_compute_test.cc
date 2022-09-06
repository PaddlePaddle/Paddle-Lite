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

#include <gtest/gtest.h>
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/core/test/arena/framework.h"
#include "lite/tests/utils/fill_data.h"

namespace paddle {
namespace lite {

template <class T = float>
class Pad2dComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string x_ = "X";
  std::string out_ = "Out";
  DDim dims_{{1, 1, 14, 14}};
  std::string mode_{"constant"};
  std::vector<int> paddings_;

  float pad_value_ = 0.f;
  std::string data_format_{"NCHW"};

 public:
  Pad2dComputeTester(const Place& place,
                     const std::string& alias,
                     std::string mode,
                     std::vector<int> paddings,
                     float pad_value,
                     std::string data_format)
      : TestCase(place, alias),
        mode_(mode),
        paddings_(paddings),
        pad_value_(pad_value),
        data_format_(data_format) {}

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(out_);
    CHECK(out);
    int out_h = dims_[2] + paddings_[0] + paddings_[1];
    int out_w = dims_[3] + paddings_[2] + paddings_[3];
    out->Resize(lite::DDim({dims_[0], dims_[1], out_h, out_w}));
    auto* out_data = out->template mutable_data<T>();
    auto* x = scope->FindTensor(x_);
    const auto* x_data = x->template data<T>();

    auto output_dims = out->dims();
    int n = output_dims[0];
    int c = output_dims[1];
    int h = output_dims[2];
    int w = output_dims[3];

    int pad_top = paddings_[0];
    int pad_bottom = paddings_[1];
    int pad_left = paddings_[2];
    int pad_right = paddings_[3];
    int pad_mode;
    if (mode_ == "constant") {
      pad_mode = 0;
    } else if (mode_ == "reflect") {
      pad_mode = 1;
    } else if (mode_ == "edge") {
      pad_mode = 2;
    } else {
      LOG(FATAL) << "Unknown mode type";
    }
    float pad_value = pad_value_;

    int in_w = w - pad_left - pad_right;
    int in_h = h - pad_bottom - pad_top;
    int spatial_size_out = w * h;
    int spatial_size_in = in_w * in_h;
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
    for (int i = 0; i < n * c; ++i) {
      const T* din_batch = x_data + i * spatial_size_in;
      T* dout_batch = out_data + i * spatial_size_out;
      int in_y = 0;
      int in_x = 0;
      for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
          switch (pad_mode) {
            case 0:
              in_y = y - pad_top;
              in_x = x - pad_left;
              dout_batch[y * w + x] =
                  (in_x >= 0 && in_x < in_w) && (in_y >= 0 && in_y < in_h)
                      ? din_batch[in_y * in_w + in_x]
                      : pad_value;
              break;
            case 1:
              in_y = y - pad_top;
              in_x = x - pad_left;
              in_y = std::max(in_y, -in_y);
              in_y = std::min(in_y, 2 * in_h - in_y - 2);
              in_x = std::max(in_x, -in_x);
              in_x = std::min(in_x, 2 * in_w - in_x - 2);
              dout_batch[y * w + x] = din_batch[in_y * in_w + in_x];
              break;
            case 2:
              in_x = std::min(std::max(pad_left, x), in_w + pad_left - 1) -
                     pad_left;
              in_y =
                  std::min(std::max(pad_top, y), in_h + pad_top - 1) - pad_top;
              dout_batch[y * w + x] = din_batch[in_y * in_w + in_x];
              break;
            default:
              LOG(ERROR) << "ERROR: unknown pad mode:" << pad_mode;
          }
        }
      }
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("pad2d");
    op_desc->SetInput("X", {x_});
    op_desc->SetOutput("Out", {out_});
    op_desc->SetAttr("mode", mode_);
    op_desc->SetAttr("pad_value", pad_value_);
    op_desc->SetAttr("paddings", paddings_);
    op_desc->SetAttr("data_format", data_format_);
  }

  void PrepareData() override {
    std::vector<T> dx(dims_.production());
    for (size_t i = 0; i < dims_.production(); i++) {
      dx[i] = static_cast<T>((i % 3) * 1.1f);
    }
    SetCommonTensor(x_, dims_, dx.data());
  }
};

template <class T = float>
void TestPad2d(const Place& place,
               const std::string& alias,
               std::string pad_mode,
               std::vector<int> paddings,
               float pad_value,
               std::string data_format,
               float abs_error = 2e-5) {
  std::unique_ptr<arena::TestCase> tester(new Pad2dComputeTester<T>(
      place, "def", pad_mode, paddings, pad_value, data_format));
  arena::Arena arena(std::move(tester), place, abs_error);
  arena.TestPrecision();
}

TEST(Pad2d, precision) {
  Place place;
  float abs_error = 2e-5;
  std::vector<std::string> pad_mode_list = {"constant", "edge", "reflect"};
#if defined(LITE_WITH_NNADAPTER)
  place = TARGET(kNNAdapter);
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  abs_error = 1e-2;
  pad_mode_list = {"constant"};
#elif defined(NNADAPTER_WITH_INTEL_OPENVINO)
  abs_error = 1e-5;
#elif defined(NNADAPTER_WITH_CAMBRICON_MLU)
  abs_error = 1e-2;
  // TODO(shentanyue): support later
  return;
#elif defined(NNADAPTER_WITH_HUAWEI_KIRIN_NPU)
  abs_error = 1e-2;
  pad_mode_list = {"constant", "reflect"};
#elif defined(NNADAPTER_WITH_QUALCOMM_QNN)
  abs_error = 1e-2;
  pad_mode_list = {"constant", "reflect"};
#else
  return;
#endif
#elif defined(LITE_WITH_XPU)
  place = TARGET(kXPU);
  pad_mode_list = {"constant",
                   "reflect"};  // XPU support constant and reflect now
#elif defined(LITE_WITH_NPU)
  place = TARGET(kNPU);
  abs_error = 1e-2;  // Using fp16 in NPU
#elif defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#elif defined(LITE_WITH_X86)
  place = TARGET(kHost);
#else
  return;
#endif
  std::string data_format = "NCHW";
  for (int pad_top : {0, 1}) {
    for (int pad_bottom : {0, 1}) {
      for (int pad_left : {0, 1}) {
        for (int pad_right : {0, 1}) {
          std::vector<int> paddings{pad_top, pad_bottom, pad_left, pad_right};
          for (std::string pad_mode : pad_mode_list) {
            for (float pad_value : {0.f, 1.f}) {
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
              // Ascend does not support the following scenarios.
              if (std::abs(pad_value - 1) < 1e-6 ||
                  ((pad_top == 1 || pad_bottom == 1) && pad_left == 0 &&
                   pad_right == 0))
                continue;
#elif defined(NNADAPTER_WITH_QUALCOMM_QNN)
              if (pad_mode == "reflect" && pad_left == 0 && pad_right == 0)
                continue;
#endif
              VLOG(5) << "pad param: " << pad_mode << " " << pad_value << " "
                      << paddings[0] << " " << paddings[1] << " " << paddings[2]
                      << " " << paddings[3];
              TestPad2d(place,
                        "def",
                        pad_mode,
                        paddings,
                        pad_value,
                        data_format,
                        abs_error);
            }
          }
        }
      }
    }
  }
}

#if defined(LITE_WITH_ARM) && defined(ENABLE_ARM_FP16)
TEST(Pad2d_fp16, precision) {
  Place place(TARGET(kARM), PRECISION(kFP16));
  float abs_error = 2e-5;
  std::string data_format = "NCHW";
  for (int pad_top : {0, 1}) {
    for (int pad_bottom : {0, 1}) {
      for (int pad_left : {0, 1}) {
        for (int pad_right : {0, 1}) {
          std::vector<int> paddings{pad_top, pad_bottom, pad_left, pad_right};
          for (std::string pad_mode : {"constant", "edge", "reflect"}) {
            for (float pad_value : {0.f, 1.0f}) {
              VLOG(5) << "pad param: " << pad_mode << " " << pad_value << " "
                      << paddings[0] << " " << paddings[1] << " " << paddings[2]
                      << " " << paddings[3];
              TestPad2d<float16_t>(place,
                                   "def",
                                   pad_mode,
                                   paddings,
                                   pad_value,
                                   data_format,
                                   abs_error);
            }
          }
        }
      }
    }
  }
}
#endif

}  // namespace lite
}  // namespace paddle
