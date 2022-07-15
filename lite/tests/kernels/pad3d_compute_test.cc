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

class Pad3dComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string x_ = "X";
  std::string out_ = "Out";
  DDim dims_{{1, 1, 14, 14, 14}};
  std::string mode_{"constant"};
  std::vector<int> paddings_;
  float pad_value_ = 0.f;
  std::string data_format_{"NCDHW"};

 public:
  Pad3dComputeTester(const Place& place,
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
    int out_d = dims_[2] + paddings_[4] + paddings_[5];
    int out_h = dims_[3] + paddings_[2] + paddings_[3];
    int out_w = dims_[4] + paddings_[0] + paddings_[1];
    if (data_format_ == "NDHWC") {
      out_d = dims_[1] + paddings_[4] + paddings_[5];
      out_h = dims_[2] + paddings_[2] + paddings_[3];
      out_w = dims_[3] + paddings_[0] + paddings_[1];
      out->Resize(lite::DDim({dims_[0], out_d, out_h, out_w, dims_[1]}));
    } else {
      out->Resize(lite::DDim({dims_[0], dims_[1], out_d, out_h, out_w}));
    }
    auto* out_data = out->mutable_data<float>();
    auto* x = scope->FindTensor(x_);
    const auto* x_data = x->data<float>();

    auto output_dims = out->dims();
    int n = output_dims[0];
    int c = output_dims[1];
    int d = output_dims[2];
    int h = output_dims[3];
    int w = output_dims[4];
    if (data_format_ == "NDHWC") {
      d = output_dims[1];
      h = output_dims[2];
      w = output_dims[3];
      c = output_dims[4];
    }

    int pad_front = paddings_[4];
    int pad_back = paddings_[5];
    int pad_top = paddings_[2];
    int pad_bottom = paddings_[3];
    int pad_left = paddings_[0];
    int pad_right = paddings_[1];
    int pad_mode;
    if (mode_ == "constant") {
      pad_mode = 0;
    } else if (mode_ == "reflect") {
      pad_mode = 1;
    } else if (mode_ == "replicate") {
      pad_mode = 2;
    } else if (mode_ == "circular") {
      pad_mode = 3;
    } else {
      LOG(FATAL) << "Unknown mode type: " << mode_;
    }
    float pad_value = pad_value_;

    int in_w = w - pad_left - pad_right;
    int in_h = h - pad_bottom - pad_top;
    int in_d = d - pad_front - pad_back;
    int in_size = in_w * in_h;
    int out_size = w * h;
    int spatial_size_out = out_size * d;
    int spatial_size_in = in_size * in_d;
    if (data_format_ == "NCDHW") {
      for (int i = 0; i < n * c; ++i) {
        const float* din_batch = x_data + i * spatial_size_in;
        float* dout_batch = out_data + i * spatial_size_out;
        int in_z = 0;
        int in_x = 0;
        int in_y = 0;
        for (int z = 0; z < d; z++) {
          for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
              switch (pad_mode) {
                case 0:
                  in_z = z - pad_front;
                  in_y = y - pad_top;
                  in_x = x - pad_left;
                  dout_batch[z * out_size + y * w + x] =
                      (in_x >= 0 && in_x < in_w) &&
                              (in_y >= 0 && in_y < in_h) &&
                              (in_z >= 0 && in_z < in_d)
                          ? din_batch[in_z * in_size + in_y * in_w + in_x]
                          : pad_value;
                  break;
                case 1:
                  in_z = z - pad_front;
                  in_y = y - pad_top;
                  in_x = x - pad_left;
                  in_z = std::max(in_z, -in_z);
                  in_z = std::min(in_z, 2 * in_d - in_z - 2);
                  in_y = std::max(in_y, -in_y);
                  in_y = std::min(in_y, 2 * in_h - in_y - 2);
                  in_x = std::max(in_x, -in_x);
                  in_x = std::min(in_x, 2 * in_w - in_x - 2);
                  dout_batch[z * out_size + y * w + x] =
                      din_batch[in_z * in_size + in_y * in_w + in_x];
                  break;
                case 2:
                  in_z = std::min(in_d - 1, std::max(z - pad_front, 0));
                  in_y = std::min(in_h - 1, std::max(y - pad_top, 0));
                  in_x = std::min(in_w - 1, std::max(x - pad_left, 0));
                  dout_batch[z * out_size + y * w + x] =
                      din_batch[in_z * in_size + in_y * in_w + in_x];
                  break;
                case 3:
                  in_z = ((z - pad_front) % in_d + in_d) % in_d;
                  in_y = ((y - pad_top) % in_h + in_h) % in_h;
                  in_x = ((x - pad_left) % in_w + in_w) % in_w;
                  dout_batch[z * out_size + y * w + x] =
                      din_batch[in_z * in_size + in_y * in_w + in_x];
                  break;
                default:
                  LOG(ERROR) << "ERROR: unknown pad mode:" << pad_mode;
              }
            }
          }
        }
      }
    } else {  // NDHWC
      int c_in_size = spatial_size_in * c;
      int c_out_size = spatial_size_out * c;
      for (int i = 0; i < n; ++i) {
        const float* din_batch = x_data + i * c_in_size;
        float* dout_batch = out_data + i * c_out_size;
        int in_z = 0;
        int in_x = 0;
        int in_y = 0;
        int in_index = 0;
        int out_index = 0;
        for (int z = 0; z < d; z++) {
          for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
              switch (pad_mode) {
                case 0:
                  in_z = z - pad_front;
                  in_y = y - pad_top;
                  in_x = x - pad_left;
                  in_index = (in_z * in_size + in_y * in_w + in_x) * c;
                  out_index = (z * out_size + y * w + x) * c;
                  if ((in_x >= 0 && in_x < in_w) &&
                      (in_y >= 0 && in_y < in_h) &&
                      (in_z >= 0 && in_z < in_d)) {
                    for (int j = 0; j < c; j++) {
                      dout_batch[out_index + j] = din_batch[in_index + j];
                    }
                  } else {
                    for (int j = 0; j < c; j++) {
                      dout_batch[out_index + j] = pad_value;
                    }
                  }
                  break;
                case 1:
                  in_z = z - pad_front;
                  in_y = y - pad_top;
                  in_x = x - pad_left;
                  in_z = std::max(in_z, -in_z);
                  in_z = std::min(in_z, 2 * in_d - in_z - 2);
                  in_y = std::max(in_y, -in_y);
                  in_y = std::min(in_y, 2 * in_h - in_y - 2);
                  in_x = std::max(in_x, -in_x);
                  in_x = std::min(in_x, 2 * in_w - in_x - 2);
                  in_index = (in_z * in_size + in_y * in_w + in_x) * c;
                  out_index = (z * out_size + y * w + x) * c;
                  for (int j = 0; j < c; j++) {
                    dout_batch[out_index + j] = din_batch[in_index + j];
                  }
                  break;
                case 2:
                  in_z = std::min(in_d - 1, std::max(z - pad_front, 0));
                  in_y = std::min(in_h - 1, std::max(y - pad_top, 0));
                  in_x = std::min(in_w - 1, std::max(x - pad_left, 0));
                  in_index = (in_z * in_size + in_y * in_w + in_x) * c;
                  out_index = (z * out_size + y * w + x) * c;
                  for (int j = 0; j < c; j++) {
                    dout_batch[out_index + j] = din_batch[in_index + j];
                  }
                  break;
                case 3:
                  in_z = ((z - pad_front) % in_d + in_d) % in_d;
                  in_y = ((y - pad_top) % in_h + in_h) % in_h;
                  in_x = ((x - pad_left) % in_w + in_w) % in_w;
                  in_index = (in_z * in_size + in_y * in_w + in_x) * c;
                  out_index = (z * out_size + y * w + x) * c;
                  for (int j = 0; j < c; j++) {
                    dout_batch[out_index + j] = din_batch[in_index + j];
                  }
                  break;
                default:
                  LOG(ERROR) << "ERROR: unknown pad mode:" << pad_mode;
              }
            }
          }
        }
      }
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("pad3d");
    op_desc->SetInput("X", {x_});
    op_desc->SetOutput("Out", {out_});
    op_desc->SetAttr("mode", mode_);
    op_desc->SetAttr("value", pad_value_);
    op_desc->SetAttr("paddings", paddings_);
    op_desc->SetAttr("data_format", data_format_);
  }

  void PrepareData() override {
    std::vector<float> x(dims_.production());
    fill_data_rand(x.data(), -1.f, 1.f, dims_.production());
    SetCommonTensor(x_, dims_, x.data());
  }
};

void TestPad3d(const Place& place, float abs_error = 2e-5) {
  std::string data_format = "NCDHW";
  const float pad_value = 0.f;
  for (int pad_top : {0, 1}) {
    for (int pad_bottom : {0, 1}) {
      for (int pad_left : {0, 1}) {
        for (int pad_right : {0, 1}) {
          for (int pad_front : {0, 1}) {
            for (int pad_back : {0, 1}) {
              std::vector<int> paddings{pad_left,
                                        pad_right,
                                        pad_top,
                                        pad_bottom,
                                        pad_front,
                                        pad_back};
              for (std::string pad_mode :
                   {"constant", "reflect", "replicate", "circular"}) {
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
                if (pad_mode != "constant" || pad_front != pad_back) continue;
#endif
#if defined(NNADAPTER_WITH_INTEL_OPENVINO)
                if (pad_mode == "circular") continue;
#endif
                VLOG(4) << "pad3d pad_mode: " << pad_mode
                        << ", pad_val: " << pad_value
                        << ", padding: " << paddings[0] << ", " << paddings[1]
                        << ", " << paddings[2] << ", " << paddings[3] << ", "
                        << paddings[4] << ", " << paddings[5];
                std::unique_ptr<arena::TestCase> tester(new Pad3dComputeTester(
                    place, "def", pad_mode, paddings, pad_value, data_format));
                arena::Arena arena(std::move(tester), place, abs_error);
                arena.TestPrecision();
              }
            }
          }
        }
      }
    }
  }
}

TEST(pad3d, precision) {
  Place place;
  float abs_error = 2e-5;
#if defined(LITE_WITH_NNADAPTER)
  place = TARGET(kNNAdapter);
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  abs_error = 1e-2;
#elif defined(NNADAPTER_WITH_INTEL_OPENVINO)
  abs_error = 1e-5;
#elif defined(NNADAPTER_WITH_CAMBRICON_MLU)
  abs_error = 1e-2;
  // TODO(shentanyue): support later
  return;
#else
  return;
#endif
#elif defined(LITE_WITH_ARM)
  place = TARGET(kHost);
#else
  return;
#endif

  TestPad3d(place, abs_error);
}

}  // namespace lite
}  // namespace paddle
